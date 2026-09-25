"""
Refinamento musical/auditivo — limpeza determinística (stdlib apenas).

Perfis:
- "natural" (default na API/frontend): partitura limpa e tocável. Grade
  preferencial 1/8 com 1/16 só sob evidência, triagem de notas curtas
  (MERGE/PRESERVAR/REMOVER), smoothing de oitavas, remoção de outliers,
  monofonia com vencedor por duração×força, acordes de `other` até 4 notas,
  harmonia por janela de 1 beat com histerese, redução rítmica e sustains.
- "detailed": preserva o comportamento das Etapas 6/7 (bit-idêntico) para
  regressão e comparação antes/depois.

Nada aqui é aleatório; toda decisão é documentada e testável.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

CLEANUP_PROFILES = ["natural", "detailed"]
DEFAULT_CLEANUP_PROFILE = "natural"

# Janela harmônica (beats): pitch classes agregadas por peso de duração.
HARMONY_WINDOW_BEATS = 1.0
# Janela do perfil natural no arranjo (2 beats: ignora micro-notas).
NATURAL_HARMONY_WINDOW = 2.0
# Peso mínimo (fração da janela) para um pc entrar na harmonia predominante.
HARMONY_MIN_WEIGHT = 0.3
# Margem de troca harmônica (histerese): só troca se o novo custo for
# melhor que o atual por ao menos esta margem.
HARMONY_CHANGE_MARGIN = 8.0
# Duração mínima de uma harmonia escolhida (beats) no natural.
MIN_HARMONY_BEATS = 1.0
# Gap máximo para fundir mesma nota/função (sustain over retrigger).
HARMONY_MERGE_GAP = 0.25
# Acorde natural: teto e alvo preferencial.
MAX_CHORD_NATURAL = 4
PREFER_CHORD_NATURAL = 3
# Nota fantasma (natural): curta + fraca + distante das vizinhas.
GHOST_MAX_BEATS = 0.15
GHOST_MAX_STRENGTH = 0.25
GHOST_MIN_DISTANCE = 6
# Fronteira de frase (beats de silêncio).
PHRASE_GAP_BEATS = 0.75
# Salto grande (semitons) para métricas e penalidades.
LARGE_LEAP = 7
VERY_LARGE_LEAP = 12


def validate_cleanup_profile(value: Any) -> str:
    """Normaliza/lança para cleanup_profile (400 na API)."""
    v = str(value or "").strip().lower()
    if v not in CLEANUP_PROFILES:
        raise ValueError(
            f"cleanup_profile inválido. Permitidos: {', '.join(CLEANUP_PROFILES)}.")
    return v


def event_strength(event: Dict[str, Any], default: float = 0.5) -> float:
    """Máximo entre amplitude/confidence/strength (0..1)."""
    vals = []
    for field in ("amplitude", "confidence", "strength"):
        try:
            if event.get(field) is not None:
                vals.append(float(event[field]))
        except (TypeError, ValueError):
            continue
    if not vals:
        return default
    return max(0.0, min(1.0, max(vals)))


# ---------------------------------------------------------------------------
# Quantização adaptativa
# ---------------------------------------------------------------------------

def _q(value: float, grid: float) -> float:
    return round(round(float(value) / grid) * grid, 6)


def adaptive_quantize(
    notes_beats: List[Dict[str, Any]],
    base_grid: float,
    profile: str = "detailed",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Quantiza inícios/fins. Natural prefere 1/8; 1/16 só com evidência.

    Evidência: outro onset da mesma linha no mesmo slot de 1/8, separado por
    >= 0.2 beat (passagem rápida real). Jitter (0.08–0.15 beat) alinha no
    grid maior. Duração mínima = 1 passo da grade usada na nota.
    Detailed: grade base exata (comportamento Etapa 6).
    """
    stats = {"rhythmic_simplifications": 0, "kept_16ths": 0}
    if profile != "natural":
        out = []
        for n in notes_beats:
            qs = _q(n["start"], base_grid)
            qe = _q(n["end"], base_grid)
            qs = max(0.0, qs)
            if qe <= qs:
                qe = qs + base_grid
            out.append({"start": qs, "end": qe, "pitch": int(n["pitch"]),
                        "velocity": int(n.get("velocity", 64)),
                        "strength": float(n.get("strength", 0.5))})
        return out, stats
    # Natural: grade preferencial 1/8 (1/32 nunca no natural).
    pref = 0.5 if base_grid <= 0.25 else base_grid
    fine = 0.25
    starts = sorted(float(n["start"]) for n in notes_beats)
    out = []
    for n in notes_beats:
        s, e = float(n["start"]), float(n["end"])
        slot = int(s // 0.5) if s >= 0 else -1
        # Exclui a própria nota da evidência (compara identidade por par).
        evidence = any(
            int(t // 0.5) == slot and abs(t - s) >= 0.2 and abs(t - s) > 1e-9
            for t in starts
        )
        use_fine = (base_grid <= 0.25 and evidence)
        g = fine if use_fine else pref
        if use_fine:
            stats["kept_16ths"] += 1
        qs = max(0.0, _q(s, g))
        qe = _q(e, g)
        if qe <= qs:
            qe = qs + g
        # Conta simplificação: difere da grade base?
        base_qs = max(0.0, _q(s, base_grid if base_grid >= 0.125 else 0.125))
        if abs(qs - base_qs) > 1e-9 and base_grid <= 0.25:
            stats["rhythmic_simplifications"] += 1
        out.append({"start": qs, "end": qe, "pitch": int(n["pitch"]),
                    "velocity": int(n.get("velocity", 64)),
                    "strength": float(n.get("strength", 0.5))})
    return out, stats


# ---------------------------------------------------------------------------
# Triagem de notas curtas: MERGE / PRESERVAR / REMOVER
# ---------------------------------------------------------------------------

def triage_short_notes(
    notes_beats: List[Dict[str, Any]],
    grid: float,
    profile: str = "detailed",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Decide por evento curto (dur < grid): une, preserva ou remove.

    - Mesma altura do vizinho com gap pequeno -> MERGE (estende vizinho).
    - Fantasma (curta+fraca+distante) -> REMOVER.
    - Passagem por graus conjuntos com força/ritmo OK -> PRESERVAR.
    - Resto -> PRESERVAR (conservador).
    Detailed: passthrough.
    """
    stats = {"short_notes_merged": 0, "short_notes_removed": 0, "short_notes_kept": 0}
    if profile != "natural" or not notes_beats:
        return [dict(n) for n in notes_beats], stats
    ordered = sorted(notes_beats, key=lambda n: (float(n["start"]), float(n["end"])))
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(ordered):
        n = dict(ordered[i])
        dur = float(n["end"]) - float(n["start"])
        if dur >= grid:
            out.append(n)
            i += 1
            continue
        prev = out[-1] if out else None
        nxt = ordered[i + 1] if i + 1 < len(ordered) else None
        strength = float(n.get("strength", 0.5))
        pitch = int(n["pitch"])
        gap_prev = float(n["start"]) - float(prev["end"]) if prev else None
        gap_next = (float(nxt["start"]) - float(n["end"])) if nxt else None
        # MERGE: mesma altura, gap pequeno.
        merged = False
        if prev is not None and int(prev["pitch"]) == pitch \
                and gap_prev is not None and -0.5 <= gap_prev <= 0.3:
            prev["end"] = round(max(float(prev["end"]), float(n["end"])), 6)
            stats["short_notes_merged"] += 1
            merged = True
        elif nxt is not None and int(nxt["pitch"]) == pitch \
                and gap_next is not None and -0.5 <= gap_next <= 0.3:
            nxt["start"] = round(min(float(nxt["start"]), float(n["start"])), 6)
            stats["short_notes_merged"] += 1
            merged = True
        if merged:
            i += 1
            continue
        # REMOVER: fantasma curta+fraca+distante.
        dists = []
        if prev is not None:
            dists.append(abs(pitch - int(prev["pitch"])))
        if nxt is not None:
            dists.append(abs(pitch - int(nxt["pitch"])))
        far = min(dists) >= GHOST_MIN_DISTANCE if dists else strength < 0.15 and dur < 0.1
        if dur <= GHOST_MAX_BEATS and strength < GHOST_MAX_STRENGTH and far:
            stats["short_notes_removed"] += 1
            i += 1
            continue
        # PRESERVAR (passagem ou conservador).
        stats["short_notes_kept"] += 1
        out.append(n)
        i += 1
    return out, stats


# ---------------------------------------------------------------------------
# Smoothing melódico (correção de oitava) + outliers
# ---------------------------------------------------------------------------

def smooth_melody(
    notes: List[Dict[str, Any]],
    profile: str = "detailed",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Corrige spike isolado deslocado ±12/±24 quando o ganho é claro (>=12).

    Só tenta se ambos os saltos >= 12 e os vizinhos próximos (|a-c| <= 7).
    Progressão legítima (ex. C4 C5 C6) nunca é tocada. Detailed: passthrough.
    """
    stats = {"octave_corrections": 0}
    if profile != "natural" or len(notes) < 3:
        return [dict(n) for n in notes], stats
    out = [dict(n) for n in notes]
    for i in range(1, len(out) - 1):
        a = int(out[i - 1]["pitch"])
        b = int(out[i]["pitch"])
        c = int(out[i + 1]["pitch"])
        if min(abs(b - a), abs(b - c)) < VERY_LARGE_LEAP:
            continue
        if abs(c - a) > 7:
            continue
        current = abs(b - a) + abs(c - b)
        best, best_cost = b, current
        for cand in (b - 24, b - 12, b + 12, b + 24):
            if 0 <= cand <= 127:
                cost = abs(cand - a) + abs(c - cand)
                if cost < best_cost:
                    best, best_cost = cand, cost
        if best != b and current - best_cost >= VERY_LARGE_LEAP:
            out[i]["pitch"] = best
            stats["octave_corrections"] += 1
    return out, stats


def remove_outliers(
    notes: List[Dict[str, Any]],
    profile: str = "detailed",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Remove evento isolado muito distante + curto + fraco.

    Nota longa/forte nunca sai só por ser aguda/grave. Detailed: passthrough.
    """
    stats = {"outliers_removed": 0}
    if profile != "natural" or not notes:
        return [dict(n) for n in notes], stats
    ordered = sorted(notes, key=lambda n: (float(n["start"]), float(n["end"])))
    keep = [True] * len(ordered)
    for i, n in enumerate(ordered):
        dur = float(n["end"]) - float(n["start"])
        strength = float(n.get("strength", 0.5))
        pitch = int(n["pitch"])
        dists = []
        if i > 0:
            dists.append(abs(pitch - int(ordered[i - 1]["pitch"])))
        if i < len(ordered) - 1:
            dists.append(abs(pitch - int(ordered[i + 1]["pitch"])))
        if not dists:
            continue
        far = min(dists) >= VERY_LARGE_LEAP
        if i == 0 or i == len(ordered) - 1:
            far = min(dists) >= 15
        if far and dur <= 0.25 and strength < 0.3:
            keep[i] = False
            stats["outliers_removed"] += 1
    return [n for n, k in zip(ordered, keep) if k], stats


# ---------------------------------------------------------------------------
# Monofonia com vencedor (vocals/bass, perfil natural)
# ---------------------------------------------------------------------------

def _mono_score(n: Dict[str, Any], continuity: bool = False) -> float:
    dur = max(0.0, float(n["end"]) - float(n["start"]))
    s = dur * float(n.get("strength", 0.5))
    if continuity:
        s += 0.3
    return s


def resolve_mono_natural(
    notes_beats: List[Dict[str, Any]],
    grid: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Overlap: vence maior duração×força (+continuidade); perdedor aparado.

    Onsets quase simultâneos (< 0.06 beat, alturas distintas): fica a mais
    forte se razão >= 2, senão ambas (aparadas). Sem apagões silenciosos.
    """
    stats = {"overlaps_resolved": 0, "large_overlaps": 0, "notes_dropped": 0}
    if not notes_beats:
        return [], stats
    ordered = sorted(notes_beats, key=lambda n: (float(n["start"]), float(n["end"])))
    result: List[Dict[str, Any]] = []
    for note in ordered:
        cur = dict(note)
        if not result or float(cur["start"]) >= float(result[-1]["end"]) - 1e-9:
            result.append(cur)
            continue
        prev = result[-1]
        overlap = float(prev["end"]) - float(cur["start"])
        stats["overlaps_resolved"] += 1
        if overlap > 1.0:
            stats["large_overlaps"] += 1
        simultaneous = abs(float(cur["start"]) - float(prev["start"])) < 0.06
        if simultaneous and int(cur["pitch"]) != int(prev["pitch"]):
            sp = float(prev.get("strength", 0.5)) or 1e-6
            sc = float(cur.get("strength", 0.5))
            if sc >= 2 * sp:
                result[-1] = cur  # atual vence; anterior descartada
                stats["notes_dropped"] += 1
            elif sp >= 2 * sc:
                stats["notes_dropped"] += 1  # anterior vence; descarta atual
            else:
                prev["end"] = float(cur["start"])  # empate: apara anterior
                result.append(cur)
            continue
        s_prev = _mono_score(prev)
        s_cur = _mono_score(cur)
        if s_cur > s_prev:
            prev["end"] = float(cur["start"])
            if float(prev["end"]) <= float(prev["start"]) + 1e-9:
                result[-1] = cur  # anterior degenerou; atual assume
                stats["notes_dropped"] += 1
            else:
                result.append(cur)
        else:
            cur["start"] = float(prev["end"])
            if float(cur["end"]) <= float(cur["start"]) + 1e-9:
                stats["notes_dropped"] += 1
            else:
                result.append(cur)
    # Normaliza degeneradas e ordena.
    final = [n for n in result if float(n["end"]) > float(n["start"]) + 1e-9]
    stats["notes_dropped"] += len(result) - len(final)
    final.sort(key=lambda n: (float(n["start"]), float(n["end"])))
    return final, stats


# ---------------------------------------------------------------------------
# Acordes de other: teto + oitavas
# ---------------------------------------------------------------------------

def simplify_chord(
    pitches: List[int],
    profile: str = "detailed",
    max_notes: int = MAX_CHORD_NATURAL,
) -> Tuple[List[int], Dict[str, int]]:
    """Natural: <= max_notes (alvo 3): baixo + topo + internas por variedade.

    Remove duplicatas, depois oitavas excedentes preservando baixo, topo e
    1–2 internas harmonicamente úteis (maior distância mínima; 4ª nota só
    com pitch class nova). Detailed: só dedupe (Etapa 6).
    """
    stats = {"chords_simplified": 0, "duplicate_octaves_removed": 0}
    uniq = sorted(set(int(p) for p in pitches))
    if profile != "natural":
        return uniq, stats
    if len(uniq) <= max_notes:
        return uniq, stats
    stats["chords_simplified"] = 1
    kept = [uniq[0], uniq[-1]]
    kept_pcs = {uniq[0] % 12, uniq[-1] % 12}
    middle = [p for p in uniq[1:-1]]
    # 3ª nota: maior distância mínima ao conjunto.
    while len(kept) < PREFER_CHORD_NATURAL and middle:
        best = max(middle, key=lambda p: min(abs(p - k) for k in kept))
        kept.append(best)
        kept_pcs.add(best % 12)
        middle.remove(best)
    # 4ª nota: só com pitch class nova (mais grave primeiro = fundamento).
    if len(kept) < max_notes:
        for p in sorted(middle):
            if p % 12 not in kept_pcs:
                kept.append(p)
                kept_pcs.add(p % 12)
                break
    removed = len(uniq) - len(kept)
    stats["duplicate_octaves_removed"] = removed
    return sorted(kept), stats


# ---------------------------------------------------------------------------
# Harmonia por janela + redução rítmica + sustains + blocos + frases
# ---------------------------------------------------------------------------

def get_window_harmony(
    other_items: List[Dict[str, Any]],
    start: float,
    window: float = HARMONY_WINDOW_BEATS,
) -> List[int]:
    """Pitch classes predominantes em [start, start+window), por peso de duração.

    Só entra pc com peso >= 30% da janela: micro-eventos não viram a harmonia.
    Janela vazia -> instante do onset (fallback); ainda vazio -> [].
    """
    weights = get_window_harmony_weights(other_items, start, window)
    if not weights:
        pcs = set()
        for it in other_items or []:
            if float(it["start"]) <= start < float(it["end"]):
                for p in it["pitches"]:
                    pcs.add(int(p) % 12)
        return sorted(pcs)
    return sorted(pc for pc, w in weights.items() if w >= HARMONY_MIN_WEIGHT * window)


def get_window_harmony_weights(
    other_items: List[Dict[str, Any]],
    start: float,
    window: float = HARMONY_WINDOW_BEATS,
    strengths: Optional[Dict[int, float]] = None,
    bass_pcs: Optional[List[int]] = None,
    strong_beat: bool = False,
) -> Dict[int, float]:
    """Peso por pc na janela: duração × força, com bônus de baixo e downbeat.

    - strengths: multiplicador por pitch (0.5 + strength) quando disponível;
    - bass_pcs: pcs do baixo na janela ganham ×1.5 (referência de raiz);
    - strong_beat: pcs soando no instante inicial ganham ×1.25 (beat 1/3).
    Determinístico; sem estes sinais equivale ao peso puro de duração.
    """
    weights: Dict[int, float] = {}
    for it in other_items or []:
        s, e = float(it["start"]), float(it["end"])
        ov = max(0.0, min(e, start + window) - max(s, start))
        if ov > 1e-9:
            for p in it["pitches"]:
                pc = int(p) % 12
                w = ov
                if strengths:
                    w *= 0.5 + float(strengths.get(int(p), 0.5))
                if bass_pcs and pc in bass_pcs:
                    w *= 1.5
                if strong_beat and s <= start + 1e-9:
                    w *= 1.25
                weights[pc] = weights.get(pc, 0.0) + w
    return weights


def reduce_chord_voicing(
    pitches: List[int],
    profile: str = "detailed",
    max_notes: int = MAX_CHORD_NATURAL,
) -> Tuple[List[int], Dict[str, int]]:
    """Voicing simples (natural): baixo + terça/quinta + topo (alvo 3, teto 4).

    Remove: duplicatas, clusters de semitom (fica o grave do par, salvo topo),
    oitavas excedentes. 4ª nota só com pitch class nova e estrutural
    (quinta ausente ou fundamental do baixo). Detailed: só dedupe.
    """
    stats = {"chords_simplified": 0, "duplicate_octaves_removed": 0}
    uniq = sorted(set(int(p) for p in pitches))
    if profile != "natural":
        return uniq, stats
    if len(uniq) <= PREFER_CHORD_NATURAL:
        # Mesmo pequeno: limpa cluster de semitom.
        cleaned = _drop_semitone_clusters(uniq)
        stats["duplicate_octaves_removed"] = len(uniq) - len(cleaned)
        if len(cleaned) != len(uniq):
            stats["chords_simplified"] = 1
        return cleaned, stats
    stats["chords_simplified"] = 1
    no_cluster = _drop_semitone_clusters(uniq)
    stats["duplicate_octaves_removed"] += len(uniq) - len(no_cluster)
    bass, top = no_cluster[0], no_cluster[-1]
    rest = [p for p in no_cluster[1:-1]]
    third = next((p for p in rest if (p - bass) % 12 in (3, 4)), None)
    fifth = next((p for p in rest if (p - bass) % 12 == 7), None)
    # Prioridade: grave + terça/estrutural + quinta/topo (alvo 3).
    kept = [bass]
    if third is not None:
        kept.append(third)
    if fifth is not None and fifth not in kept and len(kept) < PREFER_CHORD_NATURAL:
        kept.append(fifth)
    if top not in kept:
        if len(kept) >= PREFER_CHORD_NATURAL:
            # Troca a quinta pelo topo (topo = linha superior/melodia).
            kept = [k for k in kept if k != fifth] if fifth in kept else kept[:-1]
        kept.append(top)
    if len(kept) < PREFER_CHORD_NATURAL:
        # Sem terça/quinta: interna mais próxima da quinta ideal.
        cands = [p for p in rest if p not in kept]
        if cands:
            kept.append(min(cands, key=lambda p: (abs((p - bass) - 7), p)))
    kept_pcs = {p % 12 for p in kept}
    # 4ª nota: só pc nova e estrutural (quinta/fundamental ausente).
    if len(kept) < max_notes:
        for p in sorted(rest):
            if p in kept or p % 12 in kept_pcs:
                continue
            if (p - bass) % 12 in (7, 0, 5):
                kept.append(p)
                kept_pcs.add(p % 12)
                break
    stats["duplicate_octaves_removed"] += len(uniq) - len(kept) - stats["duplicate_octaves_removed"]
    return sorted(kept), stats


def _drop_semitone_clusters(pitches: List[int]) -> List[int]:
    """De cada par a 1 semitom, mantém o grave (salvo se for o topo)."""
    if len(pitches) < 2:
        return list(pitches)
    out = [pitches[0]]
    for prev, cur in zip(pitches, pitches[1:]):
        if cur - prev == 1 and cur != pitches[-1]:
            continue  # cluster: descarta o agudo do par
        out.append(cur)
    # Garante topo preservado.
    if out[-1] != pitches[-1]:
        out.append(pitches[-1])
    return out


def merge_adjacent_harmony_notes(
    line: List[Dict[str, Any]],
    max_gap: float = HARMONY_MERGE_GAP,
) -> Tuple[List[Dict[str, Any]], int]:
    """Funde mesma nota/função separada por gap <= max_gap (sustain > retrigger).

    Mesma altura com pausa curta no meio vira uma nota só. Conta fusões.
    """
    merged = 0
    if not line:
        return [], merged
    out = [dict(line[0])]
    for n in line[1:]:
        gap = float(n["start"]) - float(out[-1]["end"])
        if int(n["pitch"]) == int(out[-1]["pitch"]) and 0 <= gap <= max_gap + 1e-9:
            out[-1]["end"] = round(float(n["end"]), 6)
            merged += 1
        else:
            out.append(dict(n))
    return out, merged


def reduce_harmony_rhythm(
    line: List[Dict[str, Any]],
    min_dur: float = 0.5,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Harmonia não acompanha semicolcheias: nota < min_dur é absorvida pela
    anterior (sustain); primeira curta estende a si mesma. Conta simplificações.
    """
    stats = {"rhythmic_simplifications": 0, "sustains_merged": 0}
    if not line:
        return [], stats
    out = [dict(line[0])]
    if float(out[0]["end"]) - float(out[0]["start"]) < min_dur:
        out[0]["end"] = round(float(out[0]["start"]) + min_dur, 6)
        stats["rhythmic_simplifications"] += 1
    for n in line[1:]:
        dur = float(n["end"]) - float(n["start"])
        if dur < min_dur:
            out[-1]["end"] = round(max(float(out[-1]["end"]), float(n["end"])), 6)
            stats["rhythmic_simplifications"] += 1
            if int(out[-1]["pitch"]) == int(n["pitch"]):
                stats["sustains_merged"] += 1
        else:
            if int(out[-1]["pitch"]) == int(n["pitch"]) \
                    and abs(float(out[-1]["end"]) - float(n["start"])) < 1e-6:
                out[-1]["end"] = round(float(n["end"]), 6)
                stats["sustains_merged"] += 1
            else:
                out.append(dict(n))
    return out, stats


def sustain_merge(
    line: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """Une notas adjacentes de mesma altura (G4 G4 G4 -> G4 sustentado)."""
    merged = 0
    if not line:
        return [], merged
    out = [dict(line[0])]
    for n in line[1:]:
        if int(n["pitch"]) == int(out[-1]["pitch"]) \
                and float(n["start"]) <= float(out[-1]["end"]) + 1e-6:
            out[-1]["end"] = round(max(float(out[-1]["end"]), float(n["end"])), 6)
            merged += 1
        else:
            out.append(dict(n))
    return out, merged


def phrase_boundaries(line: List[Dict[str, Any]], gap: float = PHRASE_GAP_BEATS) -> List[int]:
    """Índices que iniciam nova frase (silêncio >= gap antes)."""
    bounds = [0]
    for i in range(1, len(line)):
        if float(line[i]["start"]) - float(line[i - 1]["end"]) >= gap:
            bounds.append(i)
    return bounds


def block_rhythm(
    harmony_lines: Dict[str, List[Dict[str, Any]]],
    slot: float = 0.5,
) -> Tuple[Dict[str, List[Dict[str, Any]]], int]:
    """Sopros em bloco (3+ ventos): harmonias compartilham ritmo por slots.

    Por slot: primeira altura soando (ou sustain anterior se pausa curta,
    resto se silêncio > 1 beat). Mesma altura adjacente une (ties no worker).
    """
    if not harmony_lines:
        return {}, 0
    total_end = 0.0
    for line in harmony_lines.values():
        for n in line:
            total_end = max(total_end, float(n["end"]))
    n_slots = max(1, int(round(total_end / slot)))
    aligned: Dict[str, List[Dict[str, Any]]] = {}
    for inst_id, line in harmony_lines.items():
        seq = sorted(line, key=lambda n: float(n["start"]))
        out: List[Dict[str, Any]] = []
        last_pitch: Optional[int] = None
        last_end = 0.0
        for k in range(n_slots):
            s0, s1 = round(k * slot, 6), round((k + 1) * slot, 6)
            sounding = [int(n["pitch"]) for n in seq
                        if float(n["start"]) < s1 - 1e-9 and float(n["end"]) > s0 + 1e-9]
            if sounding:
                pitch = sounding[0]
            elif last_pitch is not None and s0 - last_end <= 1.0:
                pitch = last_pitch  # sustain sobre pausa curta
            else:
                pitch = None
            if pitch is None:
                last_pitch = None
                continue
            if out and int(out[-1]["pitch"]) == pitch \
                    and abs(float(out[-1]["end"]) - s0) < 1e-6:
                out[-1]["end"] = s1
            else:
                out.append({"start": s0, "end": s1, "pitch": pitch,
                            "velocity": int(seq[0].get("velocity", 64)) if seq else 64})
            last_pitch, last_end = pitch, s1
        aligned[inst_id] = out
    return aligned, n_slots


# ---------------------------------------------------------------------------
# Respiração com prioridades (natural)
# ---------------------------------------------------------------------------

def apply_breathing_v2(
    line: List[Dict[str, Any]],
    beats_per_bar: float = 4.0,
    max_phrase: float = 16.0,
    breath: float = 0.5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Respiro em ponto seguro, por prioridade: pausa existente > fim de
    frase > nota repetida > final de compasso > nota mais longa. Nunca
    abaixo de 0.5 beat restante. Detailed usa a heurística antiga."""
    stats = {"breath_adjustments": 0}
    if not line:
        return [], stats
    out = [dict(m) for m in line]
    bounds = set(phrase_boundaries(out))
    since_rest = 0.0
    i = 0
    while i < len(out):
        n = out[i]
        gap_before = float(n["start"]) - (float(out[i - 1]["end"]) if i else 0.0)
        if gap_before >= 0.5:
            since_rest = 0.0
        since_rest += float(n["end"]) - float(n["start"])
        if since_rest >= max_phrase:
            cut = _find_breath_cut(out, i, bounds, beats_per_bar, breath)
            if cut is not None:
                cut["end"] = round(float(cut["end"]) - breath, 6)
                stats["breath_adjustments"] += 1
                since_rest = 0.0
            else:
                since_rest = 0.0  # sem ponto seguro; não corta importante
        i += 1
    return out, stats


def _find_breath_cut(out: List[Dict[str, Any]], upto: int,
                     bounds: set, beats_per_bar: float, breath: float) -> Optional[Dict]:
    window = out[max(0, upto - 7): upto + 1]

    def ok(n: Dict[str, Any]) -> bool:
        return float(n["end"]) - float(n["start"]) > breath + 0.5

    cands = [n for n in window if ok(n)]
    if not cands:
        return None
    # 1) fim de frase (pausa existente adiante) / 2) nota repetida /
    # 3) final de compasso / 4) mais longa.
    idx = {id(n): k for k, n in enumerate(out)}
    for n in cands:
        k = idx[id(n)]
        if k + 1 < len(out) and float(out[k + 1]["start"]) - float(n["end"]) >= 0.5:
            return n
    for n in cands:
        k = idx[id(n)]
        if k in bounds and k > 0:
            return out[k - 1] if ok(out[k - 1]) and out[k - 1] in cands else n
    for n in cands:
        if abs(float(n["end"]) % beats_per_bar) < 1e-6 or \
                abs(float(n["start"]) % beats_per_bar) < 1e-6:
            return n
    for n in cands:
        k = idx[id(n)]
        if k + 1 < len(out) and int(out[k + 1]["pitch"]) == int(n["pitch"]):
            return n
    return max(cands, key=lambda w: float(w["end"]) - float(w["start"]))


# ---------------------------------------------------------------------------
# Métricas: intervalos, mudanças, clutter
# ---------------------------------------------------------------------------

def line_stats(line: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Estatísticas de uma linha: notas, saltos >7, intervalo médio, mudanças."""
    pitches = [int(n["pitch"]) for n in sorted(line, key=lambda n: float(n["start"]))]
    intervals = [abs(b - a) for a, b in zip(pitches, pitches[1:])]
    return {
        "notes": len(pitches),
        "large_leaps": sum(1 for iv in intervals if iv > LARGE_LEAP),
        "very_large_leaps": sum(1 for iv in intervals if iv > VERY_LARGE_LEAP),
        "average_interval": round(sum(intervals) / len(intervals), 3) if intervals else 0.0,
        "harmonic_changes": sum(1 for a, b in zip(pitches, pitches[1:]) if b != a),
    }


def clutter_score(items: List[Dict[str, Any]], total_beats: float) -> Dict[str, Any]:
    """Métrica de bagunça (debug/testes): densidade + curtas + saltos + trocas.

    score = 4*density + 3*short_ratio + 2*leap_ratio + 1*change_rate.
    Menor = mais limpo. Determinístico.
    """
    seq = sorted(items, key=lambda n: (float(n["start"]), float(n.get("end", 0))))
    n = len(seq)
    tb = max(float(total_beats), 1e-6)
    density = n / tb
    short_ratio = sum(1 for x in seq if float(x["end"]) - float(x["start"]) < 0.25) / max(n, 1)
    pitches = [int(x["pitch"]) for x in seq]
    ivs = [abs(b - a) for a, b in zip(pitches, pitches[1:])]
    leap_ratio = sum(1 for iv in ivs if iv > LARGE_LEAP) / max(len(ivs), 1)
    change_rate = sum(1 for a, b in zip(pitches, pitches[1:]) if b != a) / max(len(ivs), 1)
    return {
        "score": round(4 * density + 3 * short_ratio + 2 * leap_ratio + 1 * change_rate, 3),
        "density": round(density, 3),
        "short_ratio": round(short_ratio, 3),
        "leap_ratio": round(leap_ratio, 3),
        "change_rate": round(change_rate, 3),
        "notes": n,
    }


def enforce_clutter_target(
    line: List[Dict[str, Any]],
    total_beats: float,
    limit: float = 6.0,
    max_passes: int = 2,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Se clutter acima do limite, aplica passes extras de simplificação.

    Cada passo: une mesma altura adjacente e absorve a nota mais curta na
    anterior. Máximo 2 passes (sem loop infinito). Conta passes usados.
    """
    stats = {"extra_passes": 0, "extra_merged": 0}
    out = [dict(n) for n in line]
    for _ in range(max_passes):
        if clutter_score(out, total_beats)["score"] <= limit or len(out) < 2:
            break
        out, m = sustain_merge(out)
        stats["extra_merged"] += m
        # Absorve a nota mais curta (que não seja a primeira) na anterior.
        idx = min(range(1, len(out)),
                  key=lambda i: float(out[i]["end"]) - float(out[i]["start"]))
        victim = out.pop(idx)
        out[idx - 1]["end"] = round(max(float(out[idx - 1]["end"]),
                                        float(victim["end"])), 6)
        stats["extra_merged"] += 1
        stats["extra_passes"] += 1
    return out, stats


def naturalness_score(
    wind_lines: Dict[str, List[Dict[str, Any]]],
    total_beats: float,
    avg_chord_size: float = 0.0,
    crossings: int = 0,
) -> Dict[str, Any]:
    """Métrica de naturalidade (debug/testes): sustains e tons comuns somam;
    saltos, notas curtas, cruzamentos e clusters subtraem. Maior = melhor.
    """
    leaps7 = leaps12 = short_harm = common = intervals = 0
    on_beat = onsets = 0
    durs: List[float] = []
    for line in wind_lines.values():
        seq = sorted(line, key=lambda n: float(n["start"]))
        for i, n in enumerate(seq):
            dur = float(n["end"]) - float(n["start"])
            durs.append(dur)
            onsets += 1
            if abs(float(n["start"]) - round(float(n["start"]))) < 1e-6:
                on_beat += 1
            if dur < 0.5 - 1e-9:
                short_harm += 1
            if i == 0:
                continue
            prev = seq[i - 1]
            iv = abs(int(n["pitch"]) - int(prev["pitch"]))
            intervals += 1
            if iv > LARGE_LEAP:
                leaps7 += 1
            if iv > VERY_LARGE_LEAP:
                leaps12 += 1
            if iv == 0:
                common += 1
    notes = sum(len(v) for v in wind_lines.values())
    avg_dur = (sum(durs) / len(durs)) if durs else 0.0
    cluster_pen = max(0.0, avg_chord_size - 3.0) * 2.0
    total = (10.0 * (common / max(intervals, 1))
             + 2.0 * min(avg_dur, 4.0)
             + 5.0 * (on_beat / max(onsets, 1))
             - 3.0 * leaps7 - 5.0 * leaps12
             - 2.0 * short_harm - 4.0 * crossings - cluster_pen)
    return {
        "score": round(total, 3),
        "large_leaps": leaps7,
        "very_large_leaps": leaps12,
        "short_harmony_notes": short_harm,
        "common_tones": common,
        "avg_duration": round(avg_dur, 3),
        "phrase_on_beat_ratio": round(on_beat / max(onsets, 1), 3),
        "crossings": crossings,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Acompanhamento de Other (Etapa 7.2.1, perfil natural)
# ---------------------------------------------------------------------------

# Registro preferencial de acompanhamento (concert, MIDI).
ACCOMP_CENTER = 52
# Duração mínima de evento de acompanhamento no natural (beats).
ACCOMP_MIN_BEATS = 0.5
# Gap máximo para fundir rearticulações equivalentes (beats).
ACCOMP_HOLD_GAP = 0.25


def _pcs_of(pitches: List[int]) -> set:
    return {int(p) % 12 for p in pitches}


def _bass_pc_at(bass_notes: Optional[List[Dict[str, Any]]], t: float) -> Optional[int]:
    best = None
    for n in bass_notes or []:
        if float(n["start"]) <= t < float(n["end"]):
            p = int(n["pitch"])
            if best is None or p < best:
                best = p
    return (best % 12) if best is not None else None


def trio_voicing(pitches: List[int], bass_pc: Optional[int],
                 neighbor_pcs: set) -> Tuple[List[int], Dict[str, int]]:
    """Reduz simultaneidade para tríade útil (alvo 3, teto 3).

    1) Dedupe por pc (mais próximo de 52, empate: mais grave).
    2) Cluster de semitom: de cada par a 1 st, fica o de maior score
       (persistente +2, registro médio +1); empate: o grave. Par a 2 st:
       só remove o não-persistente quando o outro persiste.
    3) Trim para 3 por score (bass-pc +3, terça/quinta do baixo +2,
       topo +1, grave +1, médio +1, persistente +2); empate: mais grave.
    """
    st = {"chords_simplified": 0, "duplicate_octaves_removed": 0,
          "cluster_events_removed": 0, "duplicate_pitch_classes_removed": 0}
    uniq = sorted(set(int(p) for p in pitches))
    if len(uniq) <= 1:
        return uniq, st
    # 1) octave dups.
    by_pc: Dict[int, List[int]] = {}
    for p in uniq:
        by_pc.setdefault(p % 12, []).append(p)
    deduped = []
    for pc, group in by_pc.items():
        deduped.append(min(group, key=lambda p: (abs(p - ACCOMP_CENTER), p)))
    st["duplicate_pitch_classes_removed"] = len(uniq) - len(deduped)
    st["duplicate_octaves_removed"] = st["duplicate_pitch_classes_removed"]
    # 2-3) clusters.
    kept = sorted(deduped)
    i = 0
    while i < len(kept) - 1:
        a, b = kept[i], kept[i + 1]
        gap = b - a
        if gap == 1 or (gap == 2 and ((a % 12) in neighbor_pcs) != ((b % 12) in neighbor_pcs)):
            sa = (2 if (a % 12) in neighbor_pcs else 0) + (1 if 48 <= a <= 72 else 0)
            sb = (2 if (b % 12) in neighbor_pcs else 0) + (1 if 48 <= b <= 72 else 0)
            if gap == 2 and sa == sb:
                i += 1  # ambos/nenhum persistem: conservador, mantém
                continue
            # Empate de score: mantém o grave (mais escuro/estrutural).
            drop = i if sb > sa else i + 1
            kept.pop(drop)
            st["cluster_events_removed"] += 1
            continue
        i += 1
    if len(kept) > 3:
        st["chords_simplified"] = 1
        bass = min(kept)
        top = max(kept)

        def score(p: int) -> Tuple[int, int]:
            s = 0
            if bass_pc is not None and p % 12 == bass_pc:
                s += 3
            if (p - bass) % 12 in (3, 4, 7):
                s += 2
            if p == top:
                s += 1
            if p == bass:
                s += 1
            if 48 <= p <= 72:
                s += 1
            if (p % 12) in neighbor_pcs:
                s += 2
            return (-s, p)

        kept = sorted(sorted(kept, key=score)[:3])
    return kept, st


def _compact_span(pitches: List[int], limit: int = 16) -> List[int]:
    """Encolhe voicing espalhado: desce o topo 12 até span <= limite.

    Preserva pitch classes. Determinístico.
    """
    out = sorted(int(p) for p in pitches)
    while len(out) >= 2 and out[-1] - out[0] > limit and out[-1] - 12 > out[0]:
        out[-1] -= 12
        out = sorted(out)
    return out


def _inversion_near(pitches: List[int], prev: Optional[List[int]]) -> List[int]:
    """Escolhe inversão (rotações com ±12) de menor movimento vs anterior.

    C-E-G -> F-A-C prefere voicing próximo (ex. C-F-A) em vez de saltar
    todas as vozes. Empate: mais grave. Sem anterior: ordenado.
    """
    cur = sorted(int(p) for p in pitches)
    if prev is None or len(cur) < 2:
        return cur
    cands = [cur]
    for k in range(1, len(cur)):
        cands.append(sorted(cur[k:] + [p + 12 for p in cur[:k]]))
        cands.append(sorted([p - 12 for p in cur[k:]] + cur[:k]))

    def movement(c: List[int]) -> Tuple[int, int]:
        n = min(len(c), len(prev))
        return (sum(abs(a - b) for a, b in zip(c[:n], prev[:n])) + 6 * abs(len(c) - len(prev)),
                min(c))

    return min(cands, key=movement)


def refine_other_accompaniment(
    items: List[Dict[str, Any]],
    profile: str = "detailed",
    bass_notes: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Acompanhamento limpo de Other (natural): trio + hold + duração mínima
    + registro compacto + voice leading. NÃO muta a entrada (cópias).

    Retorna (write_items, stats) com as 9 métricas da Etapa 7.2.1.
    Detailed: cópias intactas, zeros.
    """
    stats = {"raw_polyphony_max": 0, "final_polyphony_max": 0,
             "raw_chord_count": 0, "final_chord_count": 0,
             "cluster_events_removed": 0, "duplicate_pitch_classes_removed": 0,
             "harmony_rearticulations_removed": 0, "chords_simplified": 0,
             "duplicate_octaves_removed": 0, "average_chord_duration": 0.0,
             "average_voice_movement": 0.0}
    if profile != "natural" or not items:
        return [dict(it, pitches=list(it.get("pitches", []))) for it in items], stats
    ordered = sorted(items, key=lambda i: (float(i["start"]), float(i["end"])))
    stats["raw_polyphony_max"] = _max_overlap(ordered)
    stats["raw_chord_count"] = sum(1 for it in ordered if len(it.get("pitches", [])) > 1)
    # 1) Trio voicing por item (com contexto de baixo + vizinhos).
    voiced: List[Dict[str, Any]] = []
    for idx, it in enumerate(ordered):
        nb = set()
        if idx > 0:
            nb |= _pcs_of(ordered[idx - 1].get("pitches", []))
        if idx + 1 < len(ordered):
            nb |= _pcs_of(ordered[idx + 1].get("pitches", []))
        bp = _bass_pc_at(bass_notes, float(it["start"]))
        new_p, cap = trio_voicing(list(it.get("pitches", [])), bp, nb)
        for k in ("chords_simplified", "duplicate_octaves_removed",
                  "duplicate_pitch_classes_removed", "cluster_events_removed"):
            stats[k] += cap.get(k, 0)
        if len(new_p) != len(it.get("pitches", [])) and len(it.get("pitches", [])) > 3:
            stats["chords_simplified"] = max(stats["chords_simplified"], 1)
        nit = dict(it)
        nit["pitches"] = new_p
        nit["kind"] = "chord" if len(new_p) > 1 else "note"
        voiced.append(nit)
    # 2) Hold: funde equivalentes consecutivos (mesmo pc set, subset com
    # mesmo baixo, ou >=2 pcs compartilhados com baixo estável).
    # 2) Hold: funde equivalentes consecutivos (mesmo pc set, subset com
    # mesmo baixo, ou >=2 pcs compartilhados com baixo estável). Repete até
    # fixpoint (máx 3) pois a absorção cria novas adjacências fundíveis.
    held = voiced
    for _ in range(3):
        held, changed = _hold_once(held, bass_notes, stats)
        if not changed:
            break
    # 3) Duração mínima: absorve < 0.5 no vizinho similar, senão descarta se fraco.
    absorbed: List[Dict[str, Any]] = []
    for it in held:
        dur = float(it["end"]) - float(it["start"])
        if dur >= ACCOMP_MIN_BEATS - 1e-9 or not absorbed:
            # Primeira curta é mantida provisoriamente (dobra p/ frente abaixo).
            absorbed.append(it)
            continue
        prev = absorbed[-1]
        sim = len(_pcs_of(prev["pitches"]) & _pcs_of(it["pitches"])) >= 1
        if sim:
            prev["end"] = round(max(float(prev["end"]), float(it["end"])), 6)
            stats["harmony_rearticulations_removed"] += 1
        elif int(it.get("velocity", 64)) < 64:
            stats["harmony_rearticulations_removed"] += 1  # ruído descartado
        else:
            absorbed.append(it)
    if len(absorbed) >= 2:
        first, second = absorbed[0], absorbed[1]
        if float(first["end"]) - float(first["start"]) < ACCOMP_MIN_BEATS - 1e-9 \
                and len(_pcs_of(first["pitches"]) & _pcs_of(second["pitches"])) >= 1:
            second["start"] = round(min(float(second["start"]), float(first["start"])), 6)
            absorbed.pop(0)
            stats["harmony_rearticulations_removed"] += 1
    # Hold final: absorções podem ter criado vizinhos equivalentes.
    held2, _ = _hold_once(absorbed, bass_notes, stats)
    absorbed = held2
    # 4) Registro compacto (span <= 16) + 5) voice leading entre voicings.
    final: List[Dict[str, Any]] = []
    movements: List[float] = []
    prev_p: Optional[List[int]] = None
    for it in absorbed:
        pl = _compact_span(it["pitches"])
        if len(pl) >= 2:
            inv = _inversion_near(pl, prev_p)
            if prev_p is not None:
                n = min(len(inv), len(prev_p))
                movements.append(sum(abs(a - b) for a, b in zip(inv[:n], prev_p[:n])))
            prev_p = list(inv)
        else:
            prev_p = list(pl) if pl else prev_p
        nit = dict(it)
        nit["pitches"] = pl
        nit["kind"] = "chord" if len(pl) > 1 else "note"
        final.append(nit)
    # Fallback: nunca esvaziar parte com conteúdo real.
    if not final and ordered:
        first = dict(ordered[0])
        first["pitches"] = sorted(set(int(p) for p in first.get("pitches", [])))[:3]
        first["kind"] = "chord" if len(first["pitches"]) > 1 else "note"
        final = [first]
    chords = [it for it in final if len(it.get("pitches", [])) > 1]
    stats["final_polyphony_max"] = max([len(it.get("pitches", [])) for it in final] or [0])
    stats["final_chord_count"] = len(chords)
    stats["average_chord_duration"] = round(
        sum(float(it["end"]) - float(it["start"]) for it in chords) / len(chords), 3) \
        if chords else 0.0
    stats["average_voice_movement"] = round(
        sum(movements) / len(movements), 3) if movements else 0.0
    return final, stats


def _hold_once(items: List[Dict[str, Any]],
               bass_notes: Optional[List[Dict[str, Any]]],
               stats: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], bool]:
    """Uma passada de sustain: funde vizinhos equivalentes com gap <= 0.25.

    Equivalência: mesmo pc set; subset em qualquer direção com mesmo baixo
    (cobre C-E-G + C-E-G-C e sobreposição de mesmo onset); ou >=2 pcs
    compartilhados com baixo estável. Mantém o voicing mais rico.
    """
    held: List[Dict[str, Any]] = []
    changed = False
    for it in items:
        if not held:
            held.append(it)
            continue
        prev = held[-1]
        gap = float(it["start"]) - float(prev["end"])
        pcs_prev, pcs_cur = _pcs_of(prev["pitches"]), _pcs_of(it["pitches"])
        bp_prev = _bass_pc_at(bass_notes, float(prev["start"]))
        bp_cur = _bass_pc_at(bass_notes, float(it["start"]))
        same_onset = abs(float(it["start"]) - float(prev["start"])) < 1e-9
        subset = bool(pcs_cur and pcs_prev) and (
            pcs_cur <= pcs_prev or pcs_prev <= pcs_cur)
        equivalent = (pcs_cur == pcs_prev) or (
            subset and (bp_cur is None or bp_prev is None or bp_cur == bp_prev))
        similar = len(pcs_cur & pcs_prev) >= 2 and bp_cur is not None and bp_cur == bp_prev
        if gap <= ACCOMP_HOLD_GAP + 1e-9 and (equivalent or similar):
            # Mantém o voicing mais rico (mais pitches; empate: o anterior).
            if len(it.get("pitches", [])) > len(prev.get("pitches", [])):
                prev["pitches"] = list(it.get("pitches", []))
            prev["end"] = round(max(float(prev["end"]), float(it["end"])), 6)
            prev["velocity"] = max(int(prev.get("velocity", 0)), int(it.get("velocity", 0)))
            stats["harmony_rearticulations_removed"] += 1
            changed = True
        elif same_onset and subset:
            # Mesmo onset com subset (ex. [C] + [C,E,G]): mesma harmonia.
            if len(it.get("pitches", [])) > len(prev.get("pitches", [])):
                prev["pitches"] = list(it.get("pitches", []))
            prev["end"] = round(max(float(prev["end"]), float(it["end"])), 6)
            stats["harmony_rearticulations_removed"] += 1
            changed = True
        else:
            held.append(it)
    return held, changed


def _max_overlap(items: List[Dict[str, Any]]) -> int:
    """Profundidade máxima de sobreposição (sweep de eventos)."""
    pts: List[Tuple[float, int]] = []
    for it in items:
        s, e = float(it["start"]), float(it["end"])
        if e > s:
            pts.append((s, 1))
            pts.append((e, -1))
    depth = best = 0
    for _, d in sorted(pts, key=lambda x: (x[0], x[1])):
        depth += d
        best = max(best, depth)
    return best
