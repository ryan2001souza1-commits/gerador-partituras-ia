"""
Etapa 7 — arranjador determinístico (stdlib apenas, sem music21, sem LLM).

Entrada: linhas quantizadas da partitura base (vocals monofônico, other
polifônico em itens chord/note) + tonalidade (se confiável).
Tudo em CONCERT PITCH; a transposição para written acontece no worker.

Papéis (modo automatic):
- 1 instrumento -> melodia
- 2 -> melodia + harmonia
- 3 -> melodia + 2 harmonias
- 4+ -> melodia + harmonias + voz grave (último = low: raízes/terças)
- mode "melody": todos dobram a melodia (oitavas conforme tessitura)
- mode "harmony": todos em linhas harmônicas (do agudo ao grave)

Melhor silêncio (pausa) do que nota arbitrária ruim.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from backend.arrangement.instrument_definitions import (
    InstrumentDefinition,
    REGISTER_ORDER,
    concert_to_written,
    fit_range,
    get_instrument,
)
from backend.musical.cleanup import (
    HARMONY_CHANGE_MARGIN,
    HARMONY_MIN_WEIGHT,
    HARMONY_WINDOW_BEATS,
    LARGE_LEAP,
    MIN_HARMONY_BEATS,
    NATURAL_HARMONY_WINDOW,
    PHRASE_GAP_BEATS,
    VERY_LARGE_LEAP,
    apply_breathing_v2,
    block_rhythm,
    enforce_clutter_target,
    get_window_harmony,
    get_window_harmony_weights,
    line_stats,
    merge_adjacent_harmony_notes,
    naturalness_score,
    phrase_boundaries,
    reduce_harmony_rhythm,
    smooth_melody,
    sustain_merge,
)

# Mínimo melódico do modo simplified (beats): abaixo disso, ornamento do
# transcritor -> duração absorvida pela nota anterior (determinístico).
SIMPLIFY_MIN_BEATS = 0.5
# Frase máxima sem respiro (beats). Conservador: 4 compassos 4/4.
MAX_PHRASE_BEATS = 16.0
# Duração do respiro inserido (beats).
BREATH_BEATS = 0.5

# Penalidades do voice leading (custo arbitrário, só ordena candidatos).
W_UNISON = 100.0      # mesma altura da melodia: proibido na prática
W_MINOR_SECOND = 25.0  # 2ª menor contra melodia ou harmonia ativa
W_RANGE_EDGE = 8.0     # fora do preferido (mas dentro do absoluto)
W_CROSS = 12.0         # cruzar a melodia por cima
# Pesos adicionais do perfil natural (Etapa de refinamento).
W_LARGE_LEAP = 1.5      # por semitom além de 7
W_VERY_LARGE_LEAP = 20.0  # salto > 12: penalidade forte (não proíbe)
W_STAY_BONUS = 6.0      # histerese: manter a nota atual (common tone)
W_STEP_BONUS = 2.0      # movimento por graus conjuntos / mesmo pc
W_LONG_DISSONANCE = 15.0  # 2ªm/9ªm/trítono sustentado na harmonia


def extract_main_melody(
    vocals_notes: List[Dict[str, Any]],
    other_items: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Linha melódica principal: vocals quantizado; se vazio, voz superior de other.

    Remove microfragmentos (dur <= 0) e resolve overlaps residuais.
    Retorna (melody, stats) com itens {start, end, pitch, velocity}.
    """
    stats = {"source": "vocals", "input": 0, "output": 0, "removed": 0}
    notes = [dict(n) for n in (vocals_notes or [])]
    stats["input"] = len(notes)
    if not notes and other_items:
        stats["source"] = "other_upper"
        per_onset: Dict[float, Dict[str, Any]] = {}
        for it in other_items:
            s = round(float(it["start"]), 6)
            top = max(int(p) for p in it["pitches"])
            e = round(float(it["end"]), 6)
            if s not in per_onset or top > per_onset[s]["pitch"]:
                per_onset[s] = {"start": s, "end": e, "pitch": top,
                                "velocity": int(it.get("velocity", 64))}
        notes = [per_onset[k] for k in sorted(per_onset)]
        stats["input"] = len(notes)
    notes.sort(key=lambda n: (float(n["start"]), float(n["end"])))
    out: List[Dict[str, Any]] = []
    for n in notes:
        try:
            s, e, p = float(n["start"]), float(n["end"]), int(n["pitch"])
        except (TypeError, ValueError, KeyError):
            stats["removed"] += 1
            continue
        if e <= s or p < 0 or p > 127:
            stats["removed"] += 1
            continue
        if out and s < float(out[-1]["end"]):
            out[-1]["end"] = s  # encurta anterior (monofonia)
            if float(out[-1]["end"]) <= float(out[-1]["start"]):
                out.pop()
                stats["removed"] += 1
        out.append({"start": round(s, 6), "end": round(e, 6), "pitch": p,
                    "velocity": int(n.get("velocity", 64))})
    stats["output"] = len(out)
    return out, stats


def simplify_melody(
    melody: List[Dict[str, Any]],
    min_beats: float = SIMPLIFY_MIN_BEATS,
    full: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Simplified (default): notas < min_beats têm a duração absorvida pela
    anterior (primeira nota sempre preservada). Full: copia integral."""
    stats = {"dropped": 0, "absorbed_beats": 0.0}
    if full or not melody:
        return [dict(m) for m in melody], stats
    out: List[Dict[str, Any]] = []
    for m in melody:
        dur = float(m["end"]) - float(m["start"])
        if out and dur < min_beats:
            stats["dropped"] += 1
            stats["absorbed_beats"] = round(stats["absorbed_beats"] + dur, 6)
            out[-1]["end"] = m["end"]  # anterior sustenta (respiro/resolução)
        else:
            out.append(dict(m))
    return out, stats


def apply_breathing(
    line: List[Dict[str, Any]],
    max_phrase: float = MAX_PHRASE_BEATS,
    breath: float = BREATH_BEATS,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Insere respiro curto após frase longa sem pausa (>=0.5 beat conta como ar).

    Procura a nota mais longa dos últimos 4 beats e encurta seu fim em
    `breath` (nunca abaixo de 0.5 beat). Conservador e determinístico.
    """
    stats = {"breath_adjustments": 0}
    if not line:
        return [], stats
    out = [dict(m) for m in line]
    since_rest = 0.0
    for i, n in enumerate(out):
        gap_before = float(n["start"]) - (float(out[i - 1]["end"]) if i else 0.0)
        if gap_before >= 0.5:
            since_rest = 0.0
        since_rest += float(n["end"]) - float(n["start"])
        if since_rest >= max_phrase:
            # Encurta a nota mais longa da janela recente.
            window = out[max(0, i - 7): i + 1]
            target = max(window, key=lambda w: float(w["end"]) - float(w["start"]))
            dur = float(target["end"]) - float(target["start"])
            if dur > breath + 0.5:
                target["end"] = round(float(target["end"]) - breath, 6)
                stats["breath_adjustments"] += 1
                since_rest = 0.0
    return out, stats


def get_active_harmony(
    other_items: List[Dict[str, Any]], time_beats: float
) -> List[int]:
    """Pitch classes de `other` soando em time_beats (ordenadas, únicas)."""
    pcs = set()
    for it in other_items or []:
        if float(it["start"]) <= time_beats < float(it["end"]):
            for p in it["pitches"]:
                pcs.add(int(p) % 12)
    return sorted(pcs)


def _pc_dissonant_against(cand_pc: int, ref_pcs: List[int]) -> bool:
    return any(((cand_pc - r) % 12) in (1, 11) for r in ref_pcs)


def select_harmony_note(
    melody_pitch: int,
    melody_start: float,
    active_pcs: List[int],
    prev_pitch: Optional[int],
    definition: InstrumentDefinition,
    must_be_below: bool = True,
) -> Tuple[Optional[int], Dict[str, Any]]:
    """Escolhe nota harmônica por custo (voice leading determinístico).

    Candidatos: chord tones reais de other abaixo da melodia (concert).
    Fallbacks ordenados: oitava abaixo da melodia -> quinta justa abaixo ->
    pausa. Retorna (pitch_ou_None, info{source, cost}).
    """
    info: Dict[str, Any] = {"source": "none", "cost": None}
    # Candidatos: pcs ativas realizadas em 2 oitavas abaixo da melodia.
    cands = set()
    for pc in active_pcs:
        for octv in range(melody_pitch - 24, melody_pitch - 1):
            if octv % 12 == pc:
                cands.add(octv)
    if must_be_below:
        cands = {c for c in cands if c <= melody_pitch - 2}
    best = None
    best_cost = None
    ordered = sorted(cands, reverse=True)  # perto da melodia primeiro
    for c in ordered:
        if c < 0 or c > 127:
            continue
        fitted, _ = fit_range(c, definition)
        cost = abs(c - (prev_pitch if prev_pitch is not None else melody_pitch - 4))
        if c == melody_pitch:
            cost += W_UNISON
        if _pc_dissonant_against(c % 12, [melody_pitch % 12] + active_pcs):
            cost += W_MINOR_SECOND
        if fitted is None:
            continue
        if fitted != c:
            cost += W_RANGE_EDGE
        if fitted is not None and fitted > melody_pitch:
            cost += W_CROSS
        if best_cost is None or cost < best_cost:
            best_cost, best = cost, fitted
    if best is not None:
        info.update(source="chord_tone", cost=round(best_cost or 0.0, 3))
        return best, info
    # Fallbacks: 8ª, depois 5ª justa abaixo da melodia.
    for interval, name in ((12, "octave"), (7, "fifth")):
        fb = melody_pitch - interval
        fitted, _ = fit_range(fb, definition)
        if fitted is not None and (not must_be_below or fitted <= melody_pitch - 2):
            info.update(source="fallback_" + name, cost=None)
            return fitted, info
    return None, info


def _register_center(definition: InstrumentDefinition) -> Tuple[float, float]:
    """Centro e meia-largura da tessitura preferida (concert, sem tocar defs)."""
    from backend.arrangement.instrument_definitions import concert_low_high
    _, _, plo, phi = concert_low_high(definition)
    return (plo + phi) / 2.0, (phi - plo) / 2.0


def select_harmony_note_v2(
    melody_pitch: int,
    active_pcs: List[int],
    prev_pitch: Optional[int],
    definition: InstrumentDefinition,
    melody_dur: float = 1.0,
    must_be_below: bool = True,
    phrase_start: bool = False,
    top_pcs: Optional[List[int]] = None,
    melody_prev: Optional[int] = None,
) -> Tuple[Optional[int], Dict[str, Any]]:
    """Voice leading refinado (perfil natural): saltos progressivos, bônus de
    permanência (histerese), common tone, dissonância longa penalizada.

    Mesmos candidatos/fallbacks da v1; custos estendidos. Determinístico.
    - phrase_start: alivia penalidade de salto (frase nova permite mudar);
    - top_pcs: pcs dominantes (cadência/fundamental) ganham bônus;
    - melody_prev: salto similar grande na mesma direção é penalizado.
    """
    info: Dict[str, Any] = {"source": "none", "cost": None}
    center, half = _register_center(definition)
    cands = set()
    for pc in active_pcs:
        for octv in range(melody_pitch - 24, melody_pitch - 1):
            if octv % 12 == pc:
                cands.add(octv)
    if must_be_below:
        cands = {c for c in cands if c <= melody_pitch - 2}
    best = None
    best_cost = None
    relief = 0.5 if phrase_start else 1.0
    for c in sorted(cands, reverse=True):
        if c < 0 or c > 127:
            continue
        fitted, _ = fit_range(c, definition)
        if fitted is None:
            continue
        ref = prev_pitch if prev_pitch is not None else melody_pitch - 4
        dist = abs(c - ref)
        cost = float(dist)
        if dist > VERY_LARGE_LEAP:
            cost += (W_VERY_LARGE_LEAP + (dist - VERY_LARGE_LEAP) * W_LARGE_LEAP) * relief
        elif dist > LARGE_LEAP:
            cost += (dist - LARGE_LEAP) * W_LARGE_LEAP * relief
        elif dist > 5:
            cost += (dist - 5) * 1.0 * relief
        if c == melody_pitch:
            cost += W_UNISON
        if _pc_dissonant_against(c % 12, [melody_pitch % 12] + active_pcs):
            cost += W_MINOR_SECOND
            if melody_dur >= 1.0:
                cost += W_LONG_DISSONANCE
        if ((c - melody_pitch) % 12) == 2 and melody_dur >= 1.0:
            cost += 10.0  # 2ª maior sustentada em contexto forte
        if (c % 12) == ((melody_pitch - 6) % 12):  # trítono c/ melodia
            cost += W_MINOR_SECOND
            if melody_dur >= 1.0:
                cost += W_LONG_DISSONANCE
        excess = max(0.0, abs(fitted - center) - half)
        if fitted != c or excess > 0:
            cost += W_RANGE_EDGE if fitted != c else 0.0
            cost += excess * 0.5  # register lock: longe do centro pesa
        if fitted > melody_pitch:
            cost += W_CROSS + max(0.0, fitted - melody_pitch - 5) * 1.0
        if top_pcs and (fitted % 12) in top_pcs:
            cost -= 4.0  # cadência/fundamental: root/3rd/5th
        if melody_prev is not None and prev_pitch is not None:
            mel_leap = melody_pitch - melody_prev
            own_leap = fitted - prev_pitch
            if abs(mel_leap) > LARGE_LEAP and abs(own_leap) > LARGE_LEAP \
                    and (mel_leap > 0) == (own_leap > 0):
                cost += 12.0  # salto paralelo grande
        if prev_pitch is not None:
            if fitted == prev_pitch:
                cost -= W_STAY_BONUS  # histerese: C -> C -> C
            elif abs(fitted - prev_pitch) <= 2 or (fitted % 12) == (prev_pitch % 12):
                cost -= W_STEP_BONUS
        if best_cost is None or cost < best_cost or \
                (cost == best_cost and fitted < (best or 0)):
            best_cost, best = cost, fitted
    if best is not None:
        info.update(source="chord_tone", cost=round(best_cost or 0.0, 3))
        return best, info
    for interval, name in ((12, "octave"), (7, "fifth")):
        fb = melody_pitch - interval
        fitted, _ = fit_range(fb, definition)
        if fitted is not None and (not must_be_below or fitted <= melody_pitch - 2):
            info.update(source="fallback_" + name, cost=None)
            return fitted, info
    return None, info


def build_harmony_line(
    melody: List[Dict[str, Any]],
    other_items: List[Dict[str, Any]],
    definition: InstrumentDefinition,
    low_role: bool = False,
    profile: str = "detailed",
    harmony_window: float = HARMONY_WINDOW_BEATS,
    bass_notes: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Linha harmônica nota-a-nota sobre a melodia (mesmos onsets).

    low_role (voz grave): prefere o candidato mais grave e estende durações
    iguais consecutivas (sustentação de fundamental/terça).
    Natural: janela de 2 beats + pesos (baixo/downbeat) + memória com margem
    + duração mínima de 1 beat + reset por frase + cadência estável.
    Detailed: comportamento Etapa 7 (instantâneo, sem histerese).
    """
    stats = {"notes": 0, "rests": 0, "from_chord_tones": 0,
             "from_fallback": 0, "range_adjustments": 0, "dropped_out_of_range": 0,
             "harmonic_changes": 0, "stays": 0, "harmony_switches": 0}
    line: List[Dict[str, Any]] = []
    prev: Optional[int] = None
    last_end = 0.0
    last_pcs: List[int] = []
    harm_start = 0.0
    harm_pcs: List[int] = []
    strengths = _item_strengths(other_items)
    mel_list = list(melody)
    for idx, m in enumerate(mel_list):
        s, e = float(m["start"]), float(m["end"])
        mel_dur = float(m["end"]) - float(m["start"])
        next_gap = (float(mel_list[idx + 1]["start"]) - e) if idx + 1 < len(mel_list) else 99.0
        phrase_start = (not line) or (s - last_end >= PHRASE_GAP_BEATS)
        phrase_end = (idx == len(mel_list) - 1) or (next_gap >= PHRASE_GAP_BEATS)
        if profile == "natural":
            if line and s - last_end >= PHRASE_GAP_BEATS:
                prev = None  # nova frase: sem inércia do voice leading
            window = NATURAL_HARMONY_WINDOW
            bass_pcs = _bass_pcs_in_window(bass_notes, s, window)
            strong = (s % 2.0) < 1e-6  # beats 1/3 em 4/4
            weights = get_window_harmony_weights(
                other_items, s, window, strengths=strengths,
                bass_pcs=bass_pcs, strong_beat=strong)
            pcs = sorted(pc for pc, w in weights.items()
                         if w >= HARMONY_MIN_WEIGHT * window)
            if not pcs:
                pcs = get_active_harmony(other_items, s + 1e-6)
            top_pcs = [pc for pc, _ in sorted(weights.items(),
                                              key=lambda kv: -kv[1])[:2]]
            # Offbeat que repete a harmonia: ignora a troca.
            offbeat = (s % 1.0) > 1e-6
            if offbeat and pcs == last_pcs and prev is not None:
                pitch, info = prev, {"source": "sustain", "cost": 0.0}
            else:
                pitch, info = select_harmony_note_v2(
                    int(m["pitch"]), pcs, prev, definition, melody_dur=mel_dur,
                    must_be_below=True, phrase_start=phrase_start,
                    top_pcs=top_pcs if phrase_end else None,
                    melody_prev=(int(mel_list[idx - 1]["pitch"]) if idx else None))
                # Memória com margem + duração mínima (natural).
                if pitch is not None and prev is not None and line:
                    stay, stay_cost = _stay_option(
                        prev, pcs, definition, int(m["pitch"]), mel_dur)
                    switch = True
                    if stay is not None:
                        new_cost = float(info.get("cost") or 0.0)
                        dissonant_stay = _pc_dissonant_against(
                            stay % 12, [int(m["pitch"]) % 12]) and mel_dur >= 1.0
                        structural = not (set(pcs) & set(harm_pcs))
                        min_dur_ok = (s - harm_start) >= MIN_HARMONY_BEATS
                        if not dissonant_stay and (not structural or not min_dur_ok) \
                                and not (new_cost < stay_cost - HARMONY_CHANGE_MARGIN):
                            pitch, info = stay, {"source": "sustain", "cost": stay_cost}
                            switch = False
                    if switch:
                        harm_start, harm_pcs = s, list(pcs)
                        stats["harmony_switches"] += 1
                elif pitch is not None:
                    harm_start, harm_pcs = s, list(pcs)
            last_pcs = list(pcs)
        else:
            pcs = get_active_harmony(other_items, s + 1e-6)
            pitch, info = select_harmony_note(
                int(m["pitch"]), s, pcs, prev, definition, must_be_below=True)
        if pitch is None:
            stats["rests"] += 1
            prev = None
            continue
        if info["source"] == "chord_tone":
            stats["from_chord_tones"] += 1
        else:
            stats["from_fallback"] += 1
        fitted, adj = fit_range(pitch, definition)
        stats["range_adjustments"] += adj
        if fitted is None:
            stats["dropped_out_of_range"] += 1
            stats["rests"] += 1
            prev = None
            continue
        if pitch != fitted:
            stats["range_adjustments"] += 0  # já contado em fit_range
        if low_role and line and line[-1]["pitch"] == fitted \
                and abs(float(line[-1]["end"]) - s) < 1e-6:
            line[-1]["end"] = round(e, 6)  # sustenta
        else:
            if line and int(line[-1]["pitch"]) == int(fitted):
                stats["stays"] += 1
            line.append({"start": round(s, 6), "end": round(e, 6),
                         "pitch": int(fitted), "velocity": int(m.get("velocity", 64))})
            stats["notes"] += 1
        prev = int(fitted)
        last_end = round(e, 6)
    stats["harmonic_changes"] = sum(
        1 for a, b in zip(line, line[1:]) if int(a["pitch"]) != int(b["pitch"]))
    return line, stats


def _item_strengths(other_items: List[Dict[str, Any]]) -> Dict[int, float]:
    """Força média por pitch (para ponderar a janela harmônica)."""
    acc: Dict[int, list] = {}
    for it in other_items or []:
        try:
            s = float(it.get("strength", it.get("velocity", 64)))
            if "velocity" in it and "strength" not in it:
                s = float(it["velocity"]) / 127.0
        except (TypeError, ValueError):
            s = 0.5
        for p in it.get("pitches", []):
            acc.setdefault(int(p), []).append(max(0.0, min(1.0, s)))
    return {p: sum(v) / len(v) for p, v in acc.items()}


def _bass_pcs_in_window(
    bass_notes: Optional[List[Dict[str, Any]]], start: float, window: float
) -> List[int]:
    """Pitch classes do baixo soando na janela (peso de raiz)."""
    pcs = set()
    for n in bass_notes or []:
        if float(n["start"]) < start + window - 1e-9 and float(n["end"]) > start + 1e-9:
            pcs.add(int(n["pitch"]) % 12)
    return sorted(pcs)


def _stay_option(prev: int, pcs: List[int], definition: InstrumentDefinition,
                 melody_pitch: int, melody_dur: float) -> Tuple[Optional[int], float]:
    """Custo de manter a nota atual (histerese): válida se pc ainda ativo."""
    fitted, _ = fit_range(prev, definition)
    if fitted is None or (fitted % 12) not in pcs:
        return None, 0.0
    cost = 0.0
    if _pc_dissonant_against(fitted % 12, [melody_pitch % 12] + pcs):
        cost += W_MINOR_SECOND
        if melody_dur >= 1.0:
            cost += W_LONG_DISSONANCE
    if fitted != prev:
        cost += W_RANGE_EDGE
    return fitted, cost


def fix_crossings(
    lines: Dict[str, List[Dict[str, Any]]],
    roles: List[Tuple[str, str]],
    definitions: Dict[str, InstrumentDefinition],
) -> Tuple[Dict[str, List[Dict[str, Any]]], int, int]:
    """Tenta desfazer cruzamentos (voz grave acima da aguda) com ±12.

    Para cada onset com cruzamento entre vozes adjacentes: desce a voz grave
    12 (se couber no absoluto) senão sobe a aguda 12. Determinístico.
    Retorna (lines, fixed, remaining).
    """
    from backend.arrangement.instrument_definitions import concert_low_high
    fixed = 0
    order = [i for i, _ in roles]
    by_time: Dict[float, Dict[str, Dict[str, Any]]] = {}
    for inst_id in order:
        for n in lines.get(inst_id, []):
            by_time.setdefault(round(float(n["start"]), 6), {})[inst_id] = n
    for mapping in by_time.values():
        seq = [(i, mapping[i]) for i in order if i in mapping]
        for k in range(len(seq) - 1):
            upper_id, upper_n = seq[k]
            lower_id, lower_n = seq[k + 1]
            if int(lower_n["pitch"]) <= int(upper_n["pitch"]):
                continue
            lo_def, hi_def = definitions[lower_id], definitions[upper_id]
            lo_abs = concert_low_high(lo_def)[:2]
            hi_abs = concert_low_high(hi_def)[:2]
            if int(lower_n["pitch"]) - 12 >= lo_abs[0]:
                lower_n["pitch"] = int(lower_n["pitch"]) - 12
                fixed += 1
            elif int(upper_n["pitch"]) + 12 <= hi_abs[1]:
                upper_n["pitch"] = int(upper_n["pitch"]) + 12
                fixed += 1
    remaining = _count_crossings(lines, roles)
    return lines, fixed, remaining


def assign_roles(inst_ids: List[str], mode: str) -> List[Tuple[str, str]]:
    """Distribui papéis (determinístico, agudo->grave pelo REGISTER_ORDER).

    automatic: 1=melody; 2=melody+harmony; 3=melody+2×harmony;
               4+=melody+harmonias+low (último).
    melody: todos melody. harmony: todos harmony (1º = harmony_high).
    """
    ordered = sorted(inst_ids, key=lambda i: REGISTER_ORDER.index(i)
                     if i in REGISTER_ORDER else 99)
    if mode == "melody":
        return [(i, "melody") for i in ordered]
    if mode == "harmony":
        roles = ["harmony_high"] + ["harmony"] * (len(ordered) - 1)
        if len(ordered) >= 3:
            roles[-1] = "low"
        return list(zip(ordered, roles))
    # automatic
    if len(ordered) == 1:
        return [(ordered[0], "melody")]
    if len(ordered) == 2:
        return [(ordered[0], "melody"), (ordered[1], "harmony")]
    if len(ordered) == 3:
        return [(ordered[0], "melody"), (ordered[1], "harmony"), (ordered[2], "harmony")]
    return ([(ordered[0], "melody")]
            + [(i, "harmony") for i in ordered[1:-1]]
            + [(ordered[-1], "low")])


def arrange(
    melody: List[Dict[str, Any]],
    other_items: List[Dict[str, Any]],
    instruments: List[InstrumentDefinition],
    mode: str = "automatic",
    simplify: bool = True,
    profile: str = "detailed",
    bass_notes: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Gera linhas em concert pitch por instrumento + estatísticas.

    Natural: smooth na melodia, redução rítmica + sustains nas harmonias,
    ritmo em bloco com 3+ sopros, respiração por prioridades, correção de
    cruzamentos, alvo de clutter, métricas antes/depois por instrumento.
    Detailed: comportamento Etapa 7.
    Retorna (lines_por_inst_id, report). Linhas: [{start,end,pitch,velocity}].
    """
    report: Dict[str, Any] = {"roles": {}, "stats": {}, "warnings": []}
    if simplify:
        melody_use, simp = simplify_melody(melody)
        report["stats"]["simplify"] = simp
    else:
        melody_use, simp = [dict(m) for m in melody], {"dropped": 0}
        report["stats"]["simplify"] = simp
    if profile == "natural":
        melody_use, smo = smooth_melody(melody_use, profile)
        report["stats"]["smooth"] = smo
    roles = assign_roles([d.id for d in instruments], mode)
    report["roles"] = {i: r for i, r in roles}
    by_id = {d.id: d for d in instruments}
    lines: Dict[str, List[Dict[str, Any]]] = {}
    harmony_ids: List[str] = []
    for inst_id, role in roles:
        definition = by_id[inst_id]
        if role == "melody":
            if profile == "natural":
                merged_mel, mg = merge_adjacent_harmony_notes(melody_use)
                line, bst = apply_breathing_v2(merged_mel)
            else:
                line, bst = apply_breathing(melody_use)
            fitted_line, adj = _fit_line(line, definition)
            before = line_stats(line)
            after = line_stats(fitted_line)
            report["stats"][inst_id] = {
                "role": role, "breath": bst, "range_adjustments": adj[0],
                "dropped": adj[1], "notes": len(fitted_line),
                "merged_repeats": mg if profile == "natural" else 0,
                "source_notes": before["notes"], "final_notes": after["notes"],
                "large_leaps_before": before["large_leaps"],
                "large_leaps_after": after["large_leaps"],
                "average_interval_before": before["average_interval"],
                "average_interval_after": after["average_interval"],
                "harmonic_changes_before": before["harmonic_changes"],
                "harmonic_changes_after": after["harmonic_changes"],
            }
            lines[inst_id] = fitted_line
        else:
            low = role == "low"
            hline, hst = build_harmony_line(
                melody_use, other_items, definition, low_role=low, profile=profile,
                bass_notes=bass_notes)
            before = line_stats(hline)
            if profile == "natural":
                min_dur = 1.0  # máscara rítmica: sax e trombone sem micro-eventos
                hline, red = reduce_harmony_rhythm(hline, min_dur=min_dur)
                hst["rhythmic_simplifications"] = red["rhythmic_simplifications"]
                hline, sus = sustain_merge(hline)
                hst["sustains_merged"] = sus
                hline, mg = merge_adjacent_harmony_notes(hline)
                hst["sustains_merged"] += mg
                bline, bst = apply_breathing_v2(hline)
            else:
                bline, bst = apply_breathing(hline)
            after = line_stats(bline)
            report["stats"][inst_id] = {
                "role": role, **hst, "breath": bst, "notes": len(bline),
                "source_notes": before["notes"], "final_notes": after["notes"],
                "large_leaps_before": before["large_leaps"],
                "large_leaps_after": after["large_leaps"],
                "average_interval_before": before["average_interval"],
                "average_interval_after": after["average_interval"],
                "harmonic_changes_before": before["harmonic_changes"],
                "harmonic_changes_after": after["harmonic_changes"],
            }
            lines[inst_id] = bline
            harmony_ids.append(inst_id)
    if profile == "natural" and len(harmony_ids) >= 2 and \
            sum(1 for i, _ in roles if i in lines) >= 3:
        # Sopros em bloco: harmonias com ritmo compartilhado (seção, não solos).
        # Grade de 1 beat: mudanças só em momentos estruturais.
        harm_lines = {i: lines[i] for i in harmony_ids}
        blocked, _ = block_rhythm(harm_lines, slot=1.0)
        for i, bl in blocked.items():
            before = line_stats(lines[i])
            after = line_stats(bl)
            st = report["stats"][i]
            st["final_notes"] = after["notes"]
            st["large_leaps_after"] = after["large_leaps"]
            st["average_interval_after"] = after["average_interval"]
            st["harmonic_changes_after"] = after["harmonic_changes"]
            st["notes"] = len(bl)
            lines[i] = bl
    if profile == "natural":
        # Correção de cruzamentos (oitava na voz harmônica) + alvo de clutter.
        by_id = {d.id: d for d in instruments}
        total_beats = 0.0
        for ln in lines.values():
            for n in ln:
                total_beats = max(total_beats, float(n["end"]))
        for inst_id in harmony_ids:
            lines[inst_id], cl = enforce_clutter_target(
                lines[inst_id], total_beats, limit=6.0, max_passes=2)
            if cl["extra_passes"]:
                st = report["stats"][inst_id]
                st["clutter_passes"] = cl["extra_passes"]
                st["notes"] = len(lines[inst_id])
                after = line_stats(lines[inst_id])
                st["final_notes"] = after["notes"]
                st["large_leaps_after"] = after["large_leaps"]
                st["average_interval_after"] = after["average_interval"]
                st["harmonic_changes_after"] = after["harmonic_changes"]
        lines, fixed, remaining = fix_crossings(lines, roles, by_id)
        report["crossings_fixed"] = fixed
        # Naturalidade da versão final (debug/testes).
        avg_chord = 0.0
        if other_items:
            avg_chord = sum(len(it.get("pitches", [])) for it in other_items) / len(other_items)
        report["naturalness"] = naturalness_score(lines, total_beats,
                                                  avg_chord_size=avg_chord,
                                                  crossings=remaining)
    # Métrica de cruzamento (info p/ teste/validação, sem reescrita).
    report["voice_crossings"] = _count_crossings(lines, roles)
    return lines, report


def _fit_line(
    line: List[Dict[str, Any]], definition: InstrumentDefinition
) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    """Encaixa a linha preservando o contorno melódico.

    1) Deslocamento global de oitava (0, ±12, ±24; prefere o menor) que
       maximiza notas no preferido (desempate: no absoluto, depois menor
       deslocamento). Evita que uma frase "quebre" no meio (ex. D4->D3).
    2) Resíduos fora do absoluto passam por fit_range nota a nota; o
       impossível vira pausa (nunca pitch impossível silencioso).
    """
    from backend.arrangement.instrument_definitions import concert_low_high
    if not line:
        return [], (0, 0)
    abs_low, abs_high, pref_low, pref_high = concert_low_high(definition)
    center = (pref_low + pref_high) / 2.0
    pitches = [int(n["pitch"]) for n in line]
    best_key = None
    best_shift = 0
    for s in (0, 12, -12, 24, -24):
        shifted = [p + s for p in pitches]
        in_pref = sum(1 for p in shifted if pref_low <= p <= pref_high)
        in_abs = sum(1 for p in shifted if abs_low <= p <= abs_high)
        med = sorted(shifted)[len(shifted) // 2]
        # Register lock: só desempata (menor shift continua mandando).
        key = (in_pref, in_abs, -abs(s), -abs(med - center))
        if best_key is None or key > best_key:
            best_key, best_shift = key, s
    out: List[Dict[str, Any]] = []
    adj_total = len(line) if best_shift else 0
    dropped = 0
    for n in line:
        base = int(n["pitch"]) + best_shift
        fitted, adj = _fit_absolute(base, definition)
        adj_total += adj
        if fitted is None:
            dropped += 1
            continue
        nn = dict(n)
        nn["pitch"] = fitted
        out.append(nn)
    return out, (adj_total, dropped)


def _fit_absolute(pitch: int, definition: InstrumentDefinition) -> Tuple[Optional[int], int]:
    """Encaixa só no ABSOLUTO (sem escalar p/ preferido): preserva contorno.

    Tenta 0, depois ±12/±24 pelo lado mais próximo. Fora do absoluto -> None.
    """
    from backend.arrangement.instrument_definitions import concert_low_high
    abs_low, abs_high = concert_low_high(definition)[:2]
    p = int(pitch)
    if abs_low <= p <= abs_high:
        return p, 0
    order = (12, -12, 24, -24) if p < abs_low else (-12, 12, -24, 24)
    for s in order:
        if abs_low <= p + s <= abs_high:
            return p + s, abs(s) // 12
    return None, 0


def _count_crossings(
    lines: Dict[str, List[Dict[str, Any]]], roles: List[Tuple[str, str]]
) -> int:
    """Conta eventos onde voz mais grave soa acima de voz mais aguda."""
    order = [i for i, _ in roles]  # agudo -> grave
    points: Dict[float, Dict[str, int]] = {}
    for inst_id in order:
        for n in lines.get(inst_id, []):
            points.setdefault(round(float(n["start"]), 6), {})[inst_id] = int(n["pitch"])
    crossings = 0
    for mapping in points.values():
        seq = [mapping[i] for i in order if i in mapping]
        for a, b in zip(seq, seq[1:]):
            if b is not None and a is not None and b > a:
                crossings += 1
    return crossings


def written_line(
    concert_line: List[Dict[str, Any]], definition: InstrumentDefinition
) -> List[Dict[str, Any]]:
    """Concert -> written (para exportação MusicXML transposta)."""
    out = []
    for n in concert_line:
        nn = dict(n)
        nn["pitch"] = concert_to_written(int(n["pitch"]), definition)
        out.append(nn)
    return out
