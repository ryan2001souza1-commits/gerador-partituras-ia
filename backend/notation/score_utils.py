"""
Etapa 6 — utilitários puros de quantização / limpeza / teoria musical.

NÃO importa music21 aqui: este módulo roda no processo FastAPI principal
(.venv) e é coberto por testes unitários rápidos. A construção do Score
music21 vive em backend/workers/notation_worker.py (executado via
.venv-notation).

Documentação das decisões:
- Grade binária limpa; tercinas/tuplets automáticos NÃO são detectados
  nesta etapa ("Tuplets automáticos serão aprimorados futuramente").
- Amplitude/confidence do Basic Pitch NÃO é probabilidade científica;
  MIN_NOTE_STRENGTH é conservador (0.05) e só remove notas quando há
  evidência clara de fraqueza em todos os campos disponíveis.
- pitch bends do Basic Pitch são preservados no MIDI da Etapa 5 e NÃO
  são convertidos em microtons na partitura nesta etapa.
"""

from __future__ import annotations

import hashlib
import statistics
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constantes centralizadas
# ---------------------------------------------------------------------------

SUPPORTED_TIME_SIGNATURES = ["4/4", "3/4", "6/8"]
SUPPORTED_QUANTIZATIONS = ["1/8", "1/16", "1/32"]
SUPPORTED_KEY_MODES = ["auto", "none", "detected"]

DEFAULT_TIME_SIGNATURE = "4/4"
DEFAULT_QUANTIZATION = "1/16"
DEFAULT_KEY_MODE = "auto"

TEMPO_MIN = 30
TEMPO_MAX = 300
TEMPO_FALLBACK = 120  # usado apenas quando Etapa 3 não tem BPM (com warning)

KEY_CONFIDENCE_MIN = 0.45

# Amplitude/confidence/strength < 0.05 em TODOS os campos -> remove.
# Valor conservador: na dúvida, preserva a nota.
MIN_NOTE_STRENGTH = 0.05

# Gap máximo (em beats) entre duas notas de mesmo pitch para unir.
MERGE_GAP_BEATS = 0.15

# Duração mínima tratada como "extremamente curta" (segundos, pré-quantização).
MIN_NOTE_SECONDS = 0.02

# Tolerância para considerar duas notas "quase idênticas" (duplicatas).
DEDUP_START_TOL_SECONDS = 0.02
DEDUP_END_TOL_SECONDS = 0.03

# Tolerância de onset para agrupar acorde em `other` (beats, pós-quantização:
# notas com mesmo qstart exato formam o grupo; tolerância usada só no fallback
# pré-quantização quando grade ainda não aplicada).
CHORD_ONSET_TOL_BEATS = 0.06

MAX_VOICES = 4

# Mediana de pitch (MIDI) abaixo deste valor -> clave de fá para `other`.
OTHER_BASS_CLEF_MEDIAN = 55

# Mapeamento enarmônico -> grafia convencional com menos acidentes.
# Documentado: G# maior teórico (8 sustenidos) vira Ab maior (4 bemóis), etc.
ENHARMONIC_MAP = {
    ("G#", "major"): ("Ab", "major"),
    ("D#", "major"): ("Eb", "major"),
    ("A#", "major"): ("Bb", "major"),
    ("C#", "major"): ("Db", "major"),
    ("F#", "major"): ("Gb", "major"),
    ("G#", "minor"): ("Ab", "minor"),
    ("D#", "minor"): ("Eb", "minor"),
    ("A#", "minor"): ("Bb", "minor"),
}

PART_NAMES = {
    "vocals": ("Vocais", "Voc."),
    "bass": ("Baixo", "Bx."),
    "other": ("Outros", "Out."),
}

QUANT_TO_GRID_BEATS = {
    "1/8": 0.5,
    "1/16": 0.25,
    "1/32": 0.125,
}


# ---------------------------------------------------------------------------
# Validação de configuração
# ---------------------------------------------------------------------------

def validate_score_config(
    tempo: Any,
    time_signature: Any,
    quantization: Any,
    key_mode: Any,
) -> Dict[str, Any]:
    """Valida configuração da partitura. Retorna dict normalizado.

    Lança ValueError com mensagem amigável se inválido.
    `tempo` pode ser int/float ou None (None = resolver depois via Etapa 3).
    """
    if tempo is None:
        tempo_val: Optional[float] = None
    else:
        try:
            tempo_val = float(tempo)
        except (TypeError, ValueError):
            raise ValueError("BPM inválido. Use um valor entre 30 e 300.")
        if not (TEMPO_MIN <= tempo_val <= TEMPO_MAX):
            raise ValueError("BPM inválido. Use um valor entre 30 e 300.")
        # Normaliza: inteiro quando possível
        tempo_val = int(round(tempo_val)) if float(tempo_val).is_integer() else round(tempo_val, 2)

    if time_signature not in SUPPORTED_TIME_SIGNATURES:
        raise ValueError(
            f"Fórmula de compasso inválida. Permitidas: {', '.join(SUPPORTED_TIME_SIGNATURES)}."
        )
    if quantization not in SUPPORTED_QUANTIZATIONS:
        raise ValueError(
            f"Quantização inválida. Permitidas: {', '.join(SUPPORTED_QUANTIZATIONS)}."
        )
    if key_mode not in SUPPORTED_KEY_MODES:
        raise ValueError(
            f"key_mode inválido. Permitidos: {', '.join(SUPPORTED_KEY_MODES)}."
        )
    return {
        "tempo": tempo_val,
        "time_signature": time_signature,
        "quantization": quantization,
        "key_mode": key_mode,
    }


def config_key(tempo: Any, time_signature: str, quantization: str, key_mode: str) -> str:
    """Chave determinística da configuração (idempotência / regen).

    Mesmo config -> mesma chave; config diferente -> chave diferente.
    """
    raw = f"{tempo}|{time_signature}|{quantization}|{key_mode}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def grid_step_beats(quantization: str) -> float:
    try:
        return QUANT_TO_GRID_BEATS[quantization]
    except KeyError:
        raise ValueError(f"Quantização inválida: {quantization}")


def beats_per_measure(time_signature: str) -> float:
    """Duração do compasso em quarterLength (unidade music21)."""
    if time_signature == "4/4":
        return 4.0
    if time_signature == "3/4":
        return 3.0
    if time_signature == "6/8":
        # 6 colcheias = 3 semínimas em quarterLength
        return 3.0
    raise ValueError(f"Fórmula de compasso inválida: {time_signature}")


# ---------------------------------------------------------------------------
# Tonalidade / enarmonia
# ---------------------------------------------------------------------------

def normalize_key(key: Optional[str], mode: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Converte grafia teórica para equivalente convencional (menos acidentes)."""
    if not key or not mode:
        return key, mode
    mode_l = str(mode).lower()
    if mode_l not in ("major", "minor"):
        return key, mode
    mapped = ENHARMONIC_MAP.get((key, mode_l))
    if mapped:
        return mapped[0], mapped[1]
    return key, mode_l


def should_use_key(key_confidence: Any) -> bool:
    """Só usa armadura detectada se confiança >= KEY_CONFIDENCE_MIN."""
    try:
        conf = float(key_confidence) if key_confidence is not None else 0.0
    except (TypeError, ValueError):
        return False
    return conf >= KEY_CONFIDENCE_MIN


def resolve_key_signature(
    key: Optional[str],
    mode: Optional[str],
    key_confidence: Any,
    key_mode: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Retorna (key_norm, mode_norm, warning).

    - key_mode == "none" -> (None, None, None) armadura neutra sem warning.
    - key ausente -> neutra com warning informativo.
    - confiança < threshold -> neutra + warning padrão da spec.
    - senão -> tonalidade normalizada (enarmônica) sem warning.
    """
    if key_mode == "none":
        return None, None, None
    if not key or not mode:
        return None, None, "Tonalidade não detectada; armadura neutra utilizada."
    if not should_use_key(key_confidence):
        return (
            None,
            None,
            "Tonalidade detectada com baixa confiança; armadura neutra utilizada.",
        )
    k, m = normalize_key(key, mode)
    return k, m, None


# ---------------------------------------------------------------------------
# Limpeza de eventos (pré-quantização, em segundos)
# ---------------------------------------------------------------------------

def _strength_of(event: Dict[str, Any]) -> Optional[float]:
    """Máximo entre amplitude/confidence/strength, ou None se ausentes."""
    vals = []
    for field in ("amplitude", "confidence", "strength"):
        v = event.get(field)
        if v is None:
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return max(vals)


def clean_events(events: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Limpa eventos Basic Pitch.

    1. remove inválidos (end <= start, pitch fora 0-127, start/end ausentes);
    2. ordena por start;
    3. remove duplicatas quase idênticas (mesmo pitch, onsets/fins próximos);
    4. remove notas extremamente curtas (< MIN_NOTE_SECONDS);
    5. remove eventos muito fracos (todos os strengths < MIN_NOTE_STRENGTH).

    Retorna (cleaned, stats) com contadores por motivo.
    """
    stats = {
        "invalid": 0,
        "duplicates": 0,
        "too_short": 0,
        "weak": 0,
    }
    valid: List[Dict[str, Any]] = []
    for ev in events or []:
        try:
            start = float(ev.get("start"))
            end = float(ev.get("end"))
            pitch = int(ev.get("pitch"))
        except (TypeError, ValueError):
            stats["invalid"] += 1
            continue
        if not (end > start):
            stats["invalid"] += 1
            continue
        if pitch < 0 or pitch > 127:
            stats["invalid"] += 1
            continue
        duration = end - start
        if duration < MIN_NOTE_SECONDS:
            stats["too_short"] += 1
            continue
        strength = _strength_of(ev)
        if strength is not None and strength < MIN_NOTE_STRENGTH:
            stats["weak"] += 1
            continue
        valid.append(ev)

    valid.sort(key=lambda e: (float(e["start"]), float(e.get("end", 0)), int(e["pitch"])))

    deduped: List[Dict[str, Any]] = []
    for ev in valid:
        if deduped:
            prev = deduped[-1]
            try:
                same_pitch = int(prev["pitch"]) == int(ev["pitch"])
                d_start = abs(float(ev["start"]) - float(prev["start"]))
                d_end = abs(float(ev.get("end", 0)) - float(prev.get("end", 0)))
            except (TypeError, ValueError, KeyError):
                same_pitch, d_start, d_end = False, 999.0, 999.0
            if same_pitch and d_start <= DEDUP_START_TOL_SECONDS and d_end <= DEDUP_END_TOL_SECONDS:
                stats["duplicates"] += 1
                continue
        deduped.append(ev)
    return deduped, stats


# ---------------------------------------------------------------------------
# Conversão segundos -> beats + quantização
# ---------------------------------------------------------------------------

def seconds_to_beats(
    time_seconds: float, tempo: float, beat_offset: float = 0.0
) -> float:
    """Converte posição em segundos para beats.

    position_beats = (time_seconds - beat_offset) / (60 / BPM).
    Valores negativos (pickup antes do beat 0) são clamados para 0;
    o worker registra warning de anacruse nesses casos.
    """
    beat_dur = 60.0 / float(tempo)
    pos = (float(time_seconds) - float(beat_offset or 0.0)) / beat_dur
    return max(0.0, pos)


def quantize_value(value_beats: float, grid: float) -> float:
    return round(float(value_beats) / grid) * grid


def quantize_note(
    start_beats: float, end_beats: float, grid: float
) -> Tuple[float, float]:
    """Quantiza início/fim; duração mínima = um passo da grade."""
    qs = quantize_value(start_beats, grid)
    qe = quantize_value(end_beats, grid)
    qs = max(0.0, qs)
    if qe <= qs:
        qe = qs + grid
    # Arredonda para evitar floats como 0.30000000000000004
    qs = round(qs, 6)
    qe = round(qe, 6)
    return qs, qe


def merge_same_pitch(
    notes_beats: List[Dict[str, Any]],
    gap_threshold: float = MERGE_GAP_BEATS,
) -> Tuple[List[Dict[str, Any]], int]:
    """Une notas consecutivas de mesmo pitch com gap <= threshold (beats).

    Reduz fragmentação do Basic Pitch. Retorna (merged, merged_count).
    """
    if not notes_beats:
        return [], 0
    ordered = sorted(notes_beats, key=lambda n: (n["start"], n["end"]))
    merged: List[Dict[str, Any]] = [dict(ordered[0])]
    count = 0
    for note in ordered[1:]:
        prev = merged[-1]
        gap = float(note["start"]) - float(prev["end"])
        if int(note["pitch"]) == int(prev["pitch"]) and gap <= gap_threshold and gap >= -grid_floor(note):
            # Une: estende fim; preserva velocity máxima
            prev["end"] = max(float(prev["end"]), float(note["end"]))
            try:
                prev["velocity"] = max(int(prev.get("velocity", 0)), int(note.get("velocity", 0)))
            except (TypeError, ValueError):
                pass
            count += 1
        else:
            merged.append(dict(note))
    return merged, count


def grid_floor(note: Dict[str, Any]) -> float:
    """Pequena tolerância para overlaps na fusão (evita unir polifonia real)."""
    return 0.5


def resolve_monophonic_overlaps(
    notes_beats: List[Dict[str, Any]],
    grid: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Resolve overlaps em linha monofônica (vocals/bass).

    Encurta nota anterior até o início da próxima. Registra estatísticas;
    nada é apagado silenciosamente (notas degeneradas mantêm duração mínima).
    """
    stats = {"overlaps_resolved": 0, "large_overlaps": 0}
    if not notes_beats:
        return [], stats
    ordered = sorted(notes_beats, key=lambda n: (n["start"], n["end"]))
    result: List[Dict[str, Any]] = []
    for note in ordered:
        cur = dict(note)
        if result:
            prev = result[-1]
            if float(cur["start"]) < float(prev["end"]):
                overlap = float(prev["end"]) - float(cur["start"])
                stats["overlaps_resolved"] += 1
                if overlap > 1.0:
                    stats["large_overlaps"] += 1
                prev["end"] = float(cur["start"])
                if float(prev["end"]) <= float(prev["start"]):
                    prev["end"] = float(prev["start"]) + grid
        result.append(cur)
    return result, stats


def group_chords_other(
    notes_beats: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Agrupa notas de `other` com mesmo onset quantizado em acordes.

    Notas com (qstart, qend) idênticos -> um item "chord" com lista de pitches.
    Pitches duplicados dentro do acorde são removidos (ordenados, únicos):
    duplicatas surgem quando a quantização (ou o clamp de anacruse) mapeia
    eventos distintos para a mesma grade — um acorde com o mesmo pitch
    repetido é inválido e quebra a renderização no MuseScore.
    Notas com mesmo qstart mas qend distinto -> itens separados (o worker
    distribui em voices). Retorna (items, stats).
    """
    stats = {"chords": 0, "notes": 0, "duplicate_pitches_removed": 0}
    groups: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}
    for n in notes_beats:
        k = (round(float(n["start"]), 6), round(float(n["end"]), 6))
        groups.setdefault(k, []).append(n)
    items: List[Dict[str, Any]] = []
    for (qs, qe) in sorted(groups.keys()):
        members = sorted(groups[(qs, qe)], key=lambda m: int(m["pitch"]))
        if len(members) > 1:
            raw_pitches = [int(m["pitch"]) for m in members]
            uniq_pitches = sorted(set(raw_pitches))
            stats["duplicate_pitches_removed"] += len(raw_pitches) - len(uniq_pitches)
            items.append({
                "kind": "chord",
                "start": qs,
                "end": qe,
                "pitches": uniq_pitches,
                "velocity": max(int(m.get("velocity", 64)) for m in members),
            })
            stats["chords"] += 1
        else:
            m = members[0]
            items.append({
                "kind": "note",
                "start": qs,
                "end": qe,
                "pitches": [int(m["pitch"])],
                "velocity": int(m.get("velocity", 64)),
            })
            stats["notes"] += 1
    return items, stats


def choose_clef_other(pitches: List[int]) -> str:
    """Mediana dos pitches de `other`: < 55 -> 'bass', senão 'treble'."""
    if not pitches:
        return "treble"
    try:
        med = statistics.median(pitches)
    except statistics.StatisticsError:
        return "treble"
    return "bass" if med < OTHER_BASS_CLEF_MEDIAN else "treble"


def check_parts_nonempty(
    raw_counts: Dict[str, int],
    cleaned_counts: Dict[str, int],
    reparsed_counts: Dict[str, Dict[str, int]],
) -> Tuple[bool, List[str], List[str]]:
    """Validação forte por parte (pós-reparse do MusicXML).

    Para cada stem (vocals/bass/other), identificado pelo partName:
    - se raw == 0: nada a exigir (stem sem notas na transcrição);
    - se raw > 0 mas cleaned == 0: todos os eventos foram descartados por
      validações documentadas -> warning explícito, não falha;
    - se cleaned > 0 mas a parte reaberta tem 0 notes+chords -> erro técnico.

    Retorna (ok, errors, warnings). Não conta o Score inteiro: cada parte é
    verificada individualmente.
    """
    errors: List[str] = []
    warnings: List[str] = []
    for stem in ("vocals", "bass", "other"):
        pname = PART_NAMES[stem][0]
        raw = int(raw_counts.get(stem, 0))
        cleaned = int(cleaned_counts.get(stem, 0))
        rep = reparsed_counts.get(pname, {})
        total = int(rep.get("notes", 0)) + int(rep.get("chords", 0))
        if raw <= 0:
            continue
        if cleaned <= 0:
            warnings.append(
                f"Todos os {raw} evento(s) de '{pname}' foram descartados "
                f"na limpeza (inválidos/curtos/fracos); parte vazia com pausas."
            )
            continue
        if total <= 0:
            errors.append(
                f"Parte '{pname}' vazia no MusicXML reaberto, mas a transcrição "
                f"tem {raw} evento(s) ({cleaned} após limpeza). Falha técnica."
            )
    return (len(errors) == 0), errors, warnings
