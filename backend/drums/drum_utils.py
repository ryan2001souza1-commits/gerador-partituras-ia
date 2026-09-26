"""
Etapa 8 — DSP percussivo puro (roda no .venv principal: librosa/numpy/scipy).

Pipeline: drums.wav (Demucs) -> mono 22050 -> onset envelope -> onsets ->
features espectrais por onset -> classificação heurística -> quantização
rítmica (BPM/beat grid das Etapas 3/7) -> eventos {time, instrument,
confidence, strength}.

Sem Basic Pitch, sem pitch tracking, sem modelo pesado. Determinístico.
Bandas (Hz): LOW 30-180, LOWMID 180-500, MID 500-3000, HIGH 3000-12000
(limitadas pelo Nyquist real).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DRUM_SR = 22050
DRUM_VERSION = "drum-spectral-v1"

# Classes estáveis + fallback documentado.
DRUM_CLASSES = ["kick", "snare", "closed_hihat", "open_hihat", "crash",
                "tom_low", "tom_mid", "tom_high", "unknown_percussion"]

# GM (General MIDI) para referência/exportação.
GM_MAP = {"kick": 36, "snare": 38, "closed_hihat": 42, "open_hihat": 46,
          "crash": 49, "tom_low": 45, "tom_mid": 47, "tom_high": 50}

# Janela de análise por onset (s) e decaimento máximo medido (s).
ONSET_WINDOW = 0.15
DECAY_CAP = 0.8
# Dedupe: mesma classe dentro desta janela (s) funde (flam/duplicata).
FLAM_WINDOW = 0.03
# Retrigger de decay por classe (s): mesma classe, mais fraco que o limiar
# do vizinho mais forte, funde. Kick usa limiar mais alto (0.95) e janela
# maior (0.35s) para fundir artefatos de decay que geram onsets espúrios
# em colcheia (ex.: 0.25s após o ataque real).
RETRIGGER_WINDOW = {"kick": 0.35, "tom_low": 0.20, "tom_mid": 0.20,
                    "tom_high": 0.20, "crash": 0.25, "snare": 0.08,
                    "closed_hihat": 0.06, "open_hihat": 0.06,
                    "unknown_percussion": 0.06}
RETRIGGER_RATIO = {"kick": 2.0, "default": 0.85}
# Ghost: abaixo desta força/confiança (natural), fora de padrão, sai.
GHOST_MIN_STRENGTH = 0.35
GHOST_MIN_CONF = 0.6
# Confiança mínima para classe; abaixo vira unknown (natural: descarta se <0.2).
CLASS_MIN_SCORE = 0.35
UNKNOWN_KEEP_SCORE = 0.2


def detect_onsets(y: np.ndarray, sr: int) -> Tuple[List[float], List[float]]:
    """Onset times + forças (envelope normalizado 0..1)."""
    import librosa
    if y is None or y.size == 0:
        return [], []
    y = np.asarray(y, dtype=float)
    peak = float(np.max(np.abs(y)))
    if peak < 1e-6:
        return [], []
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr,
                                       units="time", backtrack=True,
                                       pre_max=3, post_max=3, pre_avg=12,
                                       post_avg=12, delta=0.08, wait=4)
    if len(times) == 0:
        return [], []
    frames = librosa.time_to_frames(times, sr=sr)
    env = np.asarray(onset_env, dtype=float)
    emax = float(np.max(env)) or 1.0
    # Força no pico local (backtrack retorna o vale anterior ao ataque).
    strengths = []
    for f in frames:
        f0 = max(0, min(int(f), len(env) - 1))
        f1 = min(len(env), f0 + 6)
        strengths.append(float(np.max(env[f0:f1])) / emax)
    # Remove onsets em silêncio quase total (energia local ~0).
    hop = 512
    keep_t, keep_s = [], []
    for t, s in zip(times, strengths):
        i0 = max(0, int((t - 0.02) * sr))
        i1 = min(len(y), int((t + 0.12) * sr))
        rms = float(np.sqrt(np.mean(y[i0:i1] ** 2))) if i1 > i0 else 0.0
        if rms > 0.005 * peak:
            keep_t.append(float(t))
            keep_s.append(float(s))
    return keep_t, keep_s


def _stft_once(y: np.ndarray, sr: int):
    import librosa
    n_fft = 2048
    hop = 512
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return S, freqs, hop


def extract_features(y: np.ndarray, sr: int, onset_time: float,
                     S=None, freqs=None, hop: int = 512) -> Dict[str, float]:
    """Energia por banda + centroid + rolloff + zcr + decay + transient."""
    import librosa
    if S is None or freqs is None:
        S, freqs, hop = _stft_once(y, sr)
    fr = int(round(onset_time * sr / hop))
    fr = max(0, min(fr, S.shape[1] - 1))
    n_win = max(1, int(round(ONSET_WINDOW * sr / hop)))
    sl = S[:, fr:fr + n_win]
    e = np.sum(sl, axis=1) + 1e-12
    nyq = sr / 2.0
    bands = [(30.0, 180.0), (180.0, 500.0), (500.0, 3000.0), (3000.0, min(12000.0, nyq))]
    fracs = []
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        fracs.append(float(np.sum(e[m]) / float(np.sum(e))))
    low, lowmid, mid, high = fracs
    centr = float(np.sum(freqs * e) / float(np.sum(e)))
    cum = np.cumsum(np.sort(e)[::-1])
    roll = float(freqs[np.searchsorted(np.cumsum(e / float(np.sum(e))), 0.85)] or 0.0)
    i0 = max(0, int(onset_time * sr))
    i1 = min(len(y), i0 + int(0.1 * sr))
    seg = y[i0:i1] if i1 > i0 + 8 else y[max(0, i0 - 64):i0 + 64]
    zcr = float(np.mean(np.abs(np.diff(np.sign(seg))))) / 2.0 if len(seg) > 8 else 0.0
    # Decay: quadros até energia < 30% do pico (cap).
    col_e = np.sum(S, axis=0) + 1e-12
    peak = float(col_e[fr])
    decay_frames = 0
    for k in range(fr, min(S.shape[1], fr + int(DECAY_CAP * sr / hop))):
        decay_frames = k - fr
        if float(col_e[k]) < 0.3 * peak:
            break
    decay_ms = 1000.0 * decay_frames * hop / sr
    total = float(np.sum(e))
    return {"low": low, "lowmid": lowmid, "mid": mid, "high": high,
            "centroid": centr, "rolloff": roll, "zcr": zcr,
            "decay_ms": decay_ms, "energy": total}


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _tom_subclass(y, sr: Optional[int], onset_time: Optional[float], centr: float) -> str:
    """low/mid/high pela fundamental (yin); fallback: bandas de centroid."""
    try:
        if y is not None and sr and onset_time is not None:
            import librosa
            i0 = max(0, int(float(onset_time) * sr))
            seg = np.asarray(y[i0:i0 + int(0.15 * sr)], dtype=np.float32)
            if seg.size > 512 and float(np.max(np.abs(seg))) > 1e-4:
                f0 = librosa.yin(seg, fmin=40, fmax=800, sr=sr)
                f0v = f0[np.isfinite(f0)]
                if f0v.size:
                    med = float(np.median(f0v))
                    if med < 135.0:
                        return "tom_low"
                    if med < 225.0:
                        return "tom_mid"
                    return "tom_high"
    except Exception:
        pass
    if centr < 250.0:
        return "tom_low"
    if centr < 500.0:
        return "tom_mid"
    return "tom_high"


def classify_hit(feats: Dict[str, float], transient: float, y=None,
                 sr: Optional[int] = None,
                 onset_time: Optional[float] = None) -> Tuple[str, float, Dict[str, float]]:
    """Scores heurísticos por classe; retorna (classe, confiança, scores).

    Confiança = margem do vencedor (não é certeza científica).
    """
    low, lowmid = feats["low"], feats["lowmid"]
    mid, high = feats["mid"], feats["high"]
    centr = feats["centroid"]
    zcr = feats["zcr"]
    decay = feats["decay_ms"]
    broad = 1.0 - max(low, lowmid, mid, high)  # espalhamento espectral
    hat_base = (0.55 * high + 0.20 * (1 - _clip(low * 3.0))
                + 0.15 * _clip(centr / 9000.0)
                + 0.10 * (1 - _clip(decay / 300.0))
                - 0.45 * _clip(mid / 0.3))  # gate: com mid relevante não é hat
    crash_base = (0.30 * high + 0.25 * _clip(decay / 800.0)
                  + 0.20 * broad + 0.15 * _clip(feats["energy"] / 400.0, 0, 1)
                  + 0.10 * (1 - low))
    if decay < 200.0:
        crash_base -= 0.25  # transiente curto não é crash
    else:
        crash_base += 0.20 * _clip((decay - 350.0) / 450.0)
    tom_base = (0.45 * lowmid + 0.20 * mid + 0.10 * low
                + 0.15 * (1 - _clip(abs(centr - 350.0) / 700.0))
                + 0.10 * transient
                - 0.50 * _clip((high - 0.25) / 0.3))  # muito agudo não é tom
    if lowmid > 0.14 and lowmid > low * 0.25:
        tom_base += 0.30  # dominância lowmid tonal: tom, não kick
    scores = {
        "kick": (0.55 * low + 0.20 * (1 - _clip(centr / 4000.0))
                 + 0.15 * (1 - _clip(decay / 400.0)) + 0.10 * transient
                 - 0.30 * _clip(lowmid / 0.15)),
        "snare": (0.38 * mid + 0.18 * lowmid + 0.18 * broad
                  + 0.20 * _clip(zcr / 0.25) + 0.17 * transient),
        "closed_hihat": hat_base,
        "crash": crash_base,
        "tom": tom_base,
    }
    # open_hihat deriva do closed quando há sustain real (mas não crash longo).
    if 250.0 < decay < 700.0:
        scores["open_hihat"] = hat_base + 0.22
    else:
        scores["open_hihat"] = hat_base - 0.15
    # Tom por registro (só se tom vencer).
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    best, second = ordered[0], ordered[1]
    if best[0] == "tom":
        sub = _tom_subclass(y, sr, onset_time, centr)
        best = (sub, best[1])
    margin = best[1] - second[1]
    confidence = _clip(best[1] * 0.7 + margin * 1.5)
    if best[1] < CLASS_MIN_SCORE:
        return "unknown_percussion", _clip(best[1]), scores
    return best[0], confidence, scores


def quantize_beat(t: float, tempo: float, beat_offset: float, fine: bool) -> float:
    """Segundos -> beats quantizados (grade 1/8; 1/16 com evidência)."""
    spb = 60.0 / float(tempo)
    b = max(0.0, (float(t) - float(beat_offset or 0.0)) / spb)
    grid = 0.25 if fine else 0.5
    return round(round(b / grid) * grid, 6)


# Stacking: um onset pode conter 2 classes reais (ex. kick + hi-hat).
# Segunda classe só é emitida com evidência forte e par compatível.
STACK_RULES = (("kick", "closed_hihat", 0.50),
               ("snare", "closed_hihat", 0.55),
               ("closed_hihat", "kick", 0.35))


def _stack_secondary(best: str, scores: Dict[str, float]):
    """Retorna (classe, confiança) extra ou None. Máx 2 eventos por onset."""
    for first, second, thresh in STACK_RULES:
        if best == first and scores.get(second, 0.0) >= thresh:
            return second, _clip(scores[second] * 0.9)
    return None


def _dedupe_events(raw: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Funde mesma classe dentro de FLAM_WINDOW (mantém o mais forte)."""
    raw = sorted(raw, key=lambda e: (e["time"], e["instrument"]))
    deduped: List[Dict[str, Any]] = []
    merged = 0
    for e in raw:
        if deduped and e["instrument"] == deduped[-1]["instrument"] \
                and e["time"] - deduped[-1]["time"] < FLAM_WINDOW:
            merged += 1
            if e["strength"] > deduped[-1]["strength"]:
                deduped[-1] = e
            continue
        deduped.append(e)
    return deduped, merged


def transcribe_drums(y: np.ndarray, sr: int, tempo: float, beat_offset: float,
                     time_signature: str = "4/4", profile: str = "detailed"
                     ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Onsets -> classificação -> quantização -> eventos + stats."""
    stats: Dict[str, Any] = {"raw_onsets": 0, "classified_events": 0,
                             "discarded_events": 0, "simultaneous_hits": 0,
                             "ghost_notes_removed": 0, "flams_merged": 0}
    for k in ["kick", "snare", "closed_hihat", "open_hihat", "crash",
              "tom_low", "tom_mid", "tom_high", "unknown_percussion"]:
        stats[f"{k}_count"] = 0
    times, strengths = detect_onsets(y, sr)
    stats["raw_onsets"] = len(times)
    if not times:
        stats.update(mean_confidence=0.0, quantization_error_mean_ms=0.0,
                     quantization_error_p95_ms=0.0)
        return [], stats
    S, freqs, hop = _stft_once(y, sr)
    smax = max(strengths) or 1.0
    raw = []
    for t, s in zip(times, strengths):
        feats = extract_features(y, sr, t, S=S, freqs=freqs, hop=hop)
        cls, conf, scores = classify_hit(feats, _clip(s / smax), y=y, sr=sr, onset_time=t)
        raw.append({"time": round(float(t), 6), "strength": round(float(s / smax), 4),
                    "instrument": cls, "confidence": round(float(conf), 3),
                    "decay_ms": round(float(feats["decay_ms"]), 1)})
        # Stacking: kick+hat / snare+hat simultâneos (máx 2 por onset).
        second = _stack_secondary(cls, scores)
        if second is not None:
            raw.append({"time": round(float(t), 6), "strength": round(float(s / smax), 4),
                        "instrument": second[0], "confidence": round(float(second[1]), 3),
                        "decay_ms": round(float(feats["decay_ms"]), 1)})
    # Dedupe mesma classe em FLAM_WINDOW (mantém o mais forte).
    deduped, n_flam = _dedupe_events(raw)
    stats["flams_merged"] += n_flam
    # Retrigger de decay: mesma classe, fraco perto de hit forte, funde.
    deduped, n_ret = _merge_retriggers(deduped, tempo, beat_offset)
    stats["flams_merged"] += n_ret
    # Slots de padrão (para boost + isenção de ghost).
    pattern_slots = _pattern_slots(deduped, tempo, beat_offset, time_signature) \
        if profile == "natural" else set()
    # Ghosts (natural): muito fracos e fora de padrão saem; detailed preserva.
    kept = []
    for e in deduped:
        if profile == "natural" and e["instrument"] in ("kick", "snare") \
                and e["strength"] < GHOST_MIN_STRENGTH and e["confidence"] < GHOST_MIN_CONF \
                and not _in_pattern(e, pattern_slots, tempo, beat_offset, time_signature):
            stats["ghost_notes_removed"] += 1
            stats["discarded_events"] += 1
            continue
        if e["instrument"] == "unknown_percussion":
            if profile == "natural" and e["confidence"] < UNKNOWN_KEEP_SCORE:
                stats["discarded_events"] += 1
                continue
        kept.append(e)
    # Boost de padrão repetitivo (natural): slots recorrentes +0.15.
    if profile == "natural" and kept:
        _pattern_boost(kept, tempo, beat_offset, time_signature)
    # Quantização (1/8 pref; 1/16 com subdivisão consistente).
    spb = 60.0 / float(tempo)
    starts = sorted(e["time"] for e in kept)
    events = []
    for e in kept:
        slot8 = int(e["time"] / (spb * 0.5))
        evidence = any(int(t / (spb * 0.5)) == slot8 and abs(t - e["time"]) >= 0.15 * spb
                       and abs(t - e["time"]) > 1e-9 for t in starts)
        fine = evidence
        qb = quantize_beat(e["time"], tempo, beat_offset, fine)
        beat = max(0.0, (e["time"] - (beat_offset or 0.0)) / spb)
        err_ms = (e["time"] - (beat_offset_or_zero(beat_offset) + qb * spb)) * 1000.0
        events.append({
            "time": e["time"], "quantized_time": round(beat_offset_or_zero(beat_offset) + qb * spb, 6),
            "beat": round(qb, 6), "instrument": e["instrument"],
            "confidence": e["confidence"], "strength": e["strength"],
            "original_time": e["time"], "timing_error_ms": round(err_ms, 2),
        })
    events.sort(key=lambda ev: (ev["beat"], ev["instrument"]))
    # Simultaneidade: onsets com >=2 classes no mesmo beat quantizado.
    by_beat: Dict[float, set] = {}
    for ev in events:
        by_beat.setdefault(ev["beat"], set()).add(ev["instrument"])
    stats["simultaneous_hits"] = sum(1 for v in by_beat.values() if len(v) >= 2)
    stats["classified_events"] = len(events)
    for e in events:
        stats[f"{e['instrument']}_count"] = stats.get(f"{e['instrument']}_count", 0) + 1
    if events:
        confs = [e["confidence"] for e in events]
        errs = sorted(abs(e["timing_error_ms"]) for e in events)
        stats["mean_confidence"] = round(sum(confs) / len(confs), 3)
        stats["quantization_error_mean_ms"] = round(sum(errs) / len(errs), 2)
        stats["quantization_error_p95_ms"] = round(errs[min(len(errs) - 1, int(len(errs) * 0.95))], 2)
    else:
        stats.update(mean_confidence=0.0, quantization_error_mean_ms=0.0,
                     quantization_error_p95_ms=0.0)
    return events, stats


def beat_offset_or_zero(x: Optional[float]) -> float:
    try:
        return float(x or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _slot_index(t: float, tempo: float, beat_offset: float,
                time_signature: str) -> Optional[Tuple[int, int]]:
    """(compasso, slot) de um onset; None se antes do offset."""
    spb = 60.0 / float(tempo)
    b = (t - (beat_offset or 0.0)) / spb
    if b < 0:
        return None
    beats_per_bar = 3.0 if time_signature == "6/8" else 4.0
    slots_per_bar = 6 if time_signature == "6/8" else 8
    return (int(b // beats_per_bar),
            int((b % beats_per_bar) // (beats_per_bar / slots_per_bar)))


def _slot8_index(t: float, tempo: float, beat_offset: float) -> int:
    """Índice do slot 1/8 (grade grossa) para um tempo."""
    spb = 60.0 / float(tempo)
    b = max(0.0, (float(t) - float(beat_offset or 0.0)) / spb)
    return int(b / 0.5)  # 1/8 = 0.5 beats


def _is_on_quarter_beat(t: float, tempo: float, beat_offset: float, tol: float = 0.08) -> bool:
    """Verifica se o tempo quantiza para um beat inteiro (colcheia em 4/4)."""
    spb = 60.0 / float(tempo)
    b = (float(t) - float(beat_offset or 0.0)) / spb
    return abs(b - round(b)) < tol


def _is_on_eighth_beat(t: float, tempo: float, beat_offset: float, tol: float = 0.12) -> bool:
    """Verifica se o tempo quantiza para um meio-beat (ex.: 1.5, 2.5 em 4/4)."""
    spb = 60.0 / float(tempo)
    b = (float(t) - float(beat_offset or 0.0)) / spb
    return abs(b - round(b)) > 0.2 and abs(b - (round(b * 2) / 2.0)) < tol


def _merge_retriggers(
    events: List[Dict[str, Any]], tempo: float, beat_offset: float
) -> Tuple[List[Dict[str, Any]], int]:
    """Funde retrigger de decay: mesma classe, dentro da janela da classe e
    mais fraco que RETRIGGER_RATIO x o vizinho (mantém o ataque anterior).
    Rastreia o último evento POR CLASSE. Não funde se ambos quantizam em
    beats fortes (inteiros) distintos. Funde se o último está no beat forte
    e o atual está no off-beat (artifact de decay típico ~300ms)."""
    out: List[Dict[str, Any]] = []
    merged = 0
    last_by_class: Dict[str, Dict[str, Any]] = {}
    for e in sorted(events, key=lambda x: x["time"]):
        inst = e["instrument"]
        last = last_by_class.get(inst)
        if last is not None:
            w = RETRIGGER_WINDOW.get(inst, 0.06)
            ratio = RETRIGGER_RATIO.get(inst, RETRIGGER_RATIO["default"])
            dt = e["time"] - last["time"]
            # Heurística: artifact de decay em kick/snare costuma cair ~300ms
            # após o ataque real (beat forte) e quantiza no off-beat (meio-beat).
            last_on_q = _is_on_quarter_beat(last["time"], tempo, beat_offset)
            now_on_q = _is_on_quarter_beat(e["time"], tempo, beat_offset)
            now_on_eighth = _is_on_eighth_beat(e["time"], tempo, beat_offset)
            # Se último no beat forte (inteiro) e atual no off-beat (meio-beat)
            # e dt consistente com decay (0.2-0.35s), funde mesmo em slots diff.
            decay_artifact = (
                last_on_q and not now_on_q and now_on_eighth and
                0.18 < dt < w
            )
            same_slot = not decay_artifact and _slot8_index(last["time"], tempo, beat_offset) == _slot8_index(e["time"], tempo, beat_offset)
            if (same_slot or decay_artifact) and 0 < dt < w and e["strength"] < ratio * (last["strength"] or 1e-6):
                merged += 1
                # Para decay artifact, SEMPRE mantém o evento no beat forte (último),
                # não substitui pelo artifact mesmo se este for ligeiramente mais forte.
                if not decay_artifact and e["strength"] > last["strength"]:
                    for i, o in enumerate(out):
                        if o is last:
                            out[i] = e
                            break
                    last_by_class[inst] = e
                continue
        out.append(e)
        last_by_class[inst] = e
    return out, merged


def _pattern_slots(events: List[Dict[str, Any]], tempo: float,
                   beat_offset: float, time_signature: str) -> set:
    """Slots (classe, compasso%?, slot) recorrentes: >=2 e >=50% dos compassos."""
    spb = 60.0 / float(tempo)
    beats_per_bar = 3.0 if time_signature == "6/8" else 4.0
    slots_per_bar = 6 if time_signature == "6/8" else 8
    bars: Dict[Tuple[str, int], int] = {}
    n_bars = 0
    for e in events:
        idx = _slot_index(e["time"], tempo, beat_offset, time_signature)
        if idx is None:
            continue
        bar_idx, slot_idx = idx
        bars[(e["instrument"], slot_idx)] = bars.get((e["instrument"], slot_idx), 0) + 1
        n_bars = max(n_bars, bar_idx + 1)
    if n_bars < 2:
        return set()
    return {k for k, v in bars.items() if v >= max(2, math.ceil(n_bars * 0.5))}


def _pattern_boost(events: List[Dict[str, Any]], tempo: float,
                   beat_offset: float, time_signature: str) -> None:
    """Slots recorrentes por classe ganham +0.15 confiança (in-place)."""
    slots = _pattern_slots(events, tempo, beat_offset, time_signature)
    if not slots:
        return
    for e in events:
        idx = _slot_index(e["time"], tempo, beat_offset, time_signature)
        if idx is not None and (e["instrument"], idx[1]) in slots:
            e["confidence"] = round(min(1.0, e["confidence"] + 0.15), 3)


def _in_pattern(e: Dict[str, Any], slots: set, tempo: float,
                beat_offset: float, time_signature: str) -> bool:
    idx = _slot_index(e["time"], tempo, beat_offset, time_signature)
    if idx is None:
        return False
    return (e["instrument"], idx[1]) in slots
    """Slots recorrentes por classe ganham +0.15 confiança (in-place)."""
    spb = 60.0 / float(tempo)
    slots_per_bar = 6 if time_signature == "6/8" else 8
    slot_len = spb * (0.5 if time_signature != "6/8" else 1.0 / 3.0)
    bars: Dict[Tuple[str, int], int] = {}
    n_bars = 0
    beats_per_bar = 3.0 if time_signature == "6/8" else 4.0
    for e in events:
        b = (e["time"] - (beat_offset or 0.0)) / spb
        if b < 0:
            continue
        bar_idx = int(b // beats_per_bar)
        slot_idx = int((b % beats_per_bar) // (beats_per_bar / slots_per_bar))
        bars[(e["instrument"], slot_idx)] = bars.get((e["instrument"], slot_idx), 0) + 1
        n_bars = max(n_bars, bar_idx + 1)
    if n_bars < 2:
        return
    for e in events:
        b = (e["time"] - (beat_offset or 0.0)) / spb
        if b < 0:
            continue
        slot_idx = int((b % beats_per_bar) // (beats_per_bar / slots_per_bar))
        if bars.get((e["instrument"], slot_idx), 0) >= max(2, math.ceil(n_bars * 0.5)):
            e["confidence"] = round(min(1.0, e["confidence"] + 0.15), 3)
