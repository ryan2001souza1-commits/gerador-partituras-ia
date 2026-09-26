"""
Etapa 8.1 — Utilitários para áudio longo: hash, silêncio, progresso.

- SHA-256 streaming (nunca carrega arquivo inteiro em RAM);
- Detecção conservadora de silêncio para pular chunks vazios;
- Cálculo de timeout adaptativo para músicas longas;
- Métricas RTF (Real-Time Factor).
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Hash do áudio (item 77) — SHA-256 streaming
# ---------------------------------------------------------------------------

_HASH_CHUNK_SIZE = 1024 * 1024  # 1 MB por leitura


def compute_audio_hash(file_path: Path) -> Optional[str]:
    """SHA-256 do arquivo em streaming (nunca file.read() completo).

    Retorna hash hex (64 chars) ou None se o arquivo não existir.
    """
    if not file_path.is_file():
        return None
    sha = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(_HASH_CHUNK_SIZE)
                if not chunk:
                    break
                sha.update(chunk)
        return sha.hexdigest()
    except OSError:
        return None


def audio_cache_key(audio_hash: str, stage: str,
                    stage_version: str = "v1",
                    config: str = "") -> str:
    """Cache key por etapa: audio_hash + stage + versão + config (item 79).

    Mudança de dinâmica NÃO invalida Demucs (stages independentes).
    Mudança no algoritmo de transcrição invalida notation/arrangement.
    """
    raw = f"{audio_hash}|{stage}|{stage_version}|{config}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Detecção de silêncio (itens 94-96) — threshold conservador
# ---------------------------------------------------------------------------

# Threshold conservador: RMS abaixo disso por quase toda a janela = silêncio.
# Não pular introdução suave, voz baixa, instrumento solo fraco, fade-in.
SILENCE_RMS_THRESHOLD = float(os.getenv("SILENCE_RMS_THRESHOLD", "0.003"))
# Fração da janela que precisa estar silenciosa para pular (95%)
SILENCE_MIN_FRACTION = float(os.getenv("SILENCE_MIN_FRACTION", "0.95"))


def detect_silent_chunks(y, sr: int,
                         chunk_starts: List[float],
                         chunk_ends: List[float],
                         rms_threshold: Optional[float] = None
                         ) -> List[bool]:
    """Para cada chunk, verifica se é quase totalmente silencioso.

    Conservador: só marca como silencioso se RMS < threshold em >= 95%
    dos frames da janela. Não usa VAD de fala — energia musical (item 96).

    Args:
        y: áudio mono como numpy array.
        sr: sample rate.
        chunk_starts/ends: fronteiras dos chunks em segundos.

    Returns:
        Lista de bools: True = chunk silencioso (pode pular inferência).
    """
    import numpy as np
    if rms_threshold is None:
        rms_threshold = SILENCE_RMS_THRESHOLD

    # Frame length padrão do librosa (2048) com hop 512
    frame_len = 2048
    hop = 512
    silent_flags: List[bool] = []

    for start_s, end_s in zip(chunk_starts, chunk_ends):
        i0 = max(0, int(start_s * sr))
        i1 = min(len(y), int(end_s * sr))
        if i1 <= i0:
            silent_flags.append(True)
            continue
        seg = np.asarray(y[i0:i1], dtype=float)
        if seg.size < frame_len:
            # Segmento muito curto: verifica energia total
            rms = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
            silent_flags.append(rms < rms_threshold)
            continue
        # RMS por frame
        n_frames = 1 + (seg.size - frame_len) // hop
        rms_frames = np.zeros(n_frames)
        for k in range(n_frames):
            fr = seg[k * hop: k * hop + frame_len]
            rms_frames[k] = float(np.sqrt(np.mean(fr ** 2)))
        quiet_fraction = float(np.mean(rms_frames < rms_threshold))
        silent_flags.append(quiet_fraction >= SILENCE_MIN_FRACTION)

    return silent_flags


# ---------------------------------------------------------------------------
# Timeout adaptativo (item 115) — base + duração * multiplicador
# ---------------------------------------------------------------------------

# Fórmula documentada: timeout = base + duration_seconds * multiplier
# Limitado a [min, max]. Não reduz segurança (mínimo generoso).
TIMEOUT_BASE_SECONDS = int(os.getenv("TIMEOUT_BASE", "120"))
TIMEOUT_MULTIPLIER = float(os.getenv("TIMEOUT_MULTIPLIER", "2.0"))
TIMEOUT_MIN = int(os.getenv("TIMEOUT_MIN", "120"))
TIMEOUT_MAX = int(os.getenv("TIMEOUT_MAX", "3600"))  # 1h


def compute_timeout(duration_seconds: float,
                    base: Optional[int] = None,
                    multiplier: Optional[float] = None,
                    tmin: Optional[int] = None,
                    tmax: Optional[int] = None) -> int:
    """Timeout = base + duration * multiplier, limitado a [min, max].

    Ex.: música de 300s com base=120, mult=2 → 120 + 600 = 720s (12 min).
    """
    if base is None:
        base = TIMEOUT_BASE_SECONDS
    if multiplier is None:
        multiplier = TIMEOUT_MULTIPLIER
    if tmin is None:
        tmin = TIMEOUT_MIN
    if tmax is None:
        tmax = TIMEOUT_MAX
    raw = base + float(duration_seconds) * multiplier
    return int(max(tmin, min(tmax, raw)))


# ---------------------------------------------------------------------------
# Progresso real (item 86)
# ---------------------------------------------------------------------------

def make_progress(stage: str, processed_seconds: float,
                  total_seconds: float, current_chunk: int = 0,
                  total_chunks: int = 0) -> dict:
    """Constrói payload de progresso real para o job.

    Ex.:
    {
      "stage": "Transcrevendo vocais",
      "progress_percent": 42,
      "processed_seconds": 252,
      "total_seconds": 600,
      "current_chunk": 5,
      "total_chunks": 12
    }
    """
    total_seconds = max(float(total_seconds), 0.0)
    processed = max(0.0, min(float(processed_seconds), total_seconds))
    pct = int(100.0 * processed / total_seconds) if total_seconds > 0 else 0
    return {
        "stage": stage,
        "progress_percent": pct,
        "processed_seconds": round(processed, 1),
        "total_seconds": round(total_seconds, 1),
        "current_chunk": int(current_chunk),
        "total_chunks": int(total_chunks),
    }


# ---------------------------------------------------------------------------
# RTF — Real-Time Factor (item 105)
# ---------------------------------------------------------------------------

class StageTimer:
    """Mede tempo por etapa para cálculo de RTF."""

    def __init__(self):
        self.records: List[Tuple[str, float, float]] = []  # (stage, seconds, audio_seconds)

    def record(self, stage: str, elapsed_seconds: float,
               audio_seconds: float) -> None:
        self.records.append((stage, elapsed_seconds, audio_seconds))

    def rtf(self, stage: str) -> Optional[float]:
        """RTF = processing_seconds / audio_seconds para a etapa."""
        for s, elapsed, audio in self.records:
            if s == stage and audio > 0:
                return elapsed / audio
        return None

    def summary(self) -> dict:
        out = {}
        for stage, elapsed, audio in self.records:
            out[stage] = {
                "processing_seconds": round(elapsed, 2),
                "audio_seconds": round(audio, 2),
                "rtf": round(elapsed / audio, 3) if audio > 0 else None,
            }
        return out


def measure_stage(stage: str, audio_seconds: float):
    """Decorator/context manager para medir uma etapa automaticamente."""
    class _Ctx:
        def __init__(self):
            self.timer = StageTimer()
            self._t0 = None
        def __enter__(self):
            self._t0 = time.time()
            return self
        def __exit__(self, *exc):
            elapsed = time.time() - self._t0 if self._t0 else 0.0
            self.timer.record(stage, elapsed, audio_seconds)
            return False
    return _Ctx()
