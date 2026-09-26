"""
Testes Etapa 8.1 — hash de áudio, silêncio, timeout, progresso, RTF.
"""
import os
import time
from pathlib import Path

import numpy as np
import pytest

from backend.audio.long_audio import (
    SILENCE_MIN_FRACTION,
    SILENCE_RMS_THRESHOLD,
    TIMEOUT_BASE_SECONDS,
    TIMEOUT_MAX,
    TIMEOUT_MIN,
    TIMEOUT_MULTIPLIER,
    StageTimer,
    audio_cache_key,
    compute_audio_hash,
    compute_timeout,
    detect_silent_chunks,
    make_progress,
    measure_stage,
)


# ---------------------------------------------------------------------------
# Hash de áudio (item 77)
# ---------------------------------------------------------------------------

def test_hash_exists_and_hex():
    """SHA-256 retorna 64 chars hex."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(b"hello world" * 100)
        path = Path(f.name)
    try:
        h = compute_audio_hash(path)
        assert h is not None
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
    finally:
        path.unlink(missing_ok=True)


def test_hash_deterministic():
    """Mesmo arquivo → mesmo hash."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(os.urandom(64 * 1024))  # 64KB
        path = Path(f.name)
    try:
        h1 = compute_audio_hash(path)
        h2 = compute_audio_hash(path)
        assert h1 == h2
    finally:
        path.unlink(missing_ok=True)


def test_hash_missing_file():
    assert compute_audio_hash(Path("C:/nonexistent/file.mp3")) is None


def test_hash_large_file_streaming():
    """Arquivo de 5MB: hash sem carregar tudo em RAM (streaming)."""
    import tempfile
    payload = os.urandom(1024 * 1024)  # 1MB
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        for _ in range(5):  # 5MB
            f.write(payload)
        path = Path(f.name)
    try:
        h = compute_audio_hash(path)
        assert h is not None and len(h) == 64
    finally:
        path.unlink(missing_ok=True)


def test_cache_key_independent_stages():
    """Item 79: mudar estilo de arranjo NÃO invalida cache de Demucs."""
    h = "abc123"
    demucs_1 = audio_cache_key(h, "demucs", "v1")
    demucs_2 = audio_cache_key(h, "demucs", "v1")
    assert demucs_1 == demucs_2
    # Config de estilo diferente → chave de arranjo diferente
    arr1 = audio_cache_key(h, "arrangement", "v1", config="automatic")
    arr2 = audio_cache_key(h, "arrangement", "v1", config="rock")
    assert arr1 != arr2
    # Mas demucs não muda
    assert demucs_1 == audio_cache_key(h, "demucs", "v1")


def test_cache_key_version_invalidates():
    """Mudança de versão do algoritmo invalida o cache."""
    h = "abc123"
    v1 = audio_cache_key(h, "transcription", "v1")
    v2 = audio_cache_key(h, "transcription", "v2")
    assert v1 != v2


# ---------------------------------------------------------------------------
# Silêncio (itens 94-96)
# ---------------------------------------------------------------------------

SR = 22050


def test_silence_detected_in_quiet_region():
    """Região claramente silenciosa → detectada."""
    y = np.zeros(int(10 * SR))  # 10s de silêncio absoluto
    flags = detect_silent_chunks(y, SR, [0.0], [10.0])
    assert flags == [True]


def test_silence_not_detected_in_music():
    """Música real (senoide) NÃO é marcada como silêncio."""
    t = np.arange(int(10 * SR)) / SR
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    flags = detect_silent_chunks(y, SR, [0.0], [10.0])
    assert flags == [False]


def test_silence_not_detected_soft_intro():
    """Item 95: introdução suave (fade-in) NÃO é pulada."""
    t = np.arange(int(10 * SR)) / SR
    # Fade-in: amplitude cresce de 0.01 a 0.3
    amp = np.linspace(0.01, 0.3, len(t))
    y = amp * np.sin(2 * np.pi * 440 * t)
    flags = detect_silent_chunks(y, SR, [0.0], [10.0])
    assert flags == [False]


def test_silence_not_detected_quiet_voice():
    """Voz baixa (amplitude 0.01) não é pulada — threshold conservador."""
    t = np.arange(int(10 * SR)) / SR
    y = 0.01 * np.sin(2 * np.pi * 220 * t)
    flags = detect_silent_chunks(y, SR, [0.0], [10.0])
    # 0.01 > 0.003 (threshold) → não silêncio
    assert flags == [False]


def test_silence_partial_not_skipped():
    """Chunk com 50% silêncio + 50% música NÃO é pulado."""
    sr = SR
    silence = np.zeros(int(5 * sr))
    t = np.arange(int(5 * sr)) / sr
    music = 0.3 * np.sin(2 * np.pi * 440 * t)
    y = np.concatenate([silence, music])
    flags = detect_silent_chunks(y, sr, [0.0], [10.0])
    # Apenas 50% silencioso < 95% → não pula
    assert flags == [False]


def test_silence_multiple_chunks():
    """Múltiplos chunks: mistura de silêncio e música."""
    sr = SR
    t = np.arange(int(10 * sr)) / sr
    music = 0.3 * np.sin(2 * np.pi * 440 * t)
    silence = np.zeros(int(10 * sr))
    y = np.concatenate([silence, music, silence, music])
    # 4 chunks de 10s cada
    flags = detect_silent_chunks(y, sr,
                                 [0.0, 10.0, 20.0, 30.0],
                                 [10.0, 20.0, 30.0, 40.0])
    assert flags == [True, False, True, False]


def test_silence_thresholds_configurable():
    """Thresholds vêm de env vars."""
    assert SILENCE_RMS_THRESHOLD > 0
    assert 0 < SILENCE_MIN_FRACTION <= 1.0


# ---------------------------------------------------------------------------
# Timeout adaptativo (item 115)
# ---------------------------------------------------------------------------

def test_timeout_short_audio():
    """Música curta: timeout mínimo (base)."""
    t = compute_timeout(30.0)
    assert t >= TIMEOUT_MIN


def test_timeout_long_audio():
    """Música longa: timeout escala com duração."""
    t_short = compute_timeout(60.0)
    t_long = compute_timeout(600.0)
    assert t_long > t_short


def test_timeout_formula():
    """Fórmula: base + duration * multiplier."""
    t = compute_timeout(300.0, base=100, multiplier=2.0, tmin=0, tmax=10000)
    assert t == 100 + int(300.0 * 2.0)  # 700


def test_timeout_clamped():
    """Timeout limitado entre min e max."""
    t_low = compute_timeout(0.0, base=50, multiplier=0.1, tmin=100, tmax=500)
    assert t_low == 100
    t_high = compute_timeout(10000.0, base=100, multiplier=5.0,
                             tmin=100, tmax=500)
    assert t_high == 500


def test_timeout_defaults():
    """Defaults razoáveis: min 2min, max 1h."""
    assert TIMEOUT_MIN >= 120
    assert TIMEOUT_MAX <= 7200
    assert TIMEOUT_BASE_SECONDS > 0
    assert TIMEOUT_MULTIPLIER > 0


# ---------------------------------------------------------------------------
# Progresso (item 86)
# ---------------------------------------------------------------------------

def test_progress_payload():
    p = make_progress("Transcrevendo vocais", 252.0, 600.0, 5, 12)
    assert p["stage"] == "Transcrevendo vocais"
    assert p["progress_percent"] == 42
    assert p["processed_seconds"] == 252.0
    assert p["total_seconds"] == 600.0
    assert p["current_chunk"] == 5
    assert p["total_chunks"] == 12


def test_progress_zero():
    p = make_progress("Analisando", 0.0, 600.0, 0, 10)
    assert p["progress_percent"] == 0


def test_progress_complete():
    p = make_progress("Concluído", 600.0, 600.0, 10, 10)
    assert p["progress_percent"] == 100


def test_progress_over_limit_clamped():
    p = make_progress("Processando", 999.0, 600.0)
    assert p["progress_percent"] == 100
    assert p["processed_seconds"] == 600.0


def test_progress_zero_total():
    p = make_progress("Processando", 0.0, 0.0)
    assert p["progress_percent"] == 0


# ---------------------------------------------------------------------------
# RTF (item 105)
# ---------------------------------------------------------------------------

def test_rtf_calculation():
    timer = StageTimer()
    timer.record("demucs", 300.0, 120.0)  # 5min p/ 2min de áudio
    rtf = timer.rtf("demucs")
    assert rtf == pytest.approx(2.5)


def test_rtf_summary():
    timer = StageTimer()
    timer.record("demucs", 300.0, 120.0)
    timer.record("transcription", 60.0, 120.0)
    s = timer.summary()
    assert "demucs" in s and "transcription" in s
    assert s["demucs"]["rtf"] == pytest.approx(2.5)
    assert s["transcription"]["rtf"] == pytest.approx(0.5)


def test_rtf_missing_stage():
    timer = StageTimer()
    assert timer.rtf("nonexistent") is None


def test_measure_stage_context():
    """Context manager mede tempo automaticamente."""
    with measure_stage("test_stage", 1.0) as ctx:
        time.sleep(0.05)
    assert ctx.timer.records[0][0] == "test_stage"
    assert ctx.timer.records[0][1] >= 0.05
