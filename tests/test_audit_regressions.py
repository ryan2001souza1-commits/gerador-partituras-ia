"""
Testes de regressão — Auditoria extrema (bugs #4-#9).

BUG #4: NaN/Infinity passam pelo stitching sem validação.
BUG #5: Stitching O(n^2) explode com muitos eventos (agora O(n*w)).
BUG #6: Config inválida (0, negativo, NaN) em create_chunks sem erro.
BUG #7: Env var CHUNK_DURATION='abc' crasha o import.
BUG #8: drums/ ausente do .gitignore.
BUG #9: Frontend polling não cancelado em novo upload (cross-file contamination).
"""
import json
import math
import os
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# BUG #4: NaN/Infinity no pipeline
# ---------------------------------------------------------------------------

def _load_worker():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bp_chunked_audit",
        str(BASE_DIR / "backend" / "workers" / "basic_pitch_worker_chunked.py"))
    bp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bp)
    return bp


@pytest.fixture(scope="module")
def bp():
    return _load_worker()


def test_bug4_nan_start_discarded(bp):
    """BUG #4: evento com start=NaN é descartado, não propagado."""
    ev = (float('nan'), 2.0, 60, 0.5)
    result = bp._raw_event_to_json(ev, chunk_start=0.0)
    assert result is None, f"NaN start deveria ser None, got {result}"


def test_bug4_inf_start_discarded(bp):
    """BUG #4: evento com start=Infinity é descartado."""
    ev = (float('inf'), float('inf'), 60, 0.5)
    result = bp._raw_event_to_json(ev, chunk_start=0.0)
    assert result is None


def test_bug4_nan_end_discarded(bp):
    ev = (1.0, float('nan'), 60, 0.5)
    result = bp._raw_event_to_json(ev, chunk_start=0.0)
    assert result is None


def test_bug4_nan_amplitude_defaults_not_max(bp):
    """BUG #4: NaN amplitude vira 0.5 (default), não 1.0 (max)."""
    ev = (1.0, 2.0, 60, float('nan'))
    result = bp._raw_event_to_json(ev, chunk_start=0.0)
    assert result is not None
    assert result["amplitude"] == 0.5, (
        f"NaN amplitude deveria ser 0.5, got {result['amplitude']}")


def test_bug4_inf_amplitude_defaults_not_max(bp):
    ev = (1.0, 2.0, 60, float('inf'))
    result = bp._raw_event_to_json(ev, chunk_start=0.0)
    assert result is not None
    assert result["amplitude"] == 0.5


def test_bug4_invalid_pitch_discarded(bp):
    ev = (1.0, 2.0, 200, 0.5)  # pitch > 127
    result = bp._raw_event_to_json(ev, chunk_start=0.0)
    assert result is None
    ev2 = (1.0, 2.0, -1, 0.5)
    result2 = bp._raw_event_to_json(ev2, chunk_start=0.0)
    assert result2 is None


def test_bug4_negative_start_clamped_to_zero(bp):
    """start negativo (modelo tolerante) é clampado para 0."""
    ev = (-0.5, 2.0, 60, 0.5)
    result = bp._raw_event_to_json(ev, chunk_start=0.0)
    assert result is not None
    assert result["start"] == 0.0


def test_bug4_end_before_start_fixed(bp):
    """end < start: end é ajustado para start + 0.01 (mínimo viável)."""
    ev = (5.0, 3.0, 60, 0.5)  # end < start!
    result = bp._raw_event_to_json(ev, chunk_start=0.0)
    assert result is not None
    assert result["start"] < result["end"]
    assert result["duration"] > 0


def test_bug4_stitch_discards_nan_events(bp):
    """Stitching descarta eventos NaN ANTES de processar."""
    events = [
        {"start": float('nan'), "end": 2.0, "pitch": 60,
         "confidence": 0.9, "amplitude": 0.8, "_chunk": 0},
        {"start": 1.0, "end": 3.0, "pitch": 62,
         "confidence": 0.9, "amplitude": 0.8, "_chunk": 0},
        {"start": float('inf'), "end": float('inf'), "pitch": 64,
         "confidence": 0.9, "amplitude": 0.8, "_chunk": 0},
    ]
    stitched, stats = bp.stitch_global_events(events)
    assert stats["invalid_events_discarded"] == 2
    assert len(stitched) == 1  # Apenas o evento válido
    assert all(math.isfinite(n["start"]) for n in stitched)


def test_bug4_pitch_bends_validated(bp):
    """Pitch bends com valores inválidos são descartados."""
    ev = (1.0, 2.0, 60, 0.5, [0, float('nan'), 100])
    result = bp._raw_event_to_json(ev, chunk_start=0.0)
    assert result is not None
    assert "pitch_bends" not in result  # NaN bend descarta a lista toda


def test_bug4_valid_events_unchanged(bp):
    """Eventos válidos normais não são afetados pela validação."""
    ev = (1.5, 3.2, 60, 0.8, [0, 100])
    result = bp._raw_event_to_json(ev, chunk_start=60.0)
    assert result is not None
    assert result["start"] == 61.5
    assert result["end"] == 63.2
    assert result["pitch"] == 60
    assert result["pitch_bends"] == [0, 100]


# ---------------------------------------------------------------------------
# BUG #5: Stitching O(n^2) → O(n*w) com janela deslizante
# ---------------------------------------------------------------------------

def test_bug5_stitching_10k_events_fast():
    """BUG #5: 10.000 eventos devem levar < 5s (antes: 16s)."""
    import random
    bp = _load_worker()
    rng = random.Random(42)
    events = []
    for i in range(10000):
        chunk_idx = rng.randint(0, 20)
        chunk_start = chunk_idx * 57.0
        local_start = rng.uniform(0, 55)
        local_end = local_start + rng.uniform(0.1, 3.0)
        events.append({
            "start": chunk_start + local_start,
            "end": chunk_start + local_end,
            "pitch": rng.randint(40, 80),
            "confidence": rng.uniform(0.3, 0.99),
            "amplitude": rng.uniform(0.2, 0.99),
            "_chunk": chunk_idx,
        })
    t0 = time.time()
    stitched, stats = bp.stitch_global_events(events)
    elapsed = time.time() - t0
    assert elapsed < 5.0, (
        f"Stitching de 10k eventos levou {elapsed:.1f}s (deveria ser < 5s "
        f"após otimização de janela deslizante)")
    assert len(stitched) > 0


def test_bug5_stitching_correctness_preserved(bp):
    """A otimização não altera o resultado para casos normais."""
    # Nota cruzando fronteira
    events = [
        {"start": 58.2, "end": 60.0, "pitch": 60, "confidence": 0.9,
         "amplitude": 0.8, "_chunk": 0},
        {"start": 60.2, "end": 61.0, "pitch": 60, "confidence": 0.85,
         "amplitude": 0.7, "_chunk": 1},
    ]
    stitched, stats = bp.stitch_global_events(events)
    c4 = [n for n in stitched if n["pitch"] == 60]
    assert len(c4) == 1
    assert c4[0]["start"] == 58.2
    assert c4[0]["end"] == 61.0
    assert stats["cross_chunk_notes_merged"] == 1

    # Overlap dedupe
    events2 = [
        {"start": 58.0, "end": 61.0, "pitch": 62, "confidence": 0.55,
         "amplitude": 0.5, "_chunk": 0},
        {"start": 58.5, "end": 61.5, "pitch": 62, "confidence": 0.82,
         "amplitude": 0.8, "_chunk": 1},
    ]
    stitched2, _ = bp.stitch_global_events(events2)
    d4 = [n for n in stitched2 if n["pitch"] == 62]
    assert len(d4) == 1
    assert d4[0]["confidence"] == 0.82

    # Different pitches not merged
    events3 = [
        {"start": 55.0, "end": 59.5, "pitch": 60, "confidence": 0.9,
         "amplitude": 0.8, "_chunk": 0},
        {"start": 59.6, "end": 63.0, "pitch": 62, "confidence": 0.9,
         "amplitude": 0.8, "_chunk": 1},
    ]
    stitched3, _ = bp.stitch_global_events(events3)
    assert len(stitched3) == 2


# ---------------------------------------------------------------------------
# BUG #6: Config inválida em create_chunks
# ---------------------------------------------------------------------------

def test_bug6_zero_chunk_duration_raises():
    from backend.audio.chunking import create_chunks
    with pytest.raises(ValueError, match="chunk_duration"):
        create_chunks(300.0, chunk_duration=0.0, overlap=3.0)


def test_bug6_negative_chunk_duration_raises():
    from backend.audio.chunking import create_chunks
    with pytest.raises(ValueError, match="chunk_duration"):
        create_chunks(300.0, chunk_duration=-60.0, overlap=3.0)


def test_bug6_nan_chunk_duration_raises():
    from backend.audio.chunking import create_chunks
    with pytest.raises(ValueError, match="chunk_duration"):
        create_chunks(300.0, chunk_duration=float('nan'), overlap=3.0)


def test_bug6_inf_chunk_duration_raises():
    from backend.audio.chunking import create_chunks
    with pytest.raises(ValueError, match="chunk_duration"):
        create_chunks(300.0, chunk_duration=float('inf'), overlap=3.0)


def test_bug6_negative_overlap_raises():
    from backend.audio.chunking import create_chunks
    with pytest.raises(ValueError, match="overlap"):
        create_chunks(300.0, chunk_duration=60.0, overlap=-3.0)


def test_bug6_nan_overlap_raises():
    from backend.audio.chunking import create_chunks
    with pytest.raises(ValueError, match="overlap"):
        create_chunks(300.0, chunk_duration=60.0, overlap=float('nan'))


def test_bug6_overlap_ge_chunk_duration_raises():
    """overlap >= chunk_duration agora é ValueError claro, não fallback silencioso."""
    from backend.audio.chunking import create_chunks
    with pytest.raises(ValueError, match="overlap"):
        create_chunks(300.0, chunk_duration=60.0, overlap=60.0)
    with pytest.raises(ValueError, match="overlap"):
        create_chunks(300.0, chunk_duration=60.0, overlap=90.0)


def test_bug6_nan_duration_returns_empty():
    from backend.audio.chunking import create_chunks
    assert create_chunks(float('nan')) == []


def test_bug6_valid_config_still_works():
    from backend.audio.chunking import create_chunks
    chunks = create_chunks(300.0, chunk_duration=60.0, overlap=3.0)
    assert len(chunks) >= 5
    assert chunks[0].start_seconds == 0.0
    assert chunks[-1].end_seconds == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# BUG #7: Env var inválida não crasha o import
# ---------------------------------------------------------------------------

def test_bug7_invalid_env_var_falls_back():
    """CHUNK_DURATION='abc' usa default 60, não crasha."""
    os.environ['CHUNK_DURATION'] = 'abc'
    try:
        import importlib
        import backend.audio.chunking as ch
        importlib.reload(ch)
        assert ch.CHUNK_DURATION == 60.0, (
            f"CHUNK_DURATION deveria ser 60.0 (default), got {ch.CHUNK_DURATION}")
    finally:
        del os.environ['CHUNK_DURATION']
        importlib.reload(ch)


def test_bug7_negative_env_var_falls_back():
    os.environ['CHUNK_OVERLAP'] = '-5'
    try:
        import importlib
        import backend.audio.chunking as ch
        importlib.reload(ch)
        assert ch.CHUNK_OVERLAP == 3.0, (
            f"CHUNK_OVERLAP deveria ser 3.0 (default), got {ch.CHUNK_OVERLAP}")
    finally:
        del os.environ['CHUNK_OVERLAP']
        importlib.reload(ch)


def test_bug7_valid_env_var_used():
    os.environ['CHUNK_DURATION'] = '45'
    try:
        import importlib
        import backend.audio.chunking as ch
        importlib.reload(ch)
        assert ch.CHUNK_DURATION == 45.0
    finally:
        del os.environ['CHUNK_DURATION']
        importlib.reload(ch)


# ---------------------------------------------------------------------------
# BUG #8: .gitignore contém drums/
# ---------------------------------------------------------------------------

def test_bug8_gitignore_has_drums():
    gitignore = (BASE_DIR / '.gitignore').read_text(encoding='utf-8')
    assert '/drums/' in gitignore or 'drums/' in gitignore, (
        "drums/ (diretório de resultados) deveria estar no .gitignore")


def test_bug8_gitignore_does_not_ignore_backend_drums():
    """RED TEAM: /drums/ (ancorado) não deve ignorar backend/drums/ (código).

    Se o padrão fosse apenas 'drums/' sem barra, ele ignoraria
    backend/drums/ (código fonte do módulo de bateria) também.
    """
    import subprocess
    result = subprocess.run(
        ['git', 'check-ignore', '-v', 'backend/drums/drum_utils.py'],
        capture_output=True, text=True, cwd=str(BASE_DIR))
    # check-ignore retorna 0 se o arquivo É ignorado, 1 se NÃO é
    assert result.returncode != 0, (
        f"backend/drums/ NÃO deveria ser ignorado pelo .gitignore! "
        f"Padrão muito broad está bloqueando código fonte. "
        f"check-ignore output: {result.stdout}")
    # E o diretório de resultados drums/ DEVE ser ignorado
    result2 = subprocess.run(
        ['git', 'check-ignore', '-v', 'drums/test.json'],
        capture_output=True, text=True, cwd=str(BASE_DIR))
    assert result2.returncode == 0, (
        f"drums/ (resultados) deveria ser ignorado pelo .gitignore")


# ---------------------------------------------------------------------------
# BUG #9: Frontend cancelAllPolling existe e é chamado
# ---------------------------------------------------------------------------

def test_bug9_cancel_all_polling_function_exists():
    js = (BASE_DIR / 'frontend' / 'app.js').read_text(encoding='utf-8')
    assert 'function cancelAllPolling()' in js, (
        "cancelAllPolling() deveria existir no frontend")


def test_bug9_cancel_called_in_submit():
    js = (BASE_DIR / 'frontend' / 'app.js').read_text(encoding='utf-8')
    submit_idx = js.find('form.addEventListener("submit"')
    assert submit_idx >= 0
    # Verifica que cancelAllPolling é chamado no submit (dentro dos primeiros 500 chars)
    assert 'cancelAllPolling()' in js[submit_idx:submit_idx + 500], (
        "cancelAllPolling() deveria ser chamado no submit handler")


def test_bug9_cancel_called_in_change():
    js = (BASE_DIR / 'frontend' / 'app.js').read_text(encoding='utf-8')
    change_idx = js.find('fileInput.addEventListener("change"')
    assert change_idx >= 0
    assert 'cancelAllPolling()' in js[change_idx:change_idx + 500], (
        "cancelAllPolling() deveria ser chamado no change handler")


def test_bug9_all_intervals_cancelled():
    js = (BASE_DIR / 'frontend' / 'app.js').read_text(encoding='utf-8')
    # A função deve cancelar todos os 5 intervals
    cancel_fn_start = js.find('function cancelAllPolling()')
    cancel_fn_end = js.find('}', js.find('currentArrangeJobId', cancel_fn_start))
    cancel_fn = js[cancel_fn_start:cancel_fn_end]
    for var in ['pollingInterval', 'transcribePollingInterval',
                'drumsPollingInterval', 'scorePollingInterval',
                'arrangePollingInterval']:
        assert var in cancel_fn, f"cancelAllPolling deveria cancelar {var}"


# ---------------------------------------------------------------------------
# RED TEAM PASS (Fase 150): tentar quebrar as próprias correções
# ---------------------------------------------------------------------------

def test_redteam_chunk_with_epsilon_boundaries():
    """Floating point: 116.999999999 e 117.000000001."""
    from backend.audio.chunking import create_chunks
    for d in [116.999999999, 117.000000001, 116.99999, 117.00001]:
        chunks = create_chunks(d, chunk_duration=60.0, overlap=3.0)
        # Cobertura integral
        assert chunks[0].start_seconds == 0.0
        assert chunks[-1].end_seconds == pytest.approx(d, abs=0.01)
        # Sem chunk de 0s únicos (exceto fast path)
        if len(chunks) > 1:
            for c in chunks:
                assert c.end_seconds > c.start_seconds


def test_redteam_stitching_empty_after_validation():
    """Todos eventos NaN → resultado vazio mas válido."""
    bp = _load_worker()
    events = [
        {"start": float('nan'), "end": float('nan'), "pitch": 60,
         "confidence": 0.9, "amplitude": 0.8, "_chunk": 0},
    ]
    stitched, stats = bp.stitch_global_events(events)
    assert stitched == []
    assert stats["invalid_events_discarded"] == 1
    assert stats["notes_after_stitch"] == 0


def test_redteam_stitching_mixed_valid_invalid():
    """Mistura de válidos e inválidos: apenas válidos sobrevivem."""
    bp = _load_worker()
    events = [
        {"start": 1.0, "end": 2.0, "pitch": 60, "confidence": 0.9,
         "amplitude": 0.8, "_chunk": 0},
        {"start": float('nan'), "end": 5.0, "pitch": 62, "confidence": 0.9,
         "amplitude": 0.8, "_chunk": 0},
        {"start": 3.0, "end": 4.0, "pitch": 64, "confidence": 0.9,
         "amplitude": 0.8, "_chunk": 0},
        {"start": 5.0, "end": float('inf'), "pitch": 65, "confidence": 0.9,
         "amplitude": 0.8, "_chunk": 0},
    ]
    stitched, stats = bp.stitch_global_events(events)
    assert len(stitched) == 2
    assert stats["invalid_events_discarded"] == 2
    pitches = sorted(n["pitch"] for n in stitched)
    assert pitches == [60, 64]
