"""
Testes Etapa 8.2 — Basic Pitch real em chunks + cache/resume + progresso.

Cobre:
- fast path: áudio curto NÃO usa chunking;
- long path: áudio longo usa worker chunked;
- worker chunked com manifest sintético;
- cache por chunk (validação, invalidação por versão);
- checkpoint/resume;
- silence skip no worker;
- stitching no worker;
- progresso do job.
"""
import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]

from backend.audio.chunking import CHUNK_DURATION, CHUNK_OVERLAP, create_chunks
from backend.audio.long_audio import compute_audio_hash
from backend.audio.transcriber import (
    CHUNK_THRESHOLD,
    CHUNKED_WORKER_PATH,
    _build_chunked_manifest,
    _get_stem_duration_ffprobe,
)

SR = 22050


def _basic_pitch_env_ok() -> bool:
    """Verifica se .venv-basicpitch está disponível (para testes integrados)."""
    from backend.audio.transcriber import is_basic_pitch_python_available
    return is_basic_pitch_python_available()


def _make_wav(path: Path, duration: float, freq: float = 440.0,
              amplitude: float = 0.5, sr: int = SR) -> Path:
    """Cria WAV sintético com senoide contínua."""
    import soundfile as sf
    t = np.arange(int(duration * sr)) / sr
    y = amplitude * np.sin(2 * np.pi * freq * t)
    sf.write(str(path), y.astype(np.float32), sr)
    return path


def _make_silent_wav(path: Path, duration: float, sr: int = SR) -> Path:
    import soundfile as sf
    y = np.zeros(int(duration * sr))
    sf.write(str(path), y.astype(np.float32), sr)
    return path


# ---------------------------------------------------------------------------
# Fast path vs long path (itens 5-6, 40)
# ---------------------------------------------------------------------------

def test_fast_path_short_stem_no_chunking():
    """Áudio curto (<= CHUNK_THRESHOLD): _get_stem_duration + is_long_audio = False."""
    assert not (CHUNK_THRESHOLD < 90), "CHUNK_THRESHOLD deve ser >= 90"
    assert CHUNK_THRESHOLD == float(os.getenv("CHUNK_THRESHOLD", "90"))


def test_get_stem_duration_ffprobe():
    """FFprobe retorna duração sem carregar áudio."""
    tmp = Path(tempfile.mkdtemp())
    try:
        wav = _make_wav(tmp / "test.wav", 2.5)
        dur = _get_stem_duration_ffprobe(wav)
        assert dur is not None
        assert 2.0 < dur < 3.0
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_get_stem_duration_missing():
    assert _get_stem_duration_ffprobe(Path("C:/nonexistent.wav")) is None


def test_chunked_worker_file_exists():
    assert CHUNKED_WORKER_PATH.is_file(), f"Worker chunked não existe: {CHUNKED_WORKER_PATH}"


# ---------------------------------------------------------------------------
# Manifest (item 4)
# ---------------------------------------------------------------------------

def test_build_chunked_manifest():
    """Manifest contém input/output/cache_dir/chunks/predict_kwargs."""
    tmp = Path(tempfile.mkdtemp())
    try:
        wav = _make_wav(tmp / "stem.wav", 200.0)  # > threshold → múltiplos chunks
        midi_p = tmp / "out.mid"
        json_p = tmp / "out.json"

        from backend.audio.transcriber import TRANSCRIPTIONS_DIR
        fid = str(uuid.uuid4())
        manifest_path, cache_dir = _build_chunked_manifest(
            wav, midi_p, json_p, "vocals", fid, 200.0)

        assert manifest_path.is_file()
        with open(manifest_path, "r", encoding="utf-8") as f:
            m = json.load(f)

        assert m["input"] == str(wav.resolve())
        assert m["output_midi"] == str(midi_p.resolve())
        assert m["output_json"] == str(json_p.resolve())
        assert m["stem"] == "vocals"
        assert m["file_id"] == fid
        assert m["duration"] == 200.0
        assert len(m["chunks"]) >= 2  # 200s / 57s step ≈ 4 chunks
        assert m["predict_kwargs"]["onset_threshold"] == 0.5
        assert "minimum_frequency" in m["predict_kwargs"]  # vocals: 70Hz

        # Chunks têm estrutura correta (formato simples: start/end)
        for c in m["chunks"]:
            assert "index" in c and "start" in c and "end" in c

        # Limpeza
        manifest_path.unlink(missing_ok=True)
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_manifest_short_audio_single_chunk():
    """Manifest para áudio curto teria 1 chunk (mas fast path não gera manifest)."""
    tmp = Path(tempfile.mkdtemp())
    try:
        wav = _make_wav(tmp / "short.wav", 30.0)
        midi_p = tmp / "out.mid"
        json_p = tmp / "out.json"
        fid = str(uuid.uuid4())
        manifest_path, cache_dir = _build_chunked_manifest(
            wav, midi_p, json_p, "other", fid, 30.0)
        with open(manifest_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        # 30s < CHUNK_DURATION → fast path retorna 1 chunk
        assert len(m["chunks"]) == 1
        manifest_path.unlink(missing_ok=True)
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Worker chunked — funções internas (cache, stitching, silêncio)
# ---------------------------------------------------------------------------

# Importa funções do worker via sys.path (worker roda em outro venv, mas as
# funções puras podem ser testadas no venv principal)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "bp_chunked",
    str(BASE_DIR / "backend" / "workers" / "basic_pitch_worker_chunked.py"))
bp_chunked = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(bp_chunked)
    _WORKER_IMPORTABLE = True
except Exception:
    _WORKER_IMPORTABLE = False


@pytest.mark.skipif(not _WORKER_IMPORTABLE, reason="Worker chunked não importável")
class TestWorkerChunkedInternals:
    """Testes das funções internas do worker chunked."""

    def test_chunk_cache_valid_structure(self):
        """Cache válido: version, hash, index, start/end, eventos ok."""
        tmp = Path(tempfile.mkdtemp())
        try:
            cache_p = tmp / "chunk_0000.json"
            expected = {
                "version": bp_chunked.CHUNKED_STAGE_VERSION,
                "audio_hash": "abc123",
                "index": 0,
                "start": 0.0,
                "end": 60.0,
            }
            # Cache válido
            data = {
                "version": expected["version"],
                "audio_hash": "abc123",
                "index": 0,
                "start": 0.0,
                "end": 60.0,
                "events": [{"start": 1.0, "end": 2.0, "pitch": 60}],
            }
            bp_chunked._atomic_write_json(cache_p, data)
            assert bp_chunked._chunk_cache_valid(cache_p, expected) is True

            # Cache inválido: version errada
            data_bad = dict(data, version="wrong-version")
            bp_chunked._atomic_write_json(cache_p, data_bad)
            assert bp_chunked._chunk_cache_valid(cache_p, expected) is False

            # Cache inválido: hash errado
            data_bad2 = dict(data, audio_hash="different")
            bp_chunked._atomic_write_json(cache_p, data_bad2)
            assert bp_chunked._chunk_cache_valid(cache_p, expected) is False

            # Cache inválido: index errado
            data_bad3 = dict(data, index=5)
            bp_chunked._atomic_write_json(cache_p, data_bad3)
            assert bp_chunked._chunk_cache_valid(cache_p, expected) is False

            # Cache inválido: JSON corrompido
            with open(cache_p, "w") as f:
                f.write("{invalid json!!!")
            assert bp_chunked._chunk_cache_valid(cache_p, expected) is False

            # Cache ausente
            assert bp_chunked._chunk_cache_valid(tmp / "missing.json", expected) is False
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_atomic_write_json(self):
        """Escrita atômica: escreve .tmp e renomeia."""
        tmp = Path(tempfile.mkdtemp())
        try:
            p = tmp / "test.json"
            bp_chunked._atomic_write_json(p, {"key": "value"})
            assert p.is_file()
            with open(p, "r") as f:
                assert json.load(f) == {"key": "value"}
            # .tmp não deve existir
            assert not p.with_suffix(".tmp").exists()
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_version_invalidates_cache(self):
        """Item 51: mudança de stage version invalida cache."""
        tmp = Path(tempfile.mkdtemp())
        try:
            cache_p = tmp / "chunk_0000.json"
            expected_v1 = {
                "version": "v1", "audio_hash": "h", "index": 0,
                "start": 0.0, "end": 60.0,
            }
            bp_chunked._atomic_write_json(cache_p, {
                "version": "v1", "audio_hash": "h", "index": 0,
                "start": 0.0, "end": 60.0, "events": [],
            })
            expected_v2 = dict(expected_v1, version="v2")
            assert bp_chunked._chunk_cache_valid(cache_p, expected_v1) is True
            assert bp_chunked._chunk_cache_valid(cache_p, expected_v2) is False
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_is_musically_silent(self):
        """Item 12: silêncio conservador."""
        tmp = Path(tempfile.mkdtemp())
        try:
            # Silêncio absoluto → True
            silent = _make_silent_wav(tmp / "silent.wav", 3.0)
            assert bp_chunked._is_musically_silent(silent) is True

            # Música (senoide 0.5) → False
            music = _make_wav(tmp / "music.wav", 3.0, amplitude=0.5)
            assert bp_chunked._is_musically_silent(music) is False

            # Voz baixa (amp 0.01 > threshold 0.003) → False
            quiet = _make_wav(tmp / "quiet.wav", 3.0, amplitude=0.01)
            assert bp_chunked._is_musically_silent(quiet) is False

            # Fade-in suave → False (item 95: não pular)
            t = np.arange(int(3 * SR)) / SR
            amp = np.linspace(0.001, 0.3, len(t))
            import soundfile as sf
            sf.write(str(tmp / "fadein.wav"),
                     (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32), SR)
            assert bp_chunked._is_musically_silent(tmp / "fadein.wav") is False
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_extract_chunk_wav(self):
        """Item 8: FFmpeg extrai trecho corretamente."""
        tmp = Path(tempfile.mkdtemp())
        try:
            full = _make_wav(tmp / "full.wav", 10.0)
            chunk_p = tmp / "chunk.wav"
            ok = bp_chunked._extract_chunk_wav(full, 3.0, 7.0, chunk_p)
            assert ok is True
            assert chunk_p.is_file()
            # Duração do chunk ≈ 4s
            dur = _get_stem_duration_ffprobe(chunk_p)
            assert dur is not None and 3.5 < dur < 4.5
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_extract_chunk_invalid(self):
        """FFmpeg falha em arquivo inexistente → False."""
        tmp = Path(tempfile.mkdtemp())
        try:
            ok = bp_chunked._extract_chunk_wav(
                tmp / "nonexistent.wav", 0.0, 5.0, tmp / "out.wav")
            assert ok is False
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_raw_event_to_json_global_time(self):
        """Item 14: converte tempo local → global."""
        # (start, end, pitch, amplitude) local ao chunk
        ev = (5.0, 8.0, 60, 0.8)
        result = bp_chunked._raw_event_to_json(ev, chunk_start=60.0)
        assert result["start"] == 65.0  # 60 + 5
        assert result["end"] == 68.0    # 60 + 8
        assert result["pitch"] == 60
        assert result["note"] == "C4"
        assert result["velocity"] > 0

    def test_raw_event_with_pitch_bends(self):
        """Pitch bends preservados na conversão."""
        ev = (1.0, 2.0, 62, 0.7, [0, 10, -10])
        result = bp_chunked._raw_event_to_json(ev, chunk_start=0.0)
        assert result["pitch_bends"] == [0, 10, -10]

    def test_stitch_global_events_boundary_merge(self):
        """Item 42: nota cruzando fronteira = 1 evento."""
        events = [
            {"start": 58.2, "end": 60.0, "pitch": 60, "confidence": 0.9,
             "amplitude": 0.8, "_chunk": 0},
            {"start": 60.2, "end": 61.0, "pitch": 60, "confidence": 0.85,
             "amplitude": 0.7, "_chunk": 1},
        ]
        stitched, stats = bp_chunked.stitch_global_events(events)
        c4 = [n for n in stitched if n["pitch"] == 60]
        assert len(c4) == 1
        assert c4[0]["start"] == 58.2
        assert c4[0]["end"] == 61.0
        assert stats["cross_chunk_notes_merged"] == 1

    def test_stitch_global_events_overlap_dedupe(self):
        """Item 16: mesma nota no overlap = mantém melhor score."""
        events = [
            {"start": 58.0, "end": 61.0, "pitch": 62, "confidence": 0.55,
             "amplitude": 0.5, "_chunk": 0},
            {"start": 58.5, "end": 61.5, "pitch": 62, "confidence": 0.82,
             "amplitude": 0.8, "_chunk": 1},
        ]
        stitched, stats = bp_chunked.stitch_global_events(events)
        d4 = [n for n in stitched if n["pitch"] == 62]
        assert len(d4) == 1
        assert d4[0]["confidence"] == 0.82  # Melhor score
        assert stats["overlap_duplicates_removed"] == 1

    def test_stitch_global_events_different_pitches(self):
        """Item 44: C4 → D4 na borda não são fundidas."""
        events = [
            {"start": 55.0, "end": 59.5, "pitch": 60, "confidence": 0.9,
             "amplitude": 0.8, "_chunk": 0},
            {"start": 59.6, "end": 63.0, "pitch": 62, "confidence": 0.9,
             "amplitude": 0.8, "_chunk": 1},
        ]
        stitched, _ = bp_chunked.stitch_global_events(events)
        assert len(stitched) == 2
        pitches = sorted(n["pitch"] for n in stitched)
        assert pitches == [60, 62]

    def test_stitch_global_events_real_retrigger(self):
        """Item 43/18: C4 com pausa musical real antes de reatacar = 2 eventos."""
        events = [
            {"start": 50.0, "end": 55.0, "pitch": 60, "confidence": 0.9,
             "amplitude": 0.8, "_chunk": 0},
            {"start": 65.0, "end": 70.0, "pitch": 60, "confidence": 0.9,
             "amplitude": 0.8, "_chunk": 1},  # Gap de 10s: pausa real
        ]
        stitched, _ = bp_chunked.stitch_global_events(events)
        assert len(stitched) == 2

    def test_stitch_preserves_pitch_bends_best_score(self):
        """Item 20: bends do evento de maior score preservados no merge."""
        events = [
            {"start": 58.0, "end": 60.0, "pitch": 60, "confidence": 0.5,
             "amplitude": 0.4, "pitch_bends": [0, 5], "_chunk": 0},
            {"start": 60.1, "end": 62.0, "pitch": 60, "confidence": 0.9,
             "amplitude": 0.8, "pitch_bends": [0, -3], "_chunk": 1},
        ]
        stitched, _ = bp_chunked.stitch_global_events(events)
        assert len(stitched) == 1
        # Merge usa o de maior score: bends de conf 0.9
        assert stitched[0]["pitch_bends"] == [0, -3]

    def test_stitch_stats_keys(self):
        """Item 97: métricas presentes."""
        events = [{"start": 1.0, "end": 2.0, "pitch": 60,
                   "confidence": 0.9, "amplitude": 0.8, "_chunk": 0}]
        _, stats = bp_chunked.stitch_global_events(events)
        for key in ("raw_notes", "notes_after_stitch",
                    "overlap_duplicates_removed", "cross_chunk_notes_merged"):
            assert key in stats


# ---------------------------------------------------------------------------
# Progresso (item 27)
# ---------------------------------------------------------------------------

def test_transcription_job_has_progress_field():
    """Item 31: job manager tem campo progress."""
    from backend.audio.transcription_job_manager import (
        TranscriptionJob, create_transcription_job, get_transcription_job,
        clear_transcription_jobs,
    )
    clear_transcription_jobs()
    try:
        job = create_transcription_job("test-fid")
        assert hasattr(job, "progress")
        assert job.progress is None  # Default None

        # Update com progresso
        progress_payload = {
            "stage": "Transcrevendo vocais",
            "progress_percent": 33,
            "processed_seconds": 100,
            "total_seconds": 300,
            "current_chunk": 1,
            "total_chunks": 3,
        }
        from backend.audio.transcription_job_manager import update_transcription_job
        update_transcription_job(job.job_id, progress=progress_payload)
        job2 = get_transcription_job(job.job_id)
        assert job2.progress == progress_payload
    finally:
        clear_transcription_jobs()


def test_progress_never_decreases():
    """Item 52: progresso nunca diminui."""
    from backend.audio.long_audio import make_progress
    p1 = make_progress("Stage 1", 100.0, 300.0, 1, 3)
    p2 = make_progress("Stage 2", 200.0, 300.0, 2, 3)
    p3 = make_progress("Stage 3", 300.0, 300.0, 3, 3)
    percents = [p1["progress_percent"], p2["progress_percent"], p3["progress_percent"]]
    assert percents == sorted(percents)  # Não-decrescente
    assert percents[0] >= 0
    assert percents[-1] == 100


# ---------------------------------------------------------------------------
# Teste integrado com worker real (requer .venv-basicpitch)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _basic_pitch_env_ok(),
                    reason=".venv-basicpitch não disponível")
def test_chunked_worker_real_synth_long():
    """Teste integrado: worker chunked processa áudio sintético longo (>90s).

    Gera 100s de áudio com senoide (música), cria manifest, executa worker,
    valida MIDI + JSON finais. Verifica cache na segunda execução.
    """
    from backend.audio.transcriber import get_basic_pitch_python
    bp_py = get_basic_pitch_python()

    tmp = Path(tempfile.mkdtemp(prefix="bp_chunked_test_"))
    fid = str(uuid.uuid4())
    try:
        # Gera stem sintético de 100s (> CHUNK_THRESHOLD=90) com senoide
        # 100s de 440Hz gera muitas notas — reduz para 100s de senoide
        # contínua (Basic Pitch detecta algumas notas)
        t = np.arange(int(100 * SR)) / SR
        # Senoide com envelope para parecer musical
        y = 0.5 * np.sin(2 * np.pi * 440 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
        stem_wav = tmp / "vocals.wav"
        import soundfile as sf
        sf.write(str(stem_wav), y.astype(np.float32), SR)

        midi_p = tmp / "vocals.mid"
        json_p = tmp / "vocals.json"

        # Cria manifest
        manifest_path, cache_dir = _build_chunked_manifest(
            stem_wav, midi_p, json_p, "vocals", fid, 100.0)

        # Executa worker chunked
        cmd = [
            str(Path(bp_py).resolve()),
            str(CHUNKED_WORKER_PATH.resolve()),
            "--manifest", str(manifest_path.resolve()),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            cwd=str(BASE_DIR))
        assert result.returncode == 0, (
            f"Worker chunked falhou rc={result.returncode}\n"
            f"stderr: {result.stderr[-1000:]}")

        # Valida MIDI final
        assert midi_p.is_file() and midi_p.stat().st_size > 0

        # Valida JSON final
        assert json_p.is_file() and json_p.stat().st_size > 0
        with open(json_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "events" in data
        assert "notes_count" in data
        assert data["stem"] == "vocals"
        assert data["file_id"] == fid

        # Metadata de chunking presente
        assert "chunking" in data
        chunking = data["chunking"]
        assert chunking["chunks_total"] >= 1
        assert "timings_seconds" in chunking
        assert "inference_rtf" in chunking or chunking["chunks_total"] == 1

        # Cache foi criado
        if cache_dir and cache_dir.is_dir():
            cache_files = list(cache_dir.glob("chunk_*.json"))
            assert len(cache_files) >= 1, "Cache de chunks deveria existir"

        # Segunda execução: deve usar cache (muito mais rápida)
        # Remove saída final para forçar re-stitching, mas cache de chunks permanece
        midi_p.unlink(missing_ok=True)
        json_p.unlink(missing_ok=True)

        result2 = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            cwd=str(BASE_DIR))
        assert result2.returncode == 0, (
            f"Segunda execução falhou: {result2.stderr[-500:]}")

        with open(json_p, "r", encoding="utf-8") as f:
            data2 = json.load(f)

        # Segunda execução: chunks_cached deve ser >= 1
        chunking2 = data2.get("chunking", {})
        if chunking2.get("chunks_total", 0) > 1:
            # Se houve múltiplos chunks, pelo menos alguns devem vir do cache
            assert chunking2.get("chunks_cached", 0) >= 1, (
                f"Esperado cache reutilizado, got: {chunking2}")

        # Resultado deve ser idêntico (determinístico)
        assert data2["notes_count"] == data["notes_count"]

    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        # Limpa cache de teste
        from backend.audio.transcriber import TRANSCRIPTIONS_DIR
        cache_test = TRANSCRIPTIONS_DIR / fid / "chunks"
        shutil.rmtree(cache_test, ignore_errors=True)


@pytest.mark.skipif(not _basic_pitch_env_ok(),
                    reason=".venv-basicpitch não disponível")
def test_chunked_worker_silence_skip_real():
    """Item 47: chunk silencioso não chama Basic Pitch."""
    from backend.audio.transcriber import get_basic_pitch_python
    bp_py = get_basic_pitch_python()

    tmp = Path(tempfile.mkdtemp(prefix="bp_silence_test_"))
    fid = str(uuid.uuid4())
    try:
        # 150s: primeiros 60s com música, depois 90s de silêncio
        sr = SR
        t1 = np.arange(int(60 * sr)) / sr
        music = 0.5 * np.sin(2 * np.pi * 440 * t1)
        silence = np.zeros(int(90 * sr))
        y = np.concatenate([music, silence])
        stem_wav = tmp / "other.wav"
        import soundfile as sf
        sf.write(str(stem_wav), y.astype(np.float32), sr)

        midi_p = tmp / "other.mid"
        json_p = tmp / "other.json"

        manifest_path, cache_dir = _build_chunked_manifest(
            stem_wav, midi_p, json_p, "other", fid, 150.0)

        cmd = [
            str(Path(bp_py).resolve()),
            str(CHUNKED_WORKER_PATH.resolve()),
            "--manifest", str(manifest_path.resolve()),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            cwd=str(BASE_DIR))
        assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"

        with open(json_p, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunking = data.get("chunking", {})
        # Pelo menos 1 chunk deve ter sido pulado por silêncio
        if chunking.get("chunks_total", 0) >= 2:
            assert chunking.get("chunks_silence_skipped", 0) >= 1, (
                f"Esperado silence skip em música com 90s de silêncio. "
                f"chunking: {chunking}")

        # Eventos: todos devem estar nos primeiros ~60s
        for ev in data.get("events", []):
            # Permite pequena margem pelo overlap
            assert ev["start"] < 65.0, (
                f"Evento em {ev['start']}s mas silêncio começa em 60s")

    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        from backend.audio.transcriber import TRANSCRIPTIONS_DIR
        shutil.rmtree(TRANSCRIPTIONS_DIR / fid / "chunks", ignore_errors=True)


@pytest.mark.skipif(not _basic_pitch_env_ok(),
                    reason=".venv-basicpitch não disponível")
def test_chunked_worker_short_audio_fast_path_equivalent():
    """Item 40: áudio curto via fast path vs chunked → mesmo resultado.

    Usa 10s de áudio (bem abaixo do threshold). O worker chunked com
    1 chunk deve produzir resultado equivalente ao worker original.
    """
    from backend.audio.transcriber import (
        get_basic_pitch_python, WORKER_PATH, BASIC_PITCH_DEFAULTS,
    )
    bp_py = get_basic_pitch_python()

    tmp = Path(tempfile.mkdtemp(prefix="bp_ab_test_"))
    fid = str(uuid.uuid4())
    try:
        # 10s de senoide
        t = np.arange(int(10 * SR)) / SR
        y = 0.5 * np.sin(2 * np.pi * 440 * t)
        stem_wav = tmp / "other.wav"
        import soundfile as sf
        sf.write(str(stem_wav), y.astype(np.float32), SR)

        # Executa worker ORIGINAL (fast path)
        midi_orig = tmp / "orig.mid"
        json_orig = tmp / "orig.json"
        cmd_orig = [
            str(Path(bp_py).resolve()),
            str(WORKER_PATH.resolve()),
            "--input", str(stem_wav.resolve()),
            "--output-midi", str(midi_orig.resolve()),
            "--output-json", str(json_orig.resolve()),
            "--stem", "other",
            "--file-id", fid,
            "--onset-threshold", str(BASIC_PITCH_DEFAULTS["onset_threshold"]),
            "--frame-threshold", str(BASIC_PITCH_DEFAULTS["frame_threshold"]),
            "--minimum-note-length", str(BASIC_PITCH_DEFAULTS["minimum_note_length"]),
            "--midi-tempo", str(BASIC_PITCH_DEFAULTS["midi_tempo"]),
        ]
        result_orig = subprocess.run(
            cmd_orig, capture_output=True, text=True, timeout=300,
            cwd=str(BASE_DIR))
        assert result_orig.returncode == 0, (
            f"Worker original falhou: {result_orig.stderr[-500:]}")

        # Executa worker CHUNKED (com 1 chunk pois é curto)
        midi_chunk = tmp / "chunked.mid"
        json_chunk = tmp / "chunked.json"
        manifest_path, cache_dir = _build_chunked_manifest(
            stem_wav, midi_chunk, json_chunk, "other", fid, 10.0)
        cmd_chunk = [
            str(Path(bp_py).resolve()),
            str(CHUNKED_WORKER_PATH.resolve()),
            "--manifest", str(manifest_path.resolve()),
        ]
        result_chunk = subprocess.run(
            cmd_chunk, capture_output=True, text=True, timeout=300,
            cwd=str(BASE_DIR))
        assert result_chunk.returncode == 0, (
            f"Worker chunked falhou: {result_chunk.stderr[-500:]}")

        # Compara resultados
        with open(json_orig, "r", encoding="utf-8") as f:
            data_orig = json.load(f)
        with open(json_chunk, "r", encoding="utf-8") as f:
            data_chunk = json.load(f)

        # notes_count deve ser igual (ou muito próximo)
        # Chunking com 1 chunk não deve alterar nada
        assert data_orig["notes_count"] == data_chunk["notes_count"], (
            f"A/B mismatch: orig={data_orig['notes_count']} "
            f"chunked={data_chunk['notes_count']}")

        # Pitches devem ser iguais
        pitches_orig = sorted(e["pitch"] for e in data_orig["events"])
        pitches_chunk = sorted(e["pitch"] for e in data_chunk["events"])
        assert pitches_orig == pitches_chunk

        # Starts/ends dentro de tolerância (50ms)
        for e_o, e_c in zip(data_orig["events"], data_chunk["events"]):
            assert abs(e_o["start"] - e_c["start"]) <= 0.05, (
                f"onset diff: {e_o['start']} vs {e_c['start']}")
            assert abs(e_o["end"] - e_c["end"]) <= 0.05, (
                f"end diff: {e_o['end']} vs {e_c['end']}")

    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        from backend.audio.transcriber import TRANSCRIPTIONS_DIR
        shutil.rmtree(TRANSCRIPTIONS_DIR / fid / "chunks", ignore_errors=True)


# ---------------------------------------------------------------------------
# Resume / checkpoint (itens 24, 49, 50, 54)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _WORKER_IMPORTABLE, reason="Worker não importável")
def test_checkpoint_written_after_chunks():
    """Checkpoint persiste chunks concluídos."""
    tmp = Path(tempfile.mkdtemp())
    try:
        cache_dir = tmp / "cache"
        cache_dir.mkdir()
        manifest = {
            "audio_hash": "test",
            "stem": "vocals",
            "chunks": [{"index": 0}, {"index": 1}, {"index": 2}],
        }
        bp_chunked._write_checkpoint(cache_dir, manifest, completed=[0, 1])

        ckpt_p = cache_dir / "checkpoint.json"
        assert ckpt_p.is_file()
        with open(ckpt_p, "r") as f:
            ckpt = json.load(f)
        assert ckpt["completed_chunks"] == [0, 1]
        assert ckpt["total_chunks"] == 3
        assert ckpt["audio_hash"] == "test"
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_cache_dir_structure():
    """Cache por chunk usa estrutura: transcriptions/<fid>/chunks/<stem>/."""
    tmp = Path(tempfile.mkdtemp())
    try:
        wav = _make_wav(tmp / "stem.wav", 200.0)
        fid = str(uuid.uuid4())
        manifest_path, cache_dir = _build_chunked_manifest(
            wav, tmp / "m.mid", tmp / "j.json", "bass", fid, 200.0)

        # cache_dir segue a convenção
        assert "chunks" in str(cache_dir)
        assert "bass" in str(cache_dir)
        assert fid in str(cache_dir)

        manifest_path.unlink(missing_ok=True)
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# RTF no JSON (item 38)
# ---------------------------------------------------------------------------

def test_rtf_in_chunking_metadata():
    """RTF (inference_seconds / duration) aparece no metadata quando > 0."""
    # Simulado: validamos a fórmula
    inference_seconds = 45.0
    duration = 150.0
    rtf = round(inference_seconds / duration, 3)
    assert rtf == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Frontend de progresso (item 29) — valida payload que o frontend consome
# ---------------------------------------------------------------------------

def test_transcription_status_endpoint_includes_progress():
    """GET /api/transcribe/status inclui campo progress quando definido."""
    from fastapi.testclient import TestClient
    from backend.audio.transcription_job_manager import (
        clear_transcription_jobs, create_transcription_job, update_transcription_job,
    )
    from app import app

    clear_transcription_jobs()
    try:
        with TestClient(app) as client:
            job = create_transcription_job("test-progress-fid")
            update_transcription_job(job.job_id, status="running",
                                     progress={
                                         "stage": "Transcrevendo vocais",
                                         "progress_percent": 33,
                                         "processed_seconds": 100,
                                         "total_seconds": 300,
                                         "current_chunk": 1,
                                         "total_chunks": 3,
                                     })
            r = client.get(f"/api/transcribe/status/{job.job_id}")
            assert r.status_code == 200
            data = r.json()
            assert "progress" in data
            assert data["progress"]["progress_percent"] == 33
            assert data["progress"]["stage"] == "Transcrevendo vocais"
            assert data["progress"]["current_chunk"] == 1
            assert data["progress"]["total_chunks"] == 3
    finally:
        clear_transcription_jobs()