"""
Testes Etapa 8.3 — Cache persistente da análise musical.

Cobre:
- First run: MISS (computa)
- Second run: HIT (retorna cache)
- Same result: cached == computed
- Corrupt JSON: recompute
- Hash mismatch: recompute
- Version mismatch: recompute
- No NaN in cache
- Atomic write
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

from backend.audio.analysis_cache import (
    ANALYSIS_DIR,
    ANALYSIS_STAGE_VERSION,
    _analysis_cache_path,
    _atomic_write_json,
    _sanitize_float,
    _validate_cache,
    clear_analysis_cache,
    get_cached_analysis,
    result_to_cache_dict,
    save_analysis_cache,
)
from backend.audio.music_analysis import MusicAnalysisResult, analyze_music


def _make_test_wav(tmp_path: Path, duration: float = 3.0, bpm: float = 120.0) -> Path:
    """Cria WAV sintético de teste."""
    import soundfile as sf
    sr = 22050
    t = np.arange(int(duration * sr)) / sr
    period = 60.0 / bpm
    y = np.zeros_like(t)
    for beat_t in np.arange(0, duration, period):
        i = int(beat_t * sr)
        if i + 200 < len(y):
            y[i:i+200] += np.exp(-np.arange(200) / 30.0) * 0.5
    # Adiciona tono para key detection
    y += 0.1 * np.sin(2 * np.pi * 261.63 * t)  # C4
    sf.write(str(tmp_path / "test.wav"), y.astype(np.float32), sr)
    return tmp_path / "test.wav"


class TestAnalysisCacheBasic:
    """Testes básicos do mecanismo de cache."""

    def setup_method(self):
        self.file_id = str(uuid.uuid4())
        self.cache_path = _analysis_cache_path(self.file_id)

    def teardown_method(self):
        clear_analysis_cache(self.file_id)

    def test_cache_miss_on_first(self):
        """Primeira consulta: cache vazio = None (MISS)."""
        assert get_cached_analysis(self.file_id, "fake_hash_123") is None

    def test_save_and_retrieve(self):
        """Salva e recupera com mesmo hash."""
        data = {"bpm": 120, "key": "C", "mode": "major", "duration": 30.0}
        assert save_analysis_cache(self.file_id, "hash_abc", data) is True
        cached = get_cached_analysis(self.file_id, "hash_abc")
        assert cached is not None
        assert cached["bpm"] == 120
        assert cached["key"] == "C"
        assert cached.get("analysis_cache_hit") is True

    def test_hash_mismatch_returns_none(self):
        """Hash diferente: cache inválido = None."""
        data = {"bpm": 120, "key": "C", "mode": "major", "duration": 30.0}
        save_analysis_cache(self.file_id, "hash_abc", data)
        # Tenta recuperar com hash diferente
        assert get_cached_analysis(self.file_id, "hash_different") is None

    def test_version_mismatch_returns_none(self):
        """Version antiga: cache inválido = None."""
        data = {
            "bpm": 120, "key": "C", "mode": "major", "duration": 30.0,
            "analysis_stage_version": "analysis-v1",  # versão antiga
            "audio_hash": "hash_abc",
        }
        # Salva diretamente (bypass) com versão antiga
        _atomic_write_json(self.cache_path, data)
        assert get_cached_analysis(self.file_id, "hash_abc") is None

    def test_corrupt_json_returns_none(self):
        """JSON corrompido: cache inválido = None."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            f.write("{invalid json!!!")
        assert get_cached_analysis(self.file_id, "hash_abc") is None

    def test_empty_file_returns_none(self):
        """Arquivo vazio: cache inválido = None."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text("")
        assert get_cached_analysis(self.file_id, "hash_abc") is None

    def test_missing_file_returns_none(self):
        """Arquivo inexistente: None."""
        assert get_cached_analysis("nonexistent-fid", "hash") is None

    def test_no_hash_returns_none(self):
        """Sem hash: None."""
        assert get_cached_analysis(self.file_id, "") is None
        assert get_cached_analysis(self.file_id, None) is None


class TestAnalysisCacheValidation:
    """Testa validação de campos."""

    def setup_method(self):
        self.file_id = str(uuid.uuid4())

    def teardown_method(self):
        clear_analysis_cache(self.file_id)

    def test_missing_required_field_invalidates(self):
        """Campo obrigatório ausente: cache inválido."""
        data = {
            "audio_hash": "hash_x",
            "analysis_stage_version": ANALYSIS_STAGE_VERSION,
            "duration": 30.0,
            # MISSING: bpm, key, mode, file_id
        }
        assert not _validate_cache(data, "hash_x", ANALYSIS_STAGE_VERSION)

    def test_nan_in_cache_invalidates(self):
        """NaN em qualquer float: cache inválido."""
        data = {
            "file_id": "test",
            "audio_hash": "hash_x",
            "analysis_stage_version": ANALYSIS_STAGE_VERSION,
            "duration": 30.0,
            "bpm": float('nan'),  # NaN!
            "key": "C",
            "mode": "major",
        }
        assert not _validate_cache(data, "hash_x", ANALYSIS_STAGE_VERSION)

    def test_infinity_in_cache_invalidates(self):
        """Infinity: cache inválido."""
        data = {
            "file_id": "test",
            "audio_hash": "hash_x",
            "analysis_stage_version": ANALYSIS_STAGE_VERSION,
            "duration": float('inf'),
            "bpm": 120,
            "key": "C",
            "mode": "major",
        }
        assert not _validate_cache(data, "hash_x", ANALYSIS_STAGE_VERSION)

    def test_zero_duration_invalidates(self):
        """Duration <= 0: cache inválido."""
        data = {
            "file_id": "test",
            "audio_hash": "hash_x",
            "analysis_stage_version": ANALYSIS_STAGE_VERSION,
            "duration": 0.0,
            "bpm": 120,
            "key": "C",
            "mode": "major",
        }
        assert not _validate_cache(data, "hash_x", ANALYSIS_STAGE_VERSION)

    def test_invalid_bpm_invalidates(self):
        """BPM <= 0 ou > 1000: cache inválido."""
        data = {
            "file_id": "test",
            "audio_hash": "hash_x",
            "analysis_stage_version": ANALYSIS_STAGE_VERSION,
            "duration": 30.0,
            "bpm": -5,
            "key": "C",
            "mode": "major",
        }
        assert not _validate_cache(data, "hash_x", ANALYSIS_STAGE_VERSION)


class TestSanitizeFloat:
    """Testa sanitização de NaN/Infinity."""

    def test_nan_becomes_none(self):
        assert _sanitize_float(float('nan')) is None

    def test_inf_becomes_none(self):
        assert _sanitize_float(float('inf')) is None

    def test_neg_inf_becomes_none(self):
        assert _sanitize_float(float('-inf')) is None

    def test_valid_float_preserved(self):
        assert _sanitize_float(3.14) == 3.14

    def test_none_stays_none(self):
        assert _sanitize_float(None) is None

    def test_int_converted(self):
        assert _sanitize_float(120) == 120.0


class TestAtomicWrite:
    """Testa escrita atômica."""

    def test_no_tmp_file_left_behind(self):
        """Após save, arquivo .tmp não deve existir."""
        p = ANALYSIS_DIR / str(uuid.uuid4()) / "test.json"
        _atomic_write_json(p, {"key": "value"})
        assert p.is_file()
        assert not p.with_suffix(".tmp").exists()
        # Cleanup
        p.unlink(missing_ok=True)
        p.parent.rmdir()

    def test_valid_json_written(self):
        """JSON escrito é parseável."""
        p = ANALYSIS_DIR / str(uuid.uuid4()) / "test.json"
        _atomic_write_json(p, {"bpm": 120, "list": [1, 2, 3]})
        with open(p, "r") as f:
            data = json.load(f)
        assert data["bpm"] == 120
        p.unlink(missing_ok=True)
        p.parent.rmdir()

    def test_nan_rejected_in_json(self):
        """json.dump com allow_nan=False rejeita NaN."""
        p = ANALYSIS_DIR / str(uuid.uuid4()) / "test.json"
        tmp = p.with_suffix(".tmp")
        try:
            with pytest.raises(ValueError):
                _atomic_write_json(p, {"bad": float('nan')})
            # Arquivo final não deve existir
            assert not p.exists()
        finally:
            # Limpa .tmp órfão e diretório
            tmp.unlink(missing_ok=True)
            try:
                p.parent.rmdir()
            except OSError:
                pass


class TestResultToCacheDict:
    """Testa conversão de MusicAnalysisResult para dict de cache."""

    def test_includes_all_precision_fields(self):
        r = MusicAnalysisResult()
        r.bpm = 89.0
        r.bpm_rounded = 89
        r.bpm_confidence = 0.95
        r.bpm_stability = 0.93
        r.key = "A#"
        r.mode = "minor"
        r.key_confidence = 0.15
        r.key_window_agreement = 0.38
        r.key_score_margin = 0.11
        r.first_beat_time = 0.30
        r.duration_analyzed = 203.2
        d = result_to_cache_dict(r)
        for field in ["bpm", "bpm_stability", "key_window_agreement",
                      "key_score_margin", "first_beat_time", "duration"]:
            assert field in d, f"Campo {field} deveria estar no cache dict"

    def test_nan_fields_sanitized_in_save(self):
        """NaN em result é sanitizado no save."""
        r = MusicAnalysisResult()
        r.bpm = 89.0
        r.bpm_stability = float('nan')  # NaN!
        d = result_to_cache_dict(r)
        fid = str(uuid.uuid4())
        try:
            save_analysis_cache(fid, "hash_test", d)
            # Verifica que NaN não está no JSON
            with open(_analysis_cache_path(fid), "r") as f:
                content = f.read()
            assert "NaN" not in content
            assert "Infinity" not in content
        finally:
            clear_analysis_cache(fid)


class TestAnalyzeMusicCacheIntegration:
    """Testa integração do cache com analyze_music()."""

    def test_cache_miss_then_hit(self, tmp_path):
        """Primeira chamada: MISS (computa). Segunda: HIT (cache)."""
        wav = _make_test_wav(tmp_path, duration=3.0, bpm=120.0)
        fid = str(uuid.uuid4())

        try:
            # Primeira: computa (MISS)
            t0 = time.perf_counter()
            r1 = analyze_music(wav, file_id=fid)
            compute_time = time.perf_counter() - t0
            assert r1.bpm is not None or r1.error is not None

            # Verifica que cache foi salvo
            cache_p = _analysis_cache_path(fid)
            if r1.error is None:
                assert cache_p.is_file(), "Cache deveria ter sido salvo"

            # Segunda: cache (HIT) — deve ser muito mais rápido
            t0 = time.perf_counter()
            r2 = analyze_music(wav, file_id=fid)
            cache_time = time.perf_counter() - t0

            # Resultado deve ser equivalente
            if r1.bpm is not None and r2.bpm is not None:
                assert abs(r1.bpm - r2.bpm) < 0.01
            if r1.key is not None:
                assert r1.key == r2.key
                assert r1.mode == r2.mode

            # Cache hit deve ser muito mais rápido (hash + JSON read)
            if cache_p.is_file():
                assert cache_time < compute_time, (
                    f"Cache hit ({cache_time:.3f}s) deveria ser mais rápido "
                    f"que compute ({compute_time:.3f}s)")
        finally:
            clear_analysis_cache(fid)

    def test_cache_preserves_precision_fields(self, tmp_path):
        """Campos de precisão (stability, agreement) são preservados no cache."""
        # Para este teste usamos um áudio sintético curto (< 30s)
        # Window analysis não roda, mas os campos devem existir como None
        wav = _make_test_wav(tmp_path, duration=2.0)
        fid = str(uuid.uuid4())
        try:
            r1 = analyze_music(wav, file_id=fid)
            r2 = analyze_music(wav, file_id=fid)
            # bpm_stability deve ser None em ambos (áudio curto)
            assert r1.bpm_stability is None or isinstance(r1.bpm_stability, float)
            assert r2.bpm_stability is None or isinstance(r2.bpm_stability, float)
            if r1.bpm_stability is not None:
                assert r2.bpm_stability == r1.bpm_stability
        finally:
            clear_analysis_cache(fid)


class TestCacheDirectoryStructure:
    """Testa estrutura de diretórios do cache."""

    def test_cache_dir_exists(self):
        assert ANALYSIS_DIR.is_dir()

    def test_cache_path_format(self):
        """Cache path: analysis/<file_id>/analysis.json"""
        p = _analysis_cache_path("test-fid")
        assert p.parent == ANALYSIS_DIR / "test-fid"
        assert p.name == "analysis.json"

    def test_clear_removes_file(self):
        fid = str(uuid.uuid4())
        save_analysis_cache(fid, "hash", {"bpm": 120})
        assert _analysis_cache_path(fid).is_file()
        clear_analysis_cache(fid)
        assert not _analysis_cache_path(fid).is_file()
