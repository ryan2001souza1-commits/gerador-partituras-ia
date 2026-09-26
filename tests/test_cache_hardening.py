"""
Etapa 8.3.1 — Hardening do cache: testes de invalidação por hash/config/version.

Cada teste verifica que cache obsoleto NÃO é reutilizado:
- audio_hash mudou → todas as stages downstream invalidam
- stem_hash mudou → transcription daquele stem invalida
- config mudou → stage com config diferente invalida
- version mudou → cache de version antiga invalida
- metadata corrompida → MISS seguro (não crasha)
"""
import json
import os
import uuid
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]

from backend.pipeline import cache_identity as ci


def _make_wav(path: Path, duration: float = 1.0, sr: int = 22050, seed: int = 42):
    """Cria WAV sintético."""
    import soundfile as sf
    rng = np.random.default_rng(seed)
    y = 0.5 * rng.standard_normal(int(duration * sr))
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y.astype(np.float32), sr)
    return path


def _make_stems(file_id: str, seed: int = 42):
    """Cria 4 stems sintéticos + metadata."""
    from backend.audio.stem_separator import STEMS_DIR
    d = STEMS_DIR / file_id
    d.mkdir(parents=True, exist_ok=True)
    for stem in ci.EXPECTED_STEMS:
        _make_wav(d / f"{stem}.wav", duration=0.5, seed=seed)
    return d


def _make_transcription_files(file_id: str, stem: str, notes: int = 5):
    """Cria MIDI + JSON de transcrição sintéticos."""
    from backend.audio.transcriber import MIDI_DIR, TRANSCRIPTIONS_DIR
    midi_p = MIDI_DIR / file_id / f"{stem}.mid"
    json_p = TRANSCRIPTIONS_DIR / file_id / f"{stem}.json"
    midi_p.parent.mkdir(parents=True, exist_ok=True)
    json_p.parent.mkdir(parents=True, exist_ok=True)
    # MIDI fake (não precisa ser válido para estes testes de metadata)
    midi_p.write_bytes(b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x01\xe0MTrk\x00\x00\x00\x00")
    events = [{"start": i * 0.5, "end": i * 0.5 + 0.3, "pitch": 60 + i,
               "velocity": 64, "amplitude": 0.5, "confidence": 0.5,
               "strength": 0.5, "duration": 0.3, "note": "C4"}
              for i in range(notes)]
    json_p.write_text(json.dumps({
        "file_id": file_id, "stem": stem, "notes_count": notes,
        "duration": 5.0, "events": events,
    }))
    return midi_p, json_p


def _make_drums_json(file_id: str):
    from backend.drums.drum_transcriber import DRUMS_DIR, get_drums_json_path
    p = get_drums_json_path(file_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "file_id": file_id, "events": [{"time": 0.5, "instrument": "kick",
        "confidence": 0.8, "strength": 0.7}],
        "stats": {"kick_count": 1},
    }))
    return p


def _make_score_files(file_id: str):
    from backend.notation.score_generator import SCORES_DIR, SCORE_MODELS_DIR
    xml_p = SCORES_DIR / file_id / "score.musicxml"
    model_p = SCORE_MODELS_DIR / file_id / "score.json"
    xml_p.parent.mkdir(parents=True, exist_ok=True)
    model_p.parent.mkdir(parents=True, exist_ok=True)
    xml_p.write_text("<?xml version='1.0'?><score-partwise/>")
    model_p.write_text(json.dumps({"file_id": file_id, "tempo": 120,
                                   "parts": {}, "warnings": []}))
    return xml_p, model_p


def _make_arrangement_files(file_id: str):
    from backend.arrangement.arrangement_generator import ARRANGEMENTS_DIR
    xml_p = ARRANGEMENTS_DIR / file_id / "arrangement.musicxml"
    model_p = ARRANGEMENTS_DIR / file_id / "arrangement.json"
    xml_p.parent.mkdir(parents=True, exist_ok=True)
    xml_p.write_text("<?xml version='1.0'?><score-partwise/>")
    model_p.write_text(json.dumps({"file_id": file_id, "instruments": []}))
    return xml_p, model_p


# ---------------------------------------------------------------------------
# Helper: cleanup
# ---------------------------------------------------------------------------

def _cleanup_file_id(file_id: str):
    """Remove todos os arquivos/artefatos de um file_id de teste."""
    from backend.audio.stem_separator import STEMS_DIR
    from backend.audio.transcriber import MIDI_DIR, TRANSCRIPTIONS_DIR
    from backend.drums.drum_transcriber import DRUMS_DIR
    from backend.notation.score_generator import SCORES_DIR, SCORE_MODELS_DIR
    from backend.arrangement.arrangement_generator import ARRANGEMENTS_DIR
    from backend.audio.analysis_cache import ANALYSIS_DIR
    import shutil
    for d in [STEMS_DIR / file_id, MIDI_DIR / file_id,
              TRANSCRIPTIONS_DIR / file_id, DRUMS_DIR / file_id,
              SCORES_DIR / file_id, SCORE_MODELS_DIR / file_id,
              ARRANGEMENTS_DIR / file_id, ANALYSIS_DIR / file_id]:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# ANALYSIS cache
# ---------------------------------------------------------------------------

class TestAnalysisCacheHardening:

    def setup_method(self):
        self.fid = str(uuid.uuid4())

    def teardown_method(self):
        _cleanup_file_id(self.fid)

    def test_audio_hash_change_invalidates_analysis(self):
        """Mudança de audio_hash → analysis cache invalida."""
        # Salva com hash A
        ci.save_analysis_metadata(self.fid, "hash_audio_A")
        assert ci.is_analysis_cache_valid(self.fid, "hash_audio_A") is False  # no output yet

        # Cria output também
        from backend.audio.analysis_cache import _analysis_cache_path
        p = _analysis_cache_path(self.fid)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "file_id": self.fid, "audio_hash": "hash_audio_A",
            "analysis_stage_version": "analysis-v3",
            "bpm": 120, "key": "C", "mode": "major", "duration": 30.0,
        }))
        assert ci.is_analysis_cache_valid(self.fid, "hash_audio_A") is True

        # Hash B: deve invalidar
        assert ci.is_analysis_cache_valid(self.fid, "hash_audio_B") is False

    def test_analysis_version_change_invalidates(self):
        """Version antiga: invalida."""
        from backend.audio.analysis_cache import ANALYSIS_STAGE_VERSION, _analysis_cache_path
        meta_p = ci.get_analysis_metadata_path(self.fid)
        meta_p.parent.mkdir(parents=True, exist_ok=True)
        meta_p.write_text(json.dumps({
            "audio_hash": "hash_x",
            "stage_version": "analysis-v0",  # versão antiga
        }))
        out_p = _analysis_cache_path(self.fid)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps({
            "file_id": self.fid, "audio_hash": "hash_x",
            "analysis_stage_version": ANALYSIS_STAGE_VERSION,
            "bpm": 120, "key": "C", "mode": "major", "duration": 30.0,
        }))
        # Metadata version antiga → inválido mesmo com output correto
        assert ci.is_analysis_cache_valid(self.fid, "hash_x") is False


# ---------------------------------------------------------------------------
# DEMUCS cache
# ---------------------------------------------------------------------------

class TestDemucsCacheHardening:

    def setup_method(self):
        self.fid = str(uuid.uuid4())
        self.audio_hash = "hash_audio_" + self.fid[:8]
        _make_stems(self.fid)

    def teardown_method(self):
        _cleanup_file_id(self.fid)

    def test_demucs_cache_valid_after_save(self):
        """Após save metadata: cache válido se stems unchanged."""
        ci.save_demucs_metadata(self.fid, self.audio_hash)
        assert ci.is_demucs_cache_valid(self.fid, self.audio_hash) is True

    def test_demucs_cache_audio_hash_change(self):
        """Audio hash diferente: cache inválido."""
        ci.save_demucs_metadata(self.fid, self.audio_hash)
        assert ci.is_demucs_cache_valid(self.fid, "different_hash") is False

    def test_demucs_cache_model_change(self):
        """Modelo diferente: cache inválido."""
        ci.save_demucs_metadata(self.fid, self.audio_hash, model="htdemucs")
        assert ci.is_demucs_cache_valid(self.fid, self.audio_hash, model="mdx") is False

    def test_demucs_cache_missing_stem(self):
        """Stem removido: cache inválido."""
        ci.save_demucs_metadata(self.fid, self.audio_hash)
        assert ci.is_demucs_cache_valid(self.fid, self.audio_hash) is True
        # Remove um stem
        from backend.audio.stem_separator import STEMS_DIR
        (STEMS_DIR / self.fid / "vocals.wav").unlink()
        assert ci.is_demucs_cache_valid(self.fid, self.audio_hash) is False

    def test_demucs_cache_changed_stem_hash(self):
        """Stem modificado (hash mudou): cache inválido."""
        ci.save_demucs_metadata(self.fid, self.audio_hash)
        assert ci.is_demucs_cache_valid(self.fid, self.audio_hash) is True
        # Sobrescreve um stem (hash muda)
        from backend.audio.stem_separator import STEMS_DIR
        _make_wav(STEMS_DIR / self.fid / "vocals.wav", duration=0.5, seed=999)
        assert ci.is_demucs_cache_valid(self.fid, self.audio_hash) is False

    def test_demucs_no_metadata_invalid(self):
        """Sem metadata: cache inválido (não apenas file_exists)."""
        assert ci.is_demucs_cache_valid(self.fid, self.audio_hash) is False


# ---------------------------------------------------------------------------
# TRANSCRIPTION cache
# ---------------------------------------------------------------------------

class TestTranscriptionCacheHardening:

    def setup_method(self):
        self.fid = str(uuid.uuid4())
        _make_stems(self.fid)
        self.stem_hashes = ci.get_stem_hashes(self.fid)
        assert self.stem_hashes is not None
        self.config_hash = ci.compute_config_hash({"test": "config"})
        for stem in ci.TRANSCRIBED_STEMS:
            _make_transcription_files(self.fid, stem)
            ci.save_transcription_metadata(self.fid, stem,
                                          self.stem_hashes[stem],
                                          self.config_hash)

    def teardown_method(self):
        _cleanup_file_id(self.fid)

    def test_transcription_cache_valid(self):
        assert ci.are_all_transcriptions_valid(self.fid, self.config_hash) is True

    def test_transcription_cache_stem_hash_change(self):
        """Stem hash mudou: transcrição daquele stem invalida."""
        # Modifica vocals.wav
        from backend.audio.stem_separator import STEMS_DIR
        _make_wav(STEMS_DIR / self.fid / "vocals.wav", duration=0.5, seed=999)
        new_hashes = ci.get_stem_hashes(self.fid)
        # vocals hash mudou, bass/other não
        assert new_hashes["vocals"] != self.stem_hashes["vocals"]
        assert new_hashes["bass"] == self.stem_hashes["bass"]
        # Com hashes novos, transcrição de vocals é inválida
        assert ci.is_transcription_cache_valid(
            self.fid, "vocals", new_hashes["vocals"], self.config_hash) is False
        # Mas bass ainda é válida
        assert ci.is_transcription_cache_valid(
            self.fid, "bass", new_hashes["bass"], self.config_hash) is True

    def test_transcription_cache_config_change(self):
        """Config mudou: todas as transcrições invalidam."""
        new_config = ci.compute_config_hash({"test": "different"})
        assert ci.are_all_transcriptions_valid(self.fid, new_config) is False

    def test_transcription_no_metadata_invalid(self):
        """Sem metadata (só MIDI/JSON): cache inválido."""
        fid2 = str(uuid.uuid4())
        try:
            _make_stems(fid2)
            for stem in ci.TRANSCRIBED_STEMS:
                _make_transcription_files(fid2, stem)
            # Sem metadata → inválido mesmo com arquivos
            hashes = ci.get_stem_hashes(fid2)
            assert ci.are_all_transcriptions_valid(fid2, self.config_hash) is False
        finally:
            _cleanup_file_id(fid2)

    def test_transcription_per_stem_isolation(self):
        """vocals cached, bass cached, other missing → only other invalid."""
        # Remove metadata de other
        ci.get_transcription_metadata_path(self.fid, "other").unlink()
        # vocals e bass ainda válidos individualmente
        assert ci.is_transcription_cache_valid(
            self.fid, "vocals", self.stem_hashes["vocals"], self.config_hash) is True
        assert ci.is_transcription_cache_valid(
            self.fid, "bass", self.stem_hashes["bass"], self.config_hash) is True
        # other sem metadata → inválido
        assert ci.is_transcription_cache_valid(
            self.fid, "other", self.stem_hashes["other"], self.config_hash) is False
        # are_all → false (other inválido)
        assert ci.are_all_transcriptions_valid(self.fid, self.config_hash) is False


# ---------------------------------------------------------------------------
# DRUMS cache
# ---------------------------------------------------------------------------

class TestDrumsCacheHardening:

    def setup_method(self):
        self.fid = str(uuid.uuid4())
        _make_stems(self.fid)
        self.stem_hashes = ci.get_stem_hashes(self.fid)
        self.config_hash = ci.compute_config_hash({"test": "drums"})
        _make_drums_json(self.fid)
        ci.save_drums_metadata(self.fid, self.stem_hashes["drums"], self.config_hash)

    def teardown_method(self):
        _cleanup_file_id(self.fid)

    def test_drums_cache_valid(self):
        assert ci.is_drums_cache_valid(self.fid, self.stem_hashes["drums"], self.config_hash) is True

    def test_drums_cache_stem_hash_change(self):
        """Drums stem mudou: cache inválido."""
        from backend.audio.stem_separator import STEMS_DIR
        _make_wav(STEMS_DIR / self.fid / "drums.wav", duration=0.5, seed=999)
        new_hash = ci.compute_file_hash(STEMS_DIR / self.fid / "drums.wav")
        assert ci.is_drums_cache_valid(self.fid, new_hash, self.config_hash) is False

    def test_drums_cache_config_change(self):
        new_config = ci.compute_config_hash({"test": "different"})
        assert ci.is_drums_cache_valid(self.fid, self.stem_hashes["drums"], new_config) is False

    def test_drums_cache_version_change(self):
        """Version antiga: inválido."""
        # Sobrescreve metadata com versão antiga
        meta_p = ci.get_drums_metadata_path(self.fid)
        meta_p.write_text(json.dumps({
            "drums_stem_hash": self.stem_hashes["drums"],
            "config_hash": self.config_hash,
            "stage_version": "drum-v0",  # antiga
        }))
        assert ci.is_drums_cache_valid(self.fid, self.stem_hashes["drums"], self.config_hash) is False

    def test_drums_no_metadata_invalid(self):
        ci.get_drums_metadata_path(self.fid).unlink()
        assert ci.is_drums_cache_valid(self.fid, self.stem_hashes["drums"], self.config_hash) is False


# ---------------------------------------------------------------------------
# SCORE cache
# ---------------------------------------------------------------------------

class TestScoreCacheHardening:

    def setup_method(self):
        self.fid = str(uuid.uuid4())
        _make_score_files(self.fid)
        self.identity = {
            "config_key": "test_ck_123",
            "vocals_stem_hash": "v_hash",
            "bass_stem_hash": "b_hash",
            "other_stem_hash": "o_hash",
            "audio_hash": "a_hash",
        }
        ci.save_score_metadata(self.fid, self.identity)

    def teardown_method(self):
        _cleanup_file_id(self.fid)

    def test_score_cache_valid(self):
        assert ci.is_score_cache_valid(self.fid, self.identity) is True

    def test_score_invalidates_on_transcription_change(self):
        """Vocals stem hash mudou → score invalida."""
        new_identity = dict(self.identity, vocals_stem_hash="new_v_hash")
        assert ci.is_score_cache_valid(self.fid, new_identity) is False

    def test_score_invalidates_on_drums_change(self):
        """Other stem hash mudou → score invalida."""
        new_identity = dict(self.identity, other_stem_hash="new_o_hash")
        assert ci.is_score_cache_valid(self.fid, new_identity) is False

    def test_score_invalidates_on_config_change(self):
        new_identity = dict(self.identity, config_key="different_ck")
        assert ci.is_score_cache_valid(self.fid, new_identity) is False

    def test_score_no_metadata_invalid(self):
        ci.get_score_metadata_path(self.fid).unlink()
        assert ci.is_score_cache_valid(self.fid, self.identity) is False


# ---------------------------------------------------------------------------
# ARRANGEMENT cache
# ---------------------------------------------------------------------------

class TestArrangementCacheHardening:

    def setup_method(self):
        self.fid = str(uuid.uuid4())
        _make_arrangement_files(self.fid)
        self.identity = {
            "config_key": "test_arr_ck",
            "base_config_key": "base_ck",
            "audio_hash": "a_hash",
        }
        ci.save_arrangement_metadata(self.fid, self.identity)

    def teardown_method(self):
        _cleanup_file_id(self.fid)

    def test_arrangement_cache_valid(self):
        assert ci.is_arrangement_cache_valid(self.fid, self.identity) is True

    def test_arrangement_invalidates_on_score_change(self):
        new_identity = dict(self.identity, base_config_key="new_base_ck")
        assert ci.is_arrangement_cache_valid(self.fid, new_identity) is False

    def test_arrangement_style_only(self):
        """Mudar apenas style: SÓ arrangement config_key muda."""
        style_a = {"config_key": "ck_style_a", "base_config_key": "base_ck",
                   "audio_hash": "a_hash"}
        style_b = {"config_key": "ck_style_b", "base_config_key": "base_ck",
                   "audio_hash": "a_hash"}
        ci.save_arrangement_metadata(self.fid, style_a)
        assert ci.is_arrangement_cache_valid(self.fid, style_a) is True
        # Style B tem config_key diferente → invalida arrangement
        assert ci.is_arrangement_cache_valid(self.fid, style_b) is False
        # Mas identity de score não mudou (base_config_key igual)

    def test_arrangement_no_metadata_invalid(self):
        ci.get_arrangement_metadata_path(self.fid).unlink()
        assert ci.is_arrangement_cache_valid(self.fid, self.identity) is False


# ---------------------------------------------------------------------------
# AUDIO HASH CHANGE — invalida tudo
# ---------------------------------------------------------------------------

class TestAudioHashChangeInvalidatesAll:
    """Mudança do audio_hash invalida TODAS as stages downstream."""

    def setup_method(self):
        self.fid = str(uuid.uuid4())
        self.hash_a = "audio_hash_AAAA"
        self.hash_b = "audio_hash_BBBB"

    def teardown_method(self):
        _cleanup_file_id(self.fid)

    def test_analysis_invalidates(self):
        ci.save_analysis_metadata(self.fid, self.hash_a)
        # Com hash B: analysis inválida
        assert ci.is_analysis_cache_valid(self.fid, self.hash_b) is False

    def test_demucs_invalidates(self):
        _make_stems(self.fid)
        ci.save_demucs_metadata(self.fid, self.hash_a)
        assert ci.is_demucs_cache_valid(self.fid, self.hash_a) is True
        assert ci.is_demucs_cache_valid(self.fid, self.hash_b) is False

    def test_full_chain_invalidates(self):
        """Audio hash B: todas as stages downstream inválidas."""
        _make_stems(self.fid)
        hashes = ci.get_stem_hashes(self.fid)
        config_hash = ci.compute_config_hash({"test": 1})

        # Configura tudo com hash_a
        ci.save_demucs_metadata(self.fid, self.hash_a)
        for stem in ci.TRANSCRIBED_STEMS:
            _make_transcription_files(self.fid, stem)
            ci.save_transcription_metadata(self.fid, stem, hashes[stem], config_hash)
        _make_drums_json(self.fid)
        ci.save_drums_metadata(self.fid, hashes["drums"], config_hash)
        _make_score_files(self.fid)
        ci.save_score_metadata(self.fid, {"config_key": "ck", "audio_hash": self.hash_a})
        _make_arrangement_files(self.fid)
        ci.save_arrangement_metadata(self.fid, {"config_key": "ack", "audio_hash": self.hash_a})

        # Com hash_b: TUDO inválido
        assert ci.is_demucs_cache_valid(self.fid, self.hash_b) is False
        score_id_b = {"config_key": "ck", "audio_hash": self.hash_b}
        assert ci.is_score_cache_valid(self.fid, score_id_b) is False
        arr_id_b = {"config_key": "ack", "audio_hash": self.hash_b}
        assert ci.is_arrangement_cache_valid(self.fid, arr_id_b) is False


# ---------------------------------------------------------------------------
# CORRUPTED CACHE — safe miss
# ---------------------------------------------------------------------------

class TestCacheCorruptMetadata:
    """Metadata corrompida: MISS seguro, sem crash."""

    def setup_method(self):
        self.fid = str(uuid.uuid4())
        _make_stems(self.fid)

    def teardown_method(self):
        _cleanup_file_id(self.fid)

    def test_corrupt_demucs_metadata_safe(self):
        """JSON inválido no metadata: is_valid retorna False (não lança)."""
        meta_p = ci.get_demucs_metadata_path(self.fid)
        meta_p.parent.mkdir(parents=True, exist_ok=True)
        meta_p.write_text("{corrupt json!!!")
        # Não deve lançar exceção
        result = ci.is_demucs_cache_valid(self.fid, "any_hash")
        assert result is False

    def test_corrupt_transcription_metadata_safe(self):
        p = ci.get_transcription_metadata_path(self.fid, "vocals")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("not json at all")
        assert ci.is_transcription_cache_valid(self.fid, "vocals", "h", "c") is False

    def test_corrupt_drums_metadata_safe(self):
        p = ci.get_drums_metadata_path(self.fid)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("null")
        assert ci.is_drums_cache_valid(self.fid, "h", "c") is False

    def test_corrupt_score_metadata_safe(self):
        p = ci.get_score_metadata_path(self.fid)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("[1,2,3]")  # JSON válido mas não é dict
        assert ci.is_score_cache_valid(self.fid, {"key": "val"}) is False

    def test_corrupt_arrangement_metadata_safe(self):
        p = ci.get_arrangement_metadata_path(self.fid)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
        assert ci.is_arrangement_cache_valid(self.fid, {"key": "val"}) is False

    def test_empty_analysis_metadata_safe(self):
        p = ci.get_analysis_metadata_path(self.fid)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
        assert ci.is_analysis_cache_valid(self.fid, "hash") is False


# ---------------------------------------------------------------------------
# CONFIG HASH determinism
# ---------------------------------------------------------------------------

class TestConfigHash:

    def test_same_config_same_hash(self):
        c1 = {"a": 1, "b": 2}
        c2 = {"b": 2, "a": 1}  # ordem diferente
        assert ci.compute_config_hash(c1) == ci.compute_config_hash(c2)

    def test_different_config_different_hash(self):
        c1 = {"a": 1}
        c2 = {"a": 2}
        assert ci.compute_config_hash(c1) != ci.compute_config_hash(c2)

    def test_nested_config(self):
        c1 = {"x": {"y": [1, 2, 3]}}
        c2 = {"x": {"y": [1, 2, 3]}}
        c3 = {"x": {"y": [3, 2, 1]}}
        assert ci.compute_config_hash(c1) == ci.compute_config_hash(c2)
        assert ci.compute_config_hash(c1) != ci.compute_config_hash(c3)


# ---------------------------------------------------------------------------
# INVALIDATE ALL
# ---------------------------------------------------------------------------

class TestInvalidateAll:

    def test_invalidate_all_removes_metadata(self):
        fid = str(uuid.uuid4())
        try:
            _make_stems(fid)
            ci.save_demucs_metadata(fid, "hash_x")
            assert ci.get_demucs_metadata_path(fid).is_file()
            ci.invalidate_all_caches(fid)
            assert not ci.get_demucs_metadata_path(fid).is_file()
            assert not ci.get_transcription_metadata_path(fid, "vocals").is_file()
            assert not ci.get_drums_metadata_path(fid).is_file()
            assert not ci.get_score_metadata_path(fid).is_file()
            assert not ci.get_arrangement_metadata_path(fid).is_file()
        finally:
            _cleanup_file_id(fid)
