"""
Etapa 8.3.1 — Identidade e validação centralizada de caches do pipeline.

PRINCÍPIO: cada stage grava _metadata.json com hashes das entradas.
Validador verifica metadata, NÃO apenas file existence.

CADEIA DE IDENTIDADE:
  audio_hash
    → analysis (audio_hash + version)
    → demucs (audio_hash + model + config)
        → stems (4 x stem_hash)
            → transcription (stem_hash + version + config)
            → drums (drums_stem_hash + version + config)
                → score (analysis_id + trans_ids + drums_id + notation_version + config)
                    → arrangement (score_id + instruments + style + version)

Mudança de qualquer hash upstream → downstream invalida.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("uvicorn.error")

BASE_DIR = Path(__file__).resolve().parents[2]
STEMS_DIR = BASE_DIR / "stems"
TRANSCRIPTIONS_DIR = BASE_DIR / "transcriptions"
DRUMS_DIR = BASE_DIR / "drums"
SCORES_DIR = BASE_DIR / "scores"
SCORE_MODELS_DIR = BASE_DIR / "score_models"
ARRANGEMENTS_DIR = BASE_DIR / "arrangements"
ANALYSIS_DIR = BASE_DIR / "analysis"

EXPECTED_STEMS = ["vocals", "drums", "bass", "other"]
TRANSCRIBED_STEMS = ["vocals", "bass", "other"]

# Stage versions — incrementar quando algoritmo muda
DEMUX_STAGE_VERSION = "demucs-v1"
TRANSCRIPTION_STAGE_VERSION = "basic-pitch-v1"
DRUMS_STAGE_VERSION = "drum-spectral-v1"
NOTATION_STAGE_VERSION = "notation-v1"
ARRANGEMENT_STAGE_VERSION = "arrangement-v1"


# ---------------------------------------------------------------------------
# Helpers: atomic write + hash
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, data: Dict[str, Any]) -> bool:
    """Escrita atômica: .tmp -> os.replace(). Limpa .tmp em falha."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(str(tmp), str(path))
        return True
    except Exception as e:
        logger.debug(f"_atomic_write_json falhou {path}: {e}")
        try:
            path.with_suffix(".tmp").unlink(missing_ok=True)
        except Exception:
            pass
        return False


def _read_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    """Lê JSON com tratamento de erro — retorna None se inválido."""
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def compute_file_hash(path: Path, algo: str = "sha256") -> Optional[str]:
    """SHA-256 streaming de um arquivo."""
    if not path.is_file():
        return None
    h = hashlib.new(algo)
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        logger.debug(f"compute_file_hash falhou {path}: {e}")
        return None


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Hash determinístico de um dict de config."""
    raw = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Stage 1: ANALYSIS cache
# ---------------------------------------------------------------------------

def get_analysis_metadata_path(file_id: str) -> Path:
    return ANALYSIS_DIR / file_id / "_metadata.json"


def save_analysis_metadata(file_id: str, audio_hash: str) -> bool:
    """Salva metadata da análise (identidade)."""
    from backend.audio.analysis_cache import ANALYSIS_STAGE_VERSION
    meta = {
        "stage": "analysis",
        "audio_hash": audio_hash,
        "stage_version": ANALYSIS_STAGE_VERSION,
    }
    return _atomic_write_json(get_analysis_metadata_path(file_id), meta)


def is_analysis_cache_valid(file_id: str, audio_hash: str) -> bool:
    """Analysis válida: metadata hash + version batem, output existe."""
    from backend.audio.analysis_cache import (
        ANALYSIS_STAGE_VERSION, _analysis_cache_path,
    )
    meta = _read_json_safe(get_analysis_metadata_path(file_id))
    if meta is None:
        return False
    if meta.get("audio_hash") != audio_hash:
        return False
    if meta.get("stage_version") != ANALYSIS_STAGE_VERSION:
        return False
    # Output deve existir e ser válido
    output = _read_json_safe(_analysis_cache_path(file_id))
    if output is None:
        return False
    if output.get("audio_hash") != audio_hash:
        return False
    if output.get("analysis_stage_version") != ANALYSIS_STAGE_VERSION:
        return False
    return True


# ---------------------------------------------------------------------------
# Stage 2: DEMUCS cache
# ---------------------------------------------------------------------------

def get_demucs_metadata_path(file_id: str) -> Path:
    return STEMS_DIR / file_id / "_metadata.json"


def get_stem_hashes(file_id: str) -> Optional[Dict[str, str]]:
    """Computa SHA-256 de cada stem. Retorna None se algum stem falta."""
    hashes: Dict[str, str] = {}
    for stem in EXPECTED_STEMS:
        p = STEMS_DIR / file_id / f"{stem}.wav"
        if not p.is_file() or p.stat().st_size == 0:
            return None
        h = compute_file_hash(p)
        if h is None:
            return None
        hashes[stem] = h
    return hashes


def save_demucs_metadata(file_id: str, audio_hash: str,
                         model: str = "htdemucs") -> bool:
    """Salva metadata do Demucs com hashes de cada stem."""
    stem_hashes = get_stem_hashes(file_id)
    if stem_hashes is None:
        return False
    meta = {
        "stage": "demucs",
        "audio_hash": audio_hash,
        "demucs_model": model,
        "stage_version": DEMUX_STAGE_VERSION,
        "stem_hashes": stem_hashes,
        "stems_expected": EXPECTED_STEMS,
    }
    return _atomic_write_json(get_demucs_metadata_path(file_id), meta)


def is_demucs_cache_valid(file_id: str, audio_hash: str,
                          model: str = "htdemucs") -> bool:
    """Demucs válido: metadata hash/model/version batem, stems existem,
    hashes dos stems atuais batem com os salvos."""
    meta = _read_json_safe(get_demucs_metadata_path(file_id))
    if meta is None:
        return False
    if meta.get("audio_hash") != audio_hash:
        return False
    if meta.get("demucs_model") != model:
        return False
    if meta.get("stage_version") != DEMUX_STAGE_VERSION:
        return False
    saved_hashes = meta.get("stem_hashes")
    if not isinstance(saved_hashes, dict):
        return False
    # Verifica cada stem: existe + hash bate
    for stem in EXPECTED_STEMS:
        p = STEMS_DIR / file_id / f"{stem}.wav"
        if not p.is_file() or p.stat().st_size == 0:
            return False
        current_hash = compute_file_hash(p)
        if current_hash is None or saved_hashes.get(stem) != current_hash:
            return False
    return True


# ---------------------------------------------------------------------------
# Stage 3: TRANSCRIPTION cache
# ---------------------------------------------------------------------------

def get_transcription_metadata_path(file_id: str, stem: str) -> Path:
    return TRANSCRIPTIONS_DIR / file_id / f"_metadata_{stem}.json"


def save_transcription_metadata(file_id: str, stem: str,
                                stem_hash: str,
                                config_hash: str) -> bool:
    """Salva metadata de transcrição de um stem."""
    meta = {
        "stage": "transcription",
        "stem": stem,
        "stem_hash": stem_hash,
        "config_hash": config_hash,
        "stage_version": TRANSCRIPTION_STAGE_VERSION,
    }
    return _atomic_write_json(get_transcription_metadata_path(file_id, stem), meta)


def is_transcription_cache_valid(file_id: str, stem: str,
                                  stem_hash: str,
                                  config_hash: str) -> bool:
    """Transcription válida: metadata hash/config batem + MIDI/JSON existem."""
    meta = _read_json_safe(get_transcription_metadata_path(file_id, stem))
    if meta is None:
        return False
    if meta.get("stem_hash") != stem_hash:
        return False
    if meta.get("config_hash") != config_hash:
        return False
    if meta.get("stage_version") != TRANSCRIPTION_STAGE_VERSION:
        return False
    # Output deve existir
    from backend.audio.transcriber import MIDI_DIR, TRANSCRIPTIONS_DIR
    midi_p = MIDI_DIR / file_id / f"{stem}.mid"
    json_p = TRANSCRIPTIONS_DIR / file_id / f"{stem}.json"
    if not midi_p.is_file() or midi_p.stat().st_size == 0:
        return False
    if not json_p.is_file() or json_p.stat().st_size == 0:
        return False
    # JSON deve ser parseável
    data = _read_json_safe(json_p)
    if data is None or "events" not in data:
        return False
    return True


def are_all_transcriptions_valid(file_id: str,
                                  config_hash: str) -> bool:
    """Todos os 3 stems transcritos com hashes válidos."""
    stem_hashes = get_stem_hashes(file_id)
    if stem_hashes is None:
        return False
    for stem in TRANSCRIBED_STEMS:
        sh = stem_hashes.get(stem)
        if sh is None:
            return False
        if not is_transcription_cache_valid(file_id, stem, sh, config_hash):
            return False
    return True


# ---------------------------------------------------------------------------
# Stage 4: DRUMS cache
# ---------------------------------------------------------------------------

def get_drums_metadata_path(file_id: str) -> Path:
    return DRUMS_DIR / file_id / "_metadata.json"


def save_drums_metadata(file_id: str, drums_stem_hash: str,
                        config_hash: str) -> bool:
    """Salva metadata da transcrição de bateria."""
    meta = {
        "stage": "drums",
        "drums_stem_hash": drums_stem_hash,
        "config_hash": config_hash,
        "stage_version": DRUMS_STAGE_VERSION,
    }
    return _atomic_write_json(get_drums_metadata_path(file_id), meta)


def is_drums_cache_valid(file_id: str, drums_stem_hash: str,
                         config_hash: str) -> bool:
    """Drums válido: metadata hash/config batem + drums.json válido."""
    meta = _read_json_safe(get_drums_metadata_path(file_id))
    if meta is None:
        return False
    if meta.get("drums_stem_hash") != drums_stem_hash:
        return False
    if meta.get("config_hash") != config_hash:
        return False
    if meta.get("stage_version") != DRUMS_STAGE_VERSION:
        return False
    # Output deve existir
    from backend.drums.drum_transcriber import get_drums_json_path
    data = _read_json_safe(get_drums_json_path(file_id))
    if data is None or "events" not in data:
        return False
    return True


# ---------------------------------------------------------------------------
# Stage 5: SCORE cache
# ---------------------------------------------------------------------------

def get_score_metadata_path(file_id: str) -> Path:
    return SCORE_MODELS_DIR / file_id / "_metadata.json"


def save_score_metadata(file_id: str, identity: Dict[str, Any]) -> bool:
    """Salva metadata do score com identidade de todos os upstreams."""
    meta = {
        "stage": "score",
        "stage_version": NOTATION_STAGE_VERSION,
        **identity,  # config_key, trans_hashes, drums_hash, etc.
    }
    return _atomic_write_json(get_score_metadata_path(file_id), meta)


def is_score_cache_valid(file_id: str, expected_identity: Dict[str, Any]) -> bool:
    """Score válido: metadata identity bate + MusicXML/model existem."""
    meta = _read_json_safe(get_score_metadata_path(file_id))
    if meta is None:
        return False
    if meta.get("stage_version") != NOTATION_STAGE_VERSION:
        return False
    for key, value in expected_identity.items():
        if meta.get(key) != value:
            return False
    # Output deve existir
    from backend.notation.score_generator import get_score_paths
    xml_path, model_path = get_score_paths(file_id)
    if not xml_path.is_file() or xml_path.stat().st_size == 0:
        return False
    if not model_path.is_file() or model_path.stat().st_size == 0:
        return False
    return True


# ---------------------------------------------------------------------------
# Stage 6: ARRANGEMENT cache
# ---------------------------------------------------------------------------

def get_arrangement_metadata_path(file_id: str) -> Path:
    return ARRANGEMENTS_DIR / file_id / "_metadata.json"


def save_arrangement_metadata(file_id: str, identity: Dict[str, Any]) -> bool:
    """Salva metadata do arranjo."""
    meta = {
        "stage": "arrangement",
        "stage_version": ARRANGEMENT_STAGE_VERSION,
        **identity,
    }
    return _atomic_write_json(get_arrangement_metadata_path(file_id), meta)


def is_arrangement_cache_valid(file_id: str, expected_identity: Dict[str, Any]) -> bool:
    """Arrangement válido: metadata identity bate + MusicXML/model existem."""
    meta = _read_json_safe(get_arrangement_metadata_path(file_id))
    if meta is None:
        return False
    if meta.get("stage_version") != ARRANGEMENT_STAGE_VERSION:
        return False
    for key, value in expected_identity.items():
        if meta.get(key) != value:
            return False
    from backend.arrangement.arrangement_generator import get_arrangement_paths
    xml_p, model_p = get_arrangement_paths(file_id)
    if not xml_p.is_file() or xml_p.stat().st_size == 0:
        return False
    if not model_p.is_file() or model_p.stat().st_size == 0:
        return False
    return True


# ---------------------------------------------------------------------------
# BULK: invalidar tudo para um file_id
# ---------------------------------------------------------------------------

def invalidate_all_caches(file_id: str) -> None:
    """Remove todos os metadados de cache para um file_id."""
    for path in [
        get_analysis_metadata_path(file_id),
        get_demucs_metadata_path(file_id),
        get_drums_metadata_path(file_id),
        get_score_metadata_path(file_id),
        get_arrangement_metadata_path(file_id),
    ]:
        if path.is_file():
            try:
                path.unlink()
            except Exception:
                pass
    for stem in TRANSCRIBED_STEMS:
        p = get_transcription_metadata_path(file_id, stem)
        if p.is_file():
            try:
                p.unlink()
            except Exception:
                pass
