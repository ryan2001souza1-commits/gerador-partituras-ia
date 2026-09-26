"""
Etapa 8.3 — Cache persistente da análise musical.

COMPUTAR UMA VEZ + VALIDAR + CACHEAR + REUTILIZAR.

Identidade do cache: audio_sha256 + ANALYSIS_STAGE_VERSION + config.
Escrita atômica: .tmp -> os.replace(). Nunca NaN/Infinity no JSON.
Cache hit: NÃO roda librosa.load, HPSS, chroma, beat_track, key/BPM windows.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("uvicorn.error")

BASE_DIR = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = BASE_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Versão do algoritmo de análise — invalida cache ao mudar
# v3: inclui window analysis (key windows + BPM windows + stability)
ANALYSIS_STAGE_VERSION = "analysis-v3"

# Campos obrigatórios no cache (validação)
_REQUIRED_FIELDS = [
    "file_id", "audio_hash", "analysis_stage_version",
    "duration", "bpm", "key", "mode",
]

# Campos que não podem ser NaN/Infinity
_FLOAT_FIELDS = [
    "duration", "bpm", "bpm_confidence", "bpm_stability",
    "bpm_local_median", "bpm_local_mad", "bpm_window_agreement",
    "beat_grid_mean_error_ms", "beat_grid_p95_error_ms",
    "first_beat_time", "key_confidence", "key_window_agreement",
    "key_score_margin", "key_method_agreement",
]


def _analysis_cache_path(file_id: str) -> Path:
    return ANALYSIS_DIR / file_id / "analysis.json"


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Escrita atômica: .tmp -> os.replace(). Limpa .tmp em falha."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(str(tmp), str(path))
    except Exception:
        tmp.unlink(missing_ok=True)  # Limpa .tmp órfão
        raise


def _sanitize_float(v: Any) -> Any:
    """Converte NaN/Infinity para None (JSON válido)."""
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _validate_cache(data: Dict[str, Any], audio_hash: str,
                    stage_version: str) -> bool:
    """Cache válido: parseável, hash igual, version igual, campos presentes,
    floats finitos, duration > 0."""
    if not isinstance(data, dict):
        return False
    for field in _REQUIRED_FIELDS:
        if field not in data:
            return False
    if data.get("audio_hash") != audio_hash:
        return False
    if data.get("analysis_stage_version") != stage_version:
        return False
    # Valida floats
    for field in _FLOAT_FIELDS:
        if field in data and data[field] is not None:
            if not math.isfinite(float(data[field])):
                return False
    # Duration deve ser > 0
    duration = data.get("duration")
    if duration is not None and float(duration) <= 0:
        return False
    # BPM quando presente deve ser válido
    bpm = data.get("bpm")
    if bpm is not None and (float(bpm) <= 0 or float(bpm) > 1000):
        return False
    return True


def get_cached_analysis(file_id: str, audio_hash: str) -> Optional[Dict[str, Any]]:
    """Retorna análise em cache se válida, ou None."""
    if not audio_hash:
        return None
    cache_path = _analysis_cache_path(file_id)
    if not cache_path.is_file() or cache_path.stat().st_size == 0:
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    if not _validate_cache(data, audio_hash, ANALYSIS_STAGE_VERSION):
        return None
    data["analysis_cache_hit"] = True
    return data


def save_analysis_cache(file_id: str, audio_hash: str,
                        result_dict: Dict[str, Any]) -> bool:
    """Salva análise em cache com escrita atômica. Retorna True se OK."""
    if not audio_hash:
        return False
    # Sanitiza floats (remove NaN/Infinity)
    clean = dict(result_dict)
    for field in _FLOAT_FIELDS:
        if field in clean:
            clean[field] = _sanitize_float(clean[field])
    clean["file_id"] = file_id
    clean["audio_hash"] = audio_hash
    clean["analysis_stage_version"] = ANALYSIS_STAGE_VERSION
    clean["cached_at"] = time.time()
    try:
        _atomic_write_json(_analysis_cache_path(file_id), clean)
        return True
    except Exception as e:
        logger.warning(f"save_analysis_cache falhou: {e}")
        return False


def result_to_cache_dict(result) -> Dict[str, Any]:
    """Converte MusicAnalysisResult para dict de cache (todos os campos)."""
    d = result.to_api_dict()
    # Campos internos adicionais que o cache deve preservar
    d["first_beat_time"] = result.first_beat_time
    d["beats_count"] = result.beats_count
    d["duration"] = result.duration_analyzed
    return d


def clear_analysis_cache(file_id: str) -> None:
    """Remove cache de análise para um file_id."""
    p = _analysis_cache_path(file_id)
    if p.is_file():
        try:
            p.unlink()
        except Exception:
            pass
