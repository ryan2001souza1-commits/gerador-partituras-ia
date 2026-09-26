"""
Etapa 8 — transcrição rítmica de drums (processo FastAPI principal).

Fonte exclusiva: stems/<file_id>/drums.wav (Demucs). Sem Basic Pitch.
DSP em librosa/numpy (já no .venv) via asyncio.to_thread — segundos, sem
subprocess e sem venv extra. Saída: drums/<file_id>/drums.json.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.drums.drum_utils import DRUM_VERSION, transcribe_drums
from backend.musical.cleanup import validate_cleanup_profile

logger = logging.getLogger("uvicorn.error")

BASE_DIR = Path(__file__).resolve().parents[2]
DRUMS_DIR = BASE_DIR / "drums"
DRUMS_DIR.mkdir(parents=True, exist_ok=True)

# Bound generoso e documentado (análise leva segundos em CPU).
DRUM_TIMEOUT = 600


def _validate_file_id(file_id: str) -> bool:
    try:
        uuid.UUID(file_id)
        return True
    except ValueError:
        return False


def get_drums_wav(file_id: str) -> Optional[Path]:
    """Localiza stems/<file_id>/drums.wav de forma segura."""
    if not _validate_file_id(file_id):
        return None
    from backend.audio.stem_separator import STEMS_DIR
    p = STEMS_DIR / file_id / "drums.wav"
    try:
        if p.is_file():
            p.resolve().relative_to(STEMS_DIR.resolve())
            if p.stat().st_size > 0:
                return p
    except (ValueError, OSError):
        return None
    return None


def get_drums_json_path(file_id: str) -> Path:
    return DRUMS_DIR / file_id / "drums.json"


def drum_config_key(file_id: str, cleanup_profile: str, time_signature: str) -> str:
    raw = f"{file_id}|{cleanup_profile}|{time_signature}|{DRUM_VERSION}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def are_drums_valid(file_id: str) -> Tuple[bool, str]:
    """drums.json existe, parseável e com lista 'events'."""
    if not _validate_file_id(file_id):
        return False, "file_id inválido"
    p = get_drums_json_path(file_id)
    if not p.is_file() or p.stat().st_size == 0:
        return False, "missing"
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data.get("events"), list):
            return False, "invalid"
        return True, ""
    except Exception:
        return False, "invalid"


def get_drums_info_data(file_id: str) -> Optional[Dict[str, Any]]:
    if not _validate_file_id(file_id):
        return None
    p = get_drums_json_path(file_id)
    if not p.is_file():
        return {"file_id": file_id, "available": False}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"file_id": file_id, "available": False}
    data["available"] = True
    return data


def transcribe_drums_sync(
    drums_wav: Path,
    file_id: str,
    tempo: float,
    beat_offset: float,
    time_signature: str = "4/4",
    cleanup_profile: str = "natural",
) -> Dict[str, Any]:
    """Transcreve drums.wav -> drums.json (síncrono; chamador usa to_thread)."""
    import librosa
    profile = validate_cleanup_profile(cleanup_profile)
    y, sr = librosa.load(str(drums_wav), sr=22050, mono=True)
    duration = round(float(len(y)) / float(sr), 3) if len(y) else 0.0
    events, stats = transcribe_drums(
        y, sr, tempo=float(tempo), beat_offset=float(beat_offset or 0.0),
        time_signature=time_signature, profile=profile)
    out = get_drums_json_path(file_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "file_id": file_id,
        "duration": duration,
        "bpm": float(tempo),
        "beat_offset": float(beat_offset or 0.0),
        "time_signature": time_signature,
        "cleanup_profile": profile,
        "version": DRUM_VERSION,
        "config_key": drum_config_key(file_id, profile, time_signature),
        "events": events,
        "stats": stats,
        "warnings": [],
    }
    if not events:
        data["warnings"].append("Nenhum evento percussivo detectado.")
    if stats.get("unknown_percussion_count", 0):
        data["warnings"].append(
            f"{stats['unknown_percussion_count']} evento(s) de classe incerta.")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


async def transcribe_drums_async(
    drums_wav: Path,
    file_id: str,
    tempo: float,
    beat_offset: float,
    time_signature: str = "4/4",
    cleanup_profile: str = "natural",
    timeout: int = DRUM_TIMEOUT,
) -> Dict[str, Any]:
    """Wrapper async com timeout bound (CDN: segundos em CPU)."""
    return await asyncio.wait_for(
        asyncio.to_thread(transcribe_drums_sync, drums_wav, file_id, tempo,
                          beat_offset, time_signature, cleanup_profile),
        timeout=timeout,
    )
