"""
Etapa 7 — orquestração do arranjo (processo FastAPI; NÃO importa music21).

Pré-condição: partitura base da Etapa 6 (score_models/<file_id>/score.json),
de onde vêm tempo/compasso/quantização/tonalidade/beat_offset — garantindo
consistência entre base e arranjo. Execução via asyncio.to_thread +
subprocess.run no worker .venv-notation (5 min).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.arrangement.instrument_definitions import (
    MAX_ARRANGEMENT_INSTRUMENTS,
    SUPPORTED_ARRANGE_MODES,
    InstrumentDefinition,
    get_instrument,
    list_instruments,
)

logger = logging.getLogger("uvicorn.error")

BASE_DIR = Path(__file__).resolve().parents[2]
ARRANGEMENTS_DIR = BASE_DIR / "arrangements"
ARRANGE_WORKER_PATH = BASE_DIR / "backend" / "workers" / "arrangement_worker.py"

ARRANGEMENTS_DIR.mkdir(parents=True, exist_ok=True)

ARRANGE_TIMEOUT = 5 * 60


def _validate_file_id(file_id: str) -> bool:
    try:
        uuid.UUID(file_id)
        return True
    except ValueError:
        return False


def get_arrangement_paths(file_id: str) -> Tuple[Path, Path]:
    return (
        ARRANGEMENTS_DIR / file_id / "arrangement.musicxml",
        ARRANGEMENTS_DIR / file_id / "arrangement.json",
    )


def arrangement_config_key(instruments: List[str], mode: str, include_originals: bool,
                           cleanup_profile: str = "natural") -> str:
    raw = f"{','.join(sorted(instruments))}|{mode}|{int(bool(include_originals))}|{cleanup_profile}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def validate_arrange_config(
    instruments: Any, mode: Any = "automatic", include_original_parts: Any = True,
) -> Dict[str, Any]:
    """Valida seleção (allowlist, 1..5) e modo. Lança ValueError."""
    if not isinstance(instruments, list) or not instruments:
        raise ValueError("Selecione ao menos 1 instrumento.")
    seen, clean = set(), []
    for i in instruments:
        if not isinstance(i, str) or get_instrument(i) is None:
            raise ValueError(f"Instrumento inválido: {i!r}.")
        if i not in seen:
            seen.add(i)
            clean.append(i)
    if len(clean) > MAX_ARRANGEMENT_INSTRUMENTS:
        raise ValueError(f"Máximo de {MAX_ARRANGEMENT_INSTRUMENTS} instrumentos.")
    if mode not in SUPPORTED_ARRANGE_MODES:
        raise ValueError(f"mode inválido. Permitidos: {', '.join(SUPPORTED_ARRANGE_MODES)}.")
    return {"instruments": clean, "mode": mode,
            "include_original_parts": bool(include_original_parts)}


def read_base_score(file_id: str) -> Optional[Dict[str, Any]]:
    """Lê score.json da Etapa 6 (pré-condição do arranjo)."""
    from backend.notation.score_generator import SCORE_MODELS_DIR
    p = SCORE_MODELS_DIR / file_id / "score.json"
    if not p.is_file():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_arrangement_info(file_id: str) -> Optional[Dict[str, Any]]:
    if not _validate_file_id(file_id):
        return None
    xml_p, model_p = get_arrangement_paths(file_id)
    if not model_p.is_file():
        return {"file_id": file_id, "available": False,
                "musicxml_available": xml_p.is_file()}
    try:
        with open(model_p, "r", encoding="utf-8") as f:
            model = json.load(f)
    except Exception:
        return {"file_id": file_id, "available": False}
    model["available"] = True
    model["musicxml_available"] = xml_p.is_file()
    if xml_p.is_file():
        model["musicxml_url"] = f"/api/arrangement/{file_id}/musicxml"
        try:
            model["musicxml_size"] = xml_p.stat().st_size
        except Exception:
            pass
    return model


def instruments_info() -> List[Dict[str, Any]]:
    out = []
    for d in list_instruments():
        t = d.transposition_semitones
        out.append({
            "id": d.id, "name": d.name, "short_name": d.short_name,
            "family": d.family, "concert_key": d.concert_key,
            "transposition_semitones": t, "written_range": [d.written_low, d.written_high],
            "preferred_range": [d.preferred_low, d.preferred_high],
            "clef": d.clef, "music21_instrument": d.music21_instrument,
        })
    return out


def _build_arrange_command(
    python_path: str, file_id: str, base: Dict[str, Any],
    instruments: List[str], mode: str, include_originals: bool,
    output_musicxml: Path, output_model: Path, cleanup_profile: str = "natural",
) -> List[str]:
    from backend.audio.transcriber import TRANSCRIPTIONS_DIR
    cmd = [
        str(Path(python_path).resolve()),
        str(ARRANGE_WORKER_PATH.resolve()),
        "--file-id", file_id,
        "--transcriptions-dir", str((TRANSCRIPTIONS_DIR / file_id).resolve()),
        "--output-musicxml", str(Path(output_musicxml).resolve()),
        "--output-model", str(Path(output_model).resolve()),
        "--tempo", str(base.get("tempo", 120)),
        "--time-signature", str(base.get("time_signature", "4/4")),
        "--quantization", str(base.get("quantization", "1/16")),
        "--key-mode", str(base.get("key_mode", "auto")),
        "--beat-offset", str(base.get("beat_offset", 0.0)),
        "--instruments", ",".join(instruments),
        "--mode", mode,
    ]
    if base.get("key"):
        cmd.extend(["--key", str(base["key"])])
    if base.get("mode"):
        cmd.extend(["--concert-mode", str(base["mode"])])
    if base.get("key_confidence") is not None:
        cmd.extend(["--key-confidence", str(base["key_confidence"])])
    if include_originals:
        cmd.append("--include-original-parts")
    cmd.extend(["--base-config-key", str(base.get("config_key", ""))])
    cmd.extend(["--cleanup-profile", cleanup_profile])
    logger.info(f"Arrange comando: {[repr(c) for c in cmd]}")
    return cmd


def _run_arrange_sync(cmd: List[str], timeout: int = ARRANGE_TIMEOUT) -> Tuple[int, str, str]:
    cwd = str(BASE_DIR.resolve())
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                cwd=cwd, timeout=timeout)
        out = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
        err = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        logger.info(f"Arrange sync rc={result.returncode} out_tail={repr(out[-500:])}")
        return result.returncode, out, err
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"Arranjo excedeu {timeout}s") from e
    except FileNotFoundError as e:
        raise RuntimeError("Python de notação/worker não encontrado") from e


async def generate_arrangement_async(
    file_id: str, instruments: List[str], mode: str = "automatic",
    include_original_parts: bool = True, cleanup_profile: str = "natural",
    timeout: int = ARRANGE_TIMEOUT,
) -> Dict[str, Any]:
    """Gera arrangement.musicxml + arrangement.json (idempotente por config)."""
    if not _validate_file_id(file_id):
        raise ValueError("file_id inválido.")
    from backend.musical.cleanup import validate_cleanup_profile
    profile = validate_cleanup_profile(cleanup_profile)
    cfg = validate_arrange_config(instruments, mode, include_original_parts)
    base = read_base_score(file_id)
    if not base:
        raise FileNotFoundError("Gere a partitura base antes de criar o arranjo.")
    xml_p, model_p = get_arrangement_paths(file_id)
    ck = arrangement_config_key(cfg["instruments"], cfg["mode"],
                                cfg["include_original_parts"], cleanup_profile=profile)
    base_ck = str(base.get("config_key", ""))
    if model_p.is_file() and xml_p.is_file():
        try:
            with open(model_p, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("config_key") == ck and existing.get("base_config_key") == base_ck:
                existing["already_completed"] = True
                return existing
        except Exception as e:
            logger.debug(f"Idempotência arrange falhou, regenera: {e}")

    from backend.notation.score_generator import get_notation_python, is_notation_available
    py = get_notation_python()
    if not py or not Path(py).is_file():
        raise RuntimeError("Python de notação não encontrado (.venv-notation)")
    if not ARRANGE_WORKER_PATH.is_file():
        raise RuntimeError(f"Worker não encontrado: {ARRANGE_WORKER_PATH}")
    if not is_notation_available():
        raise RuntimeError("music21 não está instalado. Configure .venv-notation com music21==10.5.0")

    xml_p.parent.mkdir(parents=True, exist_ok=True)
    for p in (xml_p, model_p):
        if p.is_file():
            try:
                p.unlink()
            except Exception:
                pass
    cmd = _build_arrange_command(py, file_id, base, cfg["instruments"],
                                 cfg["mode"], cfg["include_original_parts"], xml_p, model_p,
                                 profile)
    try:
        loop = asyncio.get_running_loop()
        logger.info(f"Event loop arrange: type={type(loop).__name__}")
    except Exception:
        pass
    rc, stdout, stderr = await asyncio.to_thread(_run_arrange_sync, cmd, timeout)
    if rc != 0:
        logger.error(f"Arrange worker falhou rc={rc} stderr={stderr[:800]}")
        raise RuntimeError("Não foi possível gerar o arranjo.")
    if not xml_p.is_file() or xml_p.stat().st_size == 0:
        raise RuntimeError("MusicXML do arranjo não gerado.")
    if not model_p.is_file():
        raise RuntimeError("Modelo do arranjo não gerado.")
    with open(model_p, "r", encoding="utf-8") as f:
        return json.load(f)
