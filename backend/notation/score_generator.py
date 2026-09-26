"""
Etapa 6 — orquestração da geração de MusicXML (processo FastAPI principal).

NÃO importa music21 aqui. O Score é construído em
backend/workers/notation_worker.py, executado via .venv-notation com:

    asyncio.to_thread(...) + subprocess.run(...)

(shell=False, args lista, paths absolutos, cwd explícito, timeout, PIPE).
Nunca asyncio.create_subprocess_exec (incompatível com Uvicorn no Windows).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.notation.score_utils import (
    DEFAULT_KEY_MODE,
    DEFAULT_QUANTIZATION,
    DEFAULT_TIME_SIGNATURE,
    TEMPO_FALLBACK,
    config_key,
    validate_score_config,
)
from backend.musical.cleanup import validate_cleanup_profile

logger = logging.getLogger("uvicorn.error")

BASE_DIR = Path(__file__).resolve().parents[2]
SCORES_DIR = BASE_DIR / "scores"
SCORE_MODELS_DIR = BASE_DIR / "score_models"
WORKER_PATH = BASE_DIR / "backend" / "workers" / "notation_worker.py"

SCORES_DIR.mkdir(parents=True, exist_ok=True)
SCORE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Timeout: music21 é muito mais rápido que Demucs (5 minutos).
NOTATION_TIMEOUT = 5 * 60

SCORE_STEMS = ["vocals", "bass", "other"]


# ---------------------------------------------------------------------------
# Resolução do Python de notação
# ---------------------------------------------------------------------------

def get_notation_python() -> Optional[str]:
    """Resolve .venv-notation/Scripts/python.exe (Windows) com fallbacks."""
    import sys
    env = os.getenv("NOTATION_PYTHON")
    if env:
        p = Path(env)
        if p.is_file():
            return str(p)
        logger.warning(f"NOTATION_PYTHON aponta para não-arquivo: {env}")
    candidates = [
        BASE_DIR / ".venv-notation" / "Scripts" / "python.exe",
        BASE_DIR / ".venv-notation" / "Scripts" / "python",
        BASE_DIR / ".venv-notation" / "bin" / "python",
        BASE_DIR / ".venv-notation" / "bin" / "python3",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return sys.executable


def is_notation_available() -> bool:
    py = get_notation_python()
    if not py or not Path(py).is_file():
        return False
    try:
        result = subprocess.run(
            [py, "-c", "import music21; print(music21.__version__)"],
            capture_output=True, text=True, timeout=20,
        )
        return result.returncode == 0 and bool((result.stdout or "").strip())
    except Exception:
        return False


def get_music21_version() -> Optional[str]:
    py = get_notation_python()
    if not py:
        return None
    try:
        result = subprocess.run(
            [py, "-c", "import music21; print(music21.__version__)"],
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode == 0:
            return (result.stdout or "").strip() or None
    except Exception as e:
        logger.debug(f"get_music21_version falhou: {e}")
    return None


# ---------------------------------------------------------------------------
# Paths / validação
# ---------------------------------------------------------------------------

def _validate_file_id(file_id: str) -> bool:
    try:
        uuid.UUID(file_id)
        return True
    except ValueError:
        return False


def get_score_paths(file_id: str) -> Tuple[Path, Path]:
    """Retorna (musicxml_path, model_path) absolutos."""
    return (
        SCORES_DIR / file_id / "score.musicxml",
        SCORE_MODELS_DIR / file_id / "score.json",
    )


def are_transcriptions_ready(file_id: str) -> Tuple[bool, List[str]]:
    """Pré-condição: vocals/bass/other com JSON válido (permite 0 notas)."""
    from backend.audio.transcriber import TRANSCRIPTIONS_DIR
    if not _validate_file_id(file_id):
        return False, SCORE_STEMS
    missing: List[str] = []
    for stem in SCORE_STEMS:
        p = TRANSCRIPTIONS_DIR / file_id / f"{stem}.json"
        if not p.is_file() or p.stat().st_size == 0:
            missing.append(stem)
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "events" not in data:
                missing.append(stem)
        except Exception:
            missing.append(stem)
    return (len(missing) == 0), missing


def get_music_context(file_id: str) -> Dict[str, Any]:
    """BPM/tonalidade/modo + beat_offset da Etapa 3 (uso interno).

    Retorna dict com tempo/key/mode/key_confidence/beat_offset/warnings.
    beat_offset = first_beat_time quando há beat grid confiável, senão 0.
    Nunca expõe milhares de beat_times ao frontend.
    """
    ctx: Dict[str, Any] = {
        "tempo": None,
        "key": None,
        "mode": None,
        "key_confidence": None,
        "beat_offset": 0.0,
        "beat_grid": False,
        "warnings": [],
    }
    # Localiza upload
    try:
        from app import _find_upload_path  # import local p/ evitar ciclo
    except Exception:
        _find_upload_path = None  # type: ignore
    upload_path = None
    if _find_upload_path is not None:
        try:
            upload_path = _find_upload_path(file_id)
        except Exception:
            upload_path = None
    if upload_path is None:
        # Fallback: busca direta por extensões
        try:
            from app import ALLOWED_EXTENSIONS, UPLOAD_DIR
            for ext in ALLOWED_EXTENSIONS:
                cand = UPLOAD_DIR / f"{file_id}{ext}"
                if cand.is_file():
                    upload_path = cand
                    break
        except Exception:
            pass
    if upload_path is None or not Path(upload_path).is_file():
        ctx["warnings"].append("Áudio original não localizado; beat grid indisponível (offset 0).")
        return ctx
    try:
        from backend.audio.music_analysis import analyze_music, get_beat_grid, estimate_beat_offset
    except Exception as e:
        logger.debug(f"music_analysis indisponível: {e}")
        return ctx
    try:
        # ETAPA 8.3: passa file_id para usar cache persistente
        res = analyze_music(Path(upload_path), file_id=file_id)
        d = res.to_api_dict()
        if d.get("bpm_rounded"):
            ctx["tempo"] = int(d["bpm_rounded"])
        elif d.get("bpm"):
            ctx["tempo"] = int(round(float(d["bpm"])))
        ctx["key"] = d.get("key")
        ctx["mode"] = d.get("mode")
        ctx["key_confidence"] = d.get("key_confidence")
        # PERFORMANCE: first_beat_time já vem do analyze_music (extraído do
        # beat_track interno). Elimina get_beat_grid() separado que
        # re-decodificava o arquivo + executava HPSS + beat_track novamente
        # (economia: ~30s para áudio de 3min).
        first_beat_from_analysis = res.first_beat_time
    except Exception as e:
        logger.debug(f"analyze_music falhou p/ score ctx: {e}")
        ctx["warnings"].append("Análise musical indisponível para defaults.")
        first_beat_from_analysis = None
    # Início das primeiras notas transcritas (refino do offset, Etapa 7).
    earliest_note: Optional[float] = None
    try:
        from backend.audio.transcriber import TRANSCRIPTIONS_DIR
        starts: List[float] = []
        for stem in ("vocals", "bass", "other"):
            p = TRANSCRIPTIONS_DIR / file_id / f"{stem}.json"
            if not p.is_file():
                continue
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            for ev in data.get("events", []) or []:
                try:
                    starts.append(float(ev["start"]))
                except (TypeError, ValueError, KeyError):
                    continue
        if starts:
            earliest_note = min(s for s in starts if s >= 0)
            ctx["earliest_note_time"] = earliest_note
    except Exception as e:
        logger.debug(f"earliest_note falhou: {e}")
    try:
        # PERFORMANCE: usa first_beat_time do analyze_music (já computado no
        # beat_track interno). Só chama get_beat_grid() como FALLBACK se a
        # análise não conseguiu extrair o primeiro beat.
        # Antes: get_beat_grid() era chamado SEMPRE, re-decodificando o
        # arquivo e re-executando HPSS + beat_track (redundância ~30s).
        first_beat = first_beat_from_analysis
        beat_times_full = None
        if first_beat is None:
            # Fallback: análise não produziu first_beat, tenta beat grid dedicado
            grid = get_beat_grid(Path(upload_path))
            first_beat = grid.get("first_beat_time") if grid else None
            beat_times_full = grid.get("beat_times") if grid else None
        ctx["first_beat_time"] = first_beat
        if first_beat is not None and ctx.get("tempo"):
            # Back-projection (Etapa 7): offset normalizado p/ dentro de 1 beat.
            ctx["beat_offset"] = float(estimate_beat_offset(
                first_beat, ctx["tempo"],
                beat_times=beat_times_full,
                earliest_note_time=earliest_note,
            ))
            ctx["beat_grid"] = True
            if first_beat >= (60.0 / float(ctx["tempo"])):
                ctx["warnings"].append(
                    f"Primeiro beat confiável em {first_beat:.2f}s; "
                    f"grade projetada para trás (offset {ctx['beat_offset']:.3f}s)."
                )
        elif first_beat is not None:
            ctx["beat_offset"] = float(first_beat)
            ctx["beat_grid"] = True
        else:
            ctx["warnings"].append("Beat grid indisponível; usando beat_offset=0.")
    except Exception as e:
        logger.debug(f"get_beat_grid falhou: {e}")
        ctx["warnings"].append("Beat grid indisponível; usando beat_offset=0.")
    return ctx


def get_score_info(file_id: str) -> Optional[Dict[str, Any]]:
    """Lê score_models/<file_id>/score.json para GET /api/score/{file_id}."""
    if not _validate_file_id(file_id):
        return None
    musicxml_path, model_path = get_score_paths(file_id)
    if not model_path.is_file():
        return {
            "file_id": file_id,
            "available": False,
            "musicxml_available": musicxml_path.is_file(),
        }
    try:
        with open(model_path, "r", encoding="utf-8") as f:
            model = json.load(f)
    except Exception:
        return {"file_id": file_id, "available": False}
    model["available"] = True
    model["musicxml_available"] = musicxml_path.is_file()
    if musicxml_path.is_file():
        model["musicxml_url"] = f"/api/score/{file_id}/musicxml"
        try:
            model["musicxml_size"] = musicxml_path.stat().st_size
        except Exception:
            pass
    return model


# ---------------------------------------------------------------------------
# Comando do worker
# ---------------------------------------------------------------------------

def _build_worker_command(
    python_path: str,
    file_id: str,
    tempo: int,
    time_signature: str,
    quantization: str,
    key_mode: str,
    key: Optional[str],
    mode: Optional[str],
    key_confidence: Optional[float],
    beat_offset: float,
    output_musicxml: Path,
    output_model: Path,
    cleanup_profile: str = "natural",
) -> List[str]:
    from backend.audio.transcriber import TRANSCRIPTIONS_DIR
    cmd = [
        str(Path(python_path).resolve()),
        str(WORKER_PATH.resolve()),
        "--file-id", file_id,
        "--transcriptions-dir", str((TRANSCRIPTIONS_DIR / file_id).resolve()),
        "--output-musicxml", str(Path(output_musicxml).resolve()),
        "--output-model", str(Path(output_model).resolve()),
        "--tempo", str(tempo),
        "--time-signature", time_signature,
        "--quantization", quantization,
        "--key-mode", key_mode,
        "--beat-offset", str(beat_offset),
        "--cleanup-profile", cleanup_profile,
    ]
    if key:
        cmd.extend(["--key", str(key)])
    if mode:
        cmd.extend(["--mode", str(mode)])
    if key_confidence is not None:
        cmd.extend(["--key-confidence", str(key_confidence)])
    logger.info(f"Notation comando: {[repr(c) for c in cmd]}")
    return cmd


def _run_notation_sync(cmd: List[str], timeout: int = NOTATION_TIMEOUT) -> Tuple[int, str, str]:
    cwd = str(BASE_DIR.resolve())
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            timeout=timeout,
        )
        out = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
        err = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        logger.info(f"Notation sync rc={result.returncode} out_tail={repr(out[-500:])} err_tail={repr(err[-800:])}")
        return result.returncode, out, err
    except subprocess.TimeoutExpired as e:
        logger.error(f"Notation timeout após {timeout}s: {e}")
        raise TimeoutError(f"Geração da partitura excedeu {timeout}s") from e
    except FileNotFoundError as e:
        raise RuntimeError("Python de notação/worker não encontrado") from e


async def _run_notation_async(cmd: List[str], timeout: int = NOTATION_TIMEOUT) -> Tuple[int, str, str]:
    try:
        loop = asyncio.get_running_loop()
        logger.info(f"Event loop notation: type={type(loop).__name__}")
    except Exception:
        pass
    return await asyncio.to_thread(_run_notation_sync, cmd, timeout)


# ---------------------------------------------------------------------------
# Geração
# ---------------------------------------------------------------------------

async def generate_score_async(
    file_id: str,
    tempo: Optional[Any] = None,
    time_signature: str = DEFAULT_TIME_SIGNATURE,
    quantization: str = DEFAULT_QUANTIZATION,
    key_mode: str = DEFAULT_KEY_MODE,
    cleanup_profile: str = "natural",
    timeout: int = NOTATION_TIMEOUT,
) -> Dict[str, Any]:
    """Gera score.musicxml + score.json. Retorna model dict.

    - Valida config (30-300 BPM, allowlists, perfil natural/detailed).
    - Se tempo None: usa BPM da Etapa 3; fallback 120 com warning.
    - Idempotência: se score.json existe com mesma config key, reutiliza.
    """
    if not _validate_file_id(file_id):
        raise ValueError("file_id inválido.")
    profile = validate_cleanup_profile(cleanup_profile)
    ok, missing = are_transcriptions_ready(file_id)
    if not ok:
        raise FileNotFoundError(f"Transcreva os instrumentos antes de gerar a partitura. Faltando: {missing}")

    cfg = validate_score_config(
        tempo if tempo is not None else TEMPO_FALLBACK,  # valida formato primeiro
        time_signature, quantization, key_mode,
    ) if tempo is not None else validate_score_config(
        TEMPO_FALLBACK, time_signature, quantization, key_mode
    )
    # Resolve tempo real: fornecido > Etapa 3 > fallback
    warnings: List[str] = []
    tempo_val = cfg["tempo"]
    if tempo is None:
        ctx = get_music_context(file_id)
        if ctx.get("tempo"):
            tempo_val = int(ctx["tempo"])
        else:
            tempo_val = TEMPO_FALLBACK
            warnings.append(
                "BPM da Etapa 3 indisponível; usando andamento provisório 120. "
                "Revise o BPM antes de exportar para o MuseScore."
            )
        warnings.extend(ctx.get("warnings", []))
        cfg = validate_score_config(tempo_val, time_signature, quantization, key_mode)
    else:
        # Mesmo com tempo explícito, coleta contexto de tonalidade/beat grid
        ctx = get_music_context(file_id)
        warnings.extend(ctx.get("warnings", []))

    key = ctx.get("key")
    mode = ctx.get("mode")
    key_conf = ctx.get("key_confidence")
    beat_offset = float(ctx.get("beat_offset") or 0.0)

    musicxml_path, model_path = get_score_paths(file_id)
    ck = config_key(tempo_val, time_signature, quantization, key_mode,
                    cleanup_profile=profile)

    # Idempotência: mesma config -> reutiliza sem reexecutar worker
    if model_path.is_file() and musicxml_path.is_file():
        try:
            with open(model_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("config_key") == ck and existing.get("tempo") == tempo_val:
                logger.info(f"Score idempotente file_id={file_id} config_key={ck}")
                existing["already_completed"] = True
                return existing
        except Exception as e:
            logger.debug(f"Idempotência check falhou, irá regenerar: {e}")

    py = get_notation_python()
    if not py or not Path(py).is_file():
        raise RuntimeError("Python de notação não encontrado (.venv-notation)")
    if not WORKER_PATH.is_file():
        raise RuntimeError(f"Worker não encontrado: {WORKER_PATH}")
    if not is_notation_available():
        raise RuntimeError("music21 não está instalado. Configure .venv-notation com music21==10.5.0")

    musicxml_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    # Remove saídas parciais antigas (regen determinística)
    for p in (musicxml_path, model_path):
        if p.is_file():
            try:
                p.unlink()
            except Exception:
                pass

    cmd = _build_worker_command(
        py, file_id, int(tempo_val), time_signature, quantization, key_mode,
        key, mode, key_conf, beat_offset, musicxml_path, model_path, profile,
    )
    rc, stdout, stderr = await _run_notation_async(cmd, timeout=timeout)
    if rc != 0:
        logger.error(f"Notation worker falhou rc={rc} stderr={stderr[:800]}")
        raise RuntimeError("Não foi possível gerar a partitura.")

    if not musicxml_path.is_file() or musicxml_path.stat().st_size == 0:
        raise RuntimeError("MusicXML não gerado.")
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise RuntimeError("Modelo da partitura não gerado.")

    with open(model_path, "r", encoding="utf-8") as f:
        model = json.load(f)
    # Anexa warnings de resolução de defaults (além dos do worker)
    if warnings:
        model_warnings = model.get("warnings", [])
        for w in warnings:
            if w not in model_warnings:
                model_warnings.append(w)
        model["warnings"] = model_warnings
        # Persiste warnings combinados
        try:
            with open(model_path, "w", encoding="utf-8") as f:
                json.dump(model, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return model
