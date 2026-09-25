"""
Transcrição de stems para notas/MIDI via Basic Pitch — Etapa 5.

Responsabilidades:
- encontrar Python do Basic Pitch (.venv-basicpitch)
- verificar instalação e runtime (ONNX/TensorFlow)
- executar transcrição via worker isolado (basic_pitch_worker.py)
- localizar MIDI/JSON, validar, normalizar saída
- salvar em midi/<file_id>/ e transcriptions/<file_id>/
- limpar temporários, idempotência, timeout

Processa apenas: vocals, bass, other (não drums).
Não quantizar; preservar timing bruto.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Stems que serão transcritos (não drums)
TRANSCRIBED_STEMS = ["vocals", "bass", "other"]
# Todos os stems possíveis (para validação de rejeição)
ALL_STEMS = ["vocals", "drums", "bass", "other"]

BASE_DIR = Path(__file__).resolve().parents[2]
MIDI_DIR = BASE_DIR / "midi"
TRANSCRIPTIONS_DIR = BASE_DIR / "transcriptions"
WORKER_PATH = BASE_DIR / "backend" / "workers" / "basic_pitch_worker.py"

# Garante diretórios
MIDI_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Timeout por stem: 20 minutos (1200s) — centralizado
BASIC_PITCH_TIMEOUT = 20 * 60

# Configuração de frequência por stem (conservadora, validada com testes)
# None = sem filtro (faixa ampla)
STEM_FREQ_RANGES = {
    "vocals": {"minimum_frequency": 70.0, "maximum_frequency": 2000.0},
    "bass": {"minimum_frequency": 30.0, "maximum_frequency": 500.0},
    "other": {"minimum_frequency": None, "maximum_frequency": None},  # faixa ampla
}

# Thresholds Basic Pitch — defaults oficiais, centralizados
BASIC_PITCH_DEFAULTS = {
    "onset_threshold": 0.5,
    "frame_threshold": 0.3,
    "minimum_note_length": 127.7,  # ms
    "midi_tempo": 120.0,
    "multiple_pitch_bends": False,
    "melodia_trick": True,
}

# ---------------------------------------------------------------------------
# Resolução do Python Basic Pitch
# ---------------------------------------------------------------------------

def get_basic_pitch_python() -> Optional[str]:
    """
    Resolve caminho para python do Basic Pitch de forma segura.
    Ordem:
      1. env BASIC_PITCH_PYTHON se existir e for arquivo
      2. .venv-basicpitch/Scripts/python.exe (Windows)
      3. .venv-basicpitch/Scripts/python
      4. .venv-basicpitch/bin/python (Linux/Mac)
      5. fallback sys.executable (para testes)
    """
    import sys
    env = os.getenv("BASIC_PITCH_PYTHON")
    if env:
        p = Path(env)
        if p.is_file():
            return str(p)
        logger.warning(f"BASIC_PITCH_PYTHON env aponta para não-arquivo: {env}")

    candidates = [
        BASE_DIR / ".venv-basicpitch" / "Scripts" / "python.exe",
        BASE_DIR / ".venv-basicpitch" / "Scripts" / "python",
        BASE_DIR / ".venv-basicpitch" / "bin" / "python",
        BASE_DIR / ".venv-basicpitch" / "bin" / "python3",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    logger.debug("Basic Pitch python não encontrado em .venv-basicpitch, usando sys.executable como fallback")
    return sys.executable

def is_basic_pitch_python_available() -> bool:
    p = get_basic_pitch_python()
    return bool(p and Path(p).is_file())

def get_basic_pitch_version() -> Optional[str]:
    py = get_basic_pitch_python()
    if not py:
        return None
    try:
        result = subprocess.run(
            [py, "-m", "pip", "show", "basic-pitch"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in (result.stdout or "").splitlines():
                if line.lower().startswith("version:"):
                    return line.split(":", 1)[1].strip()
        # fallback import
        result2 = subprocess.run(
            [py, "-c", "import basic_pitch; print(getattr(basic_pitch, '__version__', 'unknown'))"],
            capture_output=True, text=True, timeout=10
        )
        if result2.returncode == 0:
            return (result2.stdout or "").strip()
    except Exception as e:
        logger.debug(f"get_basic_pitch_version falhou: {e}")
    return None

def is_basic_pitch_available() -> bool:
    py = get_basic_pitch_python()
    if not py or not Path(py).is_file():
        return False
    try:
        result = subprocess.run(
            [py, "-c", "import basic_pitch; print('ok')"],
            capture_output=True, text=True, timeout=15
        )
        return result.returncode == 0 and "ok" in (result.stdout or "")
    except Exception:
        return False

def get_runtime_info() -> Dict[str, object]:
    """
    Retorna info do runtime Basic Pitch (ONNX/TensorFlow) de forma serializável.
    Tenta detectar qual runtime está disponível.
    """
    py = get_basic_pitch_python()
    info: Dict[str, object] = {}
    if not py:
        return info
    try:
        # Script Python válido com newlines e try/except corretos (sem '; try:')
        script = (
            "try:\n"
            "    import onnxruntime\n"
            "    print(f\"onnx:{onnxruntime.__version__}\")\n"
            "except Exception as e:\n"
            "    print('onnx:missing')\n"
            "try:\n"
            "    import tensorflow as tf\n"
            "    print(f\"tf:{tf.__version__}\")\n"
            "except Exception as e:\n"
            "    print(f\"tf:missing:{e}\")\n"
            "try:\n"
            "    import basic_pitch\n"
            "    print(f\"bp:{getattr(basic_pitch, '__version__', 'unknown')}\")\n"
            "except Exception as e:\n"
            "    print('bp:missing')\n"
        )
        result = subprocess.run(
            [py, "-c", script],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            for line in (result.stdout or "").strip().splitlines():
                line = line.strip()
                if line.startswith("onnx:"):
                    info["onnx_version"] = line.split(":", 1)[1]
                    info["onnx_available"] = line.split(":", 1)[1] != "missing"
                elif line.startswith("tf:"):
                    val = line.split(":", 1)[1]
                    if val.startswith("missing"):
                        info["tensorflow_available"] = False
                        info["tensorflow_error"] = val
                    else:
                        info["tensorflow_available"] = True
                        info["tensorflow_version"] = val
                elif line.startswith("bp:"):
                    info["basic_pitch_version"] = line.split(":", 1)[1]
        # Fallback simples: tenta apenas basic_pitch
        if "basic_pitch_version" not in info:
            result2 = subprocess.run(
                [py, "-c", "import basic_pitch; print(basic_pitch.__version__)"],
                capture_output=True, text=True, timeout=10
            )
            if result2.returncode == 0:
                info["basic_pitch_version"] = (result2.stdout or "").strip()
    except Exception as e:
        logger.debug(f"get_runtime_info falhou: {e}")
    # Determina runtime principal
    if info.get("onnx_available"):
        info["runtime"] = "onnx"
        info["runtime_version"] = info.get("onnx_version")
    elif info.get("tensorflow_available"):
        info["runtime"] = "tensorflow"
        info["runtime_version"] = info.get("tensorflow_version")
    else:
        info["runtime"] = "unknown"
    return info

# ---------------------------------------------------------------------------
# Validação helpers
# ---------------------------------------------------------------------------

def _validate_file_id(file_id: str) -> bool:
    try:
        uuid.UUID(file_id)
        return True
    except ValueError:
        return False

def _validate_stem_for_transcription(stem: str) -> bool:
    return stem in TRANSCRIBED_STEMS

def _get_midi_path(file_id: str, stem: str) -> Path:
    return MIDI_DIR / file_id / f"{stem}.mid"

def _get_transcription_path(file_id: str, stem: str) -> Path:
    return TRANSCRIPTIONS_DIR / file_id / f"{stem}.json"

def are_required_stems_valid(file_id: str) -> Tuple[bool, List[str]]:
    """
    Verifica se os stems necessários para transcrição (vocals, bass, other) existem e são válidos.
    Retorna (valid, missing_list). Não exige drums.
    """
    if not _validate_file_id(file_id):
        return False, TRANSCRIBED_STEMS
    # Importa are_stems_valid mas filtra apenas os 3 necessários
    from backend.audio.stem_separator import are_stems_valid as check_all, STEMS_DIR as SD
    # Verifica cada um dos 3 individualmente via are_stems_valid ou direto
    missing: List[str] = []
    for stem in TRANSCRIBED_STEMS:
        p = SD / file_id / f"{stem}.wav"
        if not p.is_file() or p.stat().st_size == 0:
            missing.append(stem)
            continue
        try:
            from backend.audio.probe import probe_audio
            probe_audio(p, file_id=file_id)
        except Exception:
            missing.append(stem)
            continue
    valid = len(missing) == 0
    return valid, missing

def are_transcriptions_valid(file_id: str) -> Tuple[bool, List[str]]:
    """
    Verifica se transcriptions/<file_id>/{vocals,bass,other}.json e midi/<file_id>/*.mid existem e são válidos.
    Retorna (valid, missing_list)
    """
    if not _validate_file_id(file_id):
        return False, TRANSCRIBED_STEMS
    missing: List[str] = []
    for stem in TRANSCRIBED_STEMS:
        midi_p = _get_midi_path(file_id, stem)
        json_p = _get_transcription_path(file_id, stem)
        if not midi_p.is_file() or midi_p.stat().st_size == 0:
            missing.append(stem)
            continue
        if not json_p.is_file() or json_p.stat().st_size == 0:
            missing.append(stem)
            continue
        # Valida JSON parseável
        try:
            with open(json_p, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Verifica campos básicos
            if "events" not in data or "notes_count" not in data:
                missing.append(stem)
                continue
        except Exception as e:
            logger.debug(f"are_transcriptions_valid JSON falhou {json_p}: {e}")
            missing.append(stem)
            continue
        # Valida MIDI com pretty_midi ou mido (no ambiente Basic Pitch via subprocess, mas aqui tentamos via pretty_midi se disponível)
        try:
            # Tenta abrir com pretty_midi se disponível no .venv principal (librosa já traz pretty_midi via basic-pitch? não)
            # Como .venv principal não tem pretty_midi, fazemos validação simples: tamanho >0 já checado, e tenta via basic-pitch python
            # Para simplicidade, apenas verifica tamanho; validação profunda é feita no worker e em transcribe_stem
            pass
        except Exception:
            pass
    valid = len(missing) == 0
    return valid, missing

def get_transcription_info(file_id: str) -> Optional[Dict]:
    """
    Retorna info para GET /api/transcriptions/{file_id}
    """
    if not _validate_file_id(file_id):
        return None
    valid, missing = are_transcriptions_valid(file_id)
    stems_info = []
    for stem in TRANSCRIBED_STEMS:
        midi_p = _get_midi_path(file_id, stem)
        json_p = _get_transcription_path(file_id, stem)
        midi_exists = midi_p.is_file() and midi_p.stat().st_size > 0
        json_exists = json_p.is_file() and json_p.stat().st_size > 0
        notes_count = None
        duration = None
        warning = None
        if json_exists:
            try:
                with open(json_p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                notes_count = data.get("notes_count")
                duration = data.get("duration")
                warning = data.get("warning")
            except:
                pass
        stems_info.append({
            "stem": stem,
            "midi_available": midi_exists,
            "transcription_available": json_exists,
            "midi_url": f"/api/midi/{file_id}/{stem}" if midi_exists else None,
            "transcription_url": f"/api/transcriptions/{file_id}/{stem}" if json_exists else None,
            "notes_count": notes_count,
            "duration": duration,
            "warning": warning,
        })
    # Também inclui drums como não transcritível
    stems_info.append({
        "stem": "drums",
        "midi_available": False,
        "transcription_available": False,
        "midi_url": None,
        "transcription_url": None,
        "notes_count": None,
        "duration": None,
        "warning": "Transcrição rítmica será adicionada em etapa futura.",
    })
    return {
        "file_id": file_id,
        "available": valid,
        "missing": missing,
        "stems": stems_info,
    }

# ---------------------------------------------------------------------------
# Construção do comando worker
# ---------------------------------------------------------------------------

def _build_worker_command(
    python_path: str,
    input_wav: Path,
    output_midi: Path,
    output_json: Path,
    stem: str,
    file_id: str,
) -> List[str]:
    """
    Constrói comando para o worker Basic Pitch de forma segura.
    Usa caminhos absolutos, lista, sem shell.
    """
    py_resolved = str(Path(python_path).resolve())
    worker_resolved = str(WORKER_PATH.resolve())
    inp_resolved = str(Path(input_wav).resolve())
    midi_resolved = str(Path(output_midi).resolve())
    json_resolved = str(Path(output_json).resolve())

    # Frequências por stem
    freq_config = STEM_FREQ_RANGES.get(stem, {})
    min_freq = freq_config.get("minimum_frequency")
    max_freq = freq_config.get("maximum_frequency")

    defaults = BASIC_PITCH_DEFAULTS

    cmd = [
        py_resolved,
        str(worker_resolved),
        "--input", inp_resolved,
        "--output-midi", midi_resolved,
        "--output-json", json_resolved,
        "--stem", stem,
        "--file-id", file_id,
        "--onset-threshold", str(defaults["onset_threshold"]),
        "--frame-threshold", str(defaults["frame_threshold"]),
        "--minimum-note-length", str(defaults["minimum_note_length"]),
        "--midi-tempo", str(defaults["midi_tempo"]),
    ]
    if min_freq is not None:
        cmd.extend(["--minimum-frequency", str(min_freq)])
    if max_freq is not None:
        cmd.extend(["--maximum-frequency", str(max_freq)])
    if defaults.get("multiple_pitch_bends"):
        cmd.append("--multiple-pitch-bends")
    if not defaults.get("melodia_trick", True):
        cmd.append("--no-melodia-trick")

    logger.info(f"Basic Pitch comando: {[repr(c) for c in cmd]}")
    return cmd

# ---------------------------------------------------------------------------
# Execução síncrona em thread worker (evita Uvicorn Windows limitação)
# ---------------------------------------------------------------------------

def _run_basic_pitch_sync(
    input_wav: Path,
    output_midi: Path,
    output_json: Path,
    stem: str,
    file_id: str,
    timeout: int = BASIC_PITCH_TIMEOUT,
) -> Tuple[int, str, str]:
    """
    Executa worker via subprocess.run síncrono em thread.
    Retorna (returncode, stdout, stderr).
    """
    py = get_basic_pitch_python()
    if not py or not Path(py).is_file():
        raise RuntimeError("Python Basic Pitch não encontrado (is_basic_pitch_python_available false)")
    if not WORKER_PATH.is_file():
        raise RuntimeError(f"Worker não encontrado: {WORKER_PATH}")

    cmd = _build_worker_command(py, input_wav, output_midi, output_json, stem, file_id)
    cwd = str(BASE_DIR.resolve())
    # Garante diretórios de saída existem
    Path(output_midi).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Basic Pitch sync cwd={cwd} stem={stem} file_id={file_id} cmd_repr={[repr(c) for c in cmd[:4]]} ...")
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            timeout=timeout,
        )
        rc = result.returncode
        out_str = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
        err_str = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        out_log = out_str[-6000:] if len(out_str) > 6000 else out_str
        err_log = err_str[-6000:] if len(err_str) > 6000 else err_str
        logger.info(f"Basic Pitch sync rc={rc} stem={stem} stdout_tail={repr(out_log[-500:])} stderr_tail={repr(err_log[-800:])}")
        logger.debug(f"Basic Pitch sync stdout len={len(out_str)} stderr len={len(err_str)}")
        return rc, out_str, err_str
    except subprocess.TimeoutExpired as e:
        logger.error(
            "Basic Pitch sync timeout após %ss stem=%s file_id=%s type=%s repr=%r",
            timeout, stem, file_id, type(e).__name__, e, exc_info=True
        )
        raise TimeoutError(f"Basic Pitch timeout após {timeout}s para {stem}") from e
    except FileNotFoundError as e:
        logger.error(
            "Basic Pitch python/worker não encontrado: type=%s repr=%r str=%s",
            type(e).__name__, e, str(e), exc_info=True
        )
        raise RuntimeError("Basic Pitch não está instalado (python/worker não encontrado)") from e
    except Exception as e:
        logger.error(
            "Falha ao executar Basic Pitch sync: type=%s repr=%r str=%s",
            type(e).__name__, e, str(e), exc_info=True
        )
        raise


async def _run_basic_pitch_async(
    input_wav: Path,
    output_midi: Path,
    output_json: Path,
    stem: str,
    file_id: str,
    timeout: int = BASIC_PITCH_TIMEOUT,
) -> Tuple[int, str, str]:
    """
    Wrapper async que delega para thread via asyncio.to_thread (evita bloqueio do loop).
    Loga event loop e captura tipo/repr da exceção.
    """
    try:
        loop = asyncio.get_running_loop()
        logger.info(f"Event loop Basic Pitch: type={type(loop).__name__} repr={repr(loop)}")
    except Exception as e:
        logger.warning(f"Não foi possível obter event loop: type={type(e).__name__} repr={repr(e)}")

    try:
        return await asyncio.to_thread(_run_basic_pitch_sync, input_wav, output_midi, output_json, stem, file_id, timeout)
    except TimeoutError:
        raise
    except Exception as e:
        logger.error(
            "Falha ao executar Basic Pitch async: type=%s repr=%r str=%s",
            type(e).__name__, e, str(e), exc_info=True
        )
        raise

# ---------------------------------------------------------------------------
# Transcrição de um stem
# ---------------------------------------------------------------------------

async def transcribe_stem_async(
    stem_path: Path,
    file_id: str,
    stem: str,
    timeout: int = BASIC_PITCH_TIMEOUT,
) -> Dict:
    """
    Transcreve um único stem (vocals/bass/other) via worker.
    Valida saída MIDI/JSON e retorna info.
    """
    if stem == "drums":
        raise ValueError("Transcrição de bateria não suportada nesta etapa (drums)")
    if not _validate_stem_for_transcription(stem):
        raise ValueError(f"Stem inválido para transcrição: {stem}. Permitidos: {TRANSCRIBED_STEMS}")
    if not stem_path.is_file():
        raise FileNotFoundError(f"Stem não encontrado: {stem_path}")
    if stem_path.stat().st_size == 0:
        raise ValueError(f"Stem vazio: {stem}")

    # Idempotência: se já existe e válido, retorna
    midi_p = _get_midi_path(file_id, stem)
    json_p = _get_transcription_path(file_id, stem)
    # Verifica se ambos existem e são válidos (parse JSON)
    if midi_p.is_file() and json_p.is_file():
        try:
            if midi_p.stat().st_size > 0 and json_p.stat().st_size > 0:
                with open(json_p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Considera válido se tem events e notes_count
                if "events" in data and "notes_count" in data:
                    logger.info(f"Transcrição já válida para {file_id}/{stem}, idempotência")
                    return {
                        "already_completed": True,
                        "file_id": file_id,
                        "stem": stem,
                        "midi_path": str(midi_p),
                        "json_path": str(json_p),
                        "notes_count": data.get("notes_count", 0),
                        "warning": data.get("warning"),
                    }
        except Exception as e:
            logger.debug(f"Idempotência check falhou para {stem}: {e}, irá retranscrever")

    # Executa worker
    # Garante que diretórios de saída existem
    midi_p.parent.mkdir(parents=True, exist_ok=True)
    json_p.parent.mkdir(parents=True, exist_ok=True)

    # Remove saídas antigas parciais se existirem
    for p in [midi_p, json_p]:
        if p.is_file():
            try:
                p.unlink()
            except:
                pass

    rc, stdout, stderr = await _run_basic_pitch_async(stem_path, midi_p, json_p, stem, file_id, timeout=timeout)

    if rc != 0:
        logger.error(f"Basic Pitch falhou rc={rc} stem={stem} file_id={file_id} stderr[:500]={stderr[:500]}")
        # Tenta capturar erro do JSON de saída ou stdout
        raise RuntimeError(f"Não foi possível transcrever {stem}. (Basic Pitch rc={rc})")

    # Valida saída
    if not midi_p.is_file() or midi_p.stat().st_size == 0:
        logger.error(f"Basic Pitch saída MIDI faltando/vazia stem={stem} file_id={file_id} stdout={stdout[:500]} stderr={stderr[:500]}")
        raise RuntimeError(f"MIDI não gerado para {stem}")

    if not json_p.is_file() or json_p.stat().st_size == 0:
        logger.error(f"Basic Pitch saída JSON faltando/vazia stem={stem} file_id={file_id}")
        raise RuntimeError(f"JSON não gerado para {stem}")

    # Valida JSON
    try:
        with open(json_p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"JSON inválido para {stem}: {e} stdout={stdout[:500]}")
        raise RuntimeError(f"JSON inválido para {stem}: {e}")

    # Valida MIDI via pretty_midi no ambiente Basic Pitch (via subprocess, mas aqui tentamos simples)
    # Como .venv principal pode não ter pretty_midi, tentamos validar tamanho e tentar abrir se possível
    try:
        # Tenta validar via pretty_midi se disponível no .venv principal (pode não estar)
        # Se não estiver, apenas verifica tamanho >0 já feito
        import pretty_midi  # type: ignore
        pm = pretty_midi.PrettyMIDI(str(midi_p))
        notes_in_midi = sum(len(instr.notes) for instr in pm.instruments) if pm.instruments else 0
        # Se notes_count 0 mas MIDI tem 0 notas, é warning, não erro
        # Se notes_count >0 mas MIDI tem 0, log warning
        if data.get("notes_count", 0) > 0 and notes_in_midi == 0:
            logger.warning(f"Divergência: JSON notes_count {data.get('notes_count')} mas MIDI tem 0 notas para {stem}")
    except ImportError:
        # pretty_midi não disponível no .venv principal, ignora validação profunda
        # O worker já validou via pretty_midi no .venv-basicpitch
        pass
    except Exception as e:
        logger.warning(f"Validação MIDI falhou para {stem}: {e}")
        # Não falha se MIDI existe e JSON válido; apenas log

    # Captura warning de notes_count 0
    warning = data.get("warning")
    notes_count = data.get("notes_count", 0)

    # Tenta parse stdout resumo
    stdout_summary = None
    try:
        # Worker imprime JSON summary em stdout na última linha
        for line in stdout.strip().splitlines()[::-1]:
            line = line.strip()
            if line.startswith("{") and "notes_count" in line:
                stdout_summary = json.loads(line)
                break
    except:
        pass

    logger.info(f"Transcrição concluída stem={stem} file_id={file_id} notes={notes_count} warning={warning}")

    return {
        "already_completed": False,
        "file_id": file_id,
        "stem": stem,
        "midi_path": str(midi_p),
        "json_path": str(json_p),
        "notes_count": notes_count,
        "duration": data.get("duration"),
        "warning": warning,
        "stdout_summary": stdout_summary,
    }

# ---------------------------------------------------------------------------
# Transcrição de todos os stems de um file_id
# ---------------------------------------------------------------------------

async def transcribe_all_stems_async(file_id: str, timeout_per_stem: int = BASIC_PITCH_TIMEOUT) -> Dict:
    """
    Transcreve vocals, bass, other sequencialmente (protege CPU/RAM).
    Retorna dict com resultados por stem.
    """
    if not _validate_file_id(file_id):
        raise ValueError("file_id inválido")

    # Verifica se stems existem (pré-condição)
    from backend.audio.stem_separator import are_stems_valid, STEMS_DIR as STEMS_DIR_SEP
    valid, missing = are_stems_valid(file_id)
    if not valid:
        raise FileNotFoundError(f"Separe os instrumentos antes de transcrever. Faltando stems: {missing}")

    # Idempotência global: se todos já válidos, retorna
    valid_trans, missing_trans = are_transcriptions_valid(file_id)
    if valid_trans:
        logger.info(f"Transcrições já válidas para {file_id}, idempotência")
        return {
            "already_completed": True,
            "file_id": file_id,
            "stems": TRANSCRIBED_STEMS,
            "message": "Transcrição já concluída.",
        }

    results = {}
    for stem in TRANSCRIBED_STEMS:
        stem_path = STEMS_DIR_SEP / file_id / f"{stem}.wav"
        if not stem_path.is_file():
            raise FileNotFoundError(f"Stem não encontrado para transcrição: {stem} ({stem_path})")

        logger.info(f"Transcrevendo stem {stem} para file_id {file_id}")
        res = await transcribe_stem_async(stem_path, file_id, stem, timeout=timeout_per_stem)
        results[stem] = res

    return {
        "already_completed": False,
        "file_id": file_id,
        "stems": TRANSCRIBED_STEMS,
        "results": results,
        "message": "Transcrição concluída.",
    }

def transcribe_all_stems_sync(file_id: str, timeout_per_stem: int = BASIC_PITCH_TIMEOUT) -> Dict:
    return asyncio.run(transcribe_all_stems_async(file_id, timeout_per_stem=timeout_per_stem))
