"""
FFprobe integration — análise técnica real de áudio.

Responsabilidades:
- localizar ffprobe de forma segura (sem uso de shell)
- executar ffprobe com timeout e argumentos como lista
- validar existência de stream de áudio
- normalizar metadados para a API
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("uvicorn.error")

FFPROBE_TIMEOUT = 15  # segundos — centralizado para facilitar ajuste
FFPROBE_CMD = "ffprobe"


class FFProbeNotFoundError(RuntimeError):
    pass


class FFProbeTimeoutError(RuntimeError):
    pass


class InvalidAudioError(RuntimeError):
    pass


class FFProbeError(RuntimeError):
    pass


@dataclass
class AudioMetadata:
    file_id: str
    duration: Optional[float]  # segundos, pode ser None se não disponível
    duration_formatted: Optional[str]  # MM:SS ou HH:MM:SS
    format: Optional[str]  # container, ex: mp3, wav, mov...
    codec: Optional[str]  # codec_name, ex: mp3, pcm_s16le, flac
    sample_rate: Optional[int]
    channels: Optional[int]
    bitrate: Optional[int]  # bits por segundo
    size_bytes: Optional[int]


def _find_ffprobe() -> Optional[str]:
    """
    Tenta localizar ffprobe de forma robusta:
    1. shutil.which
    2. winget default location (Gyan.FFmpeg)
    3. locais comuns Windows
    Retorna caminho completo ou apenas "ffprobe" se encontrado via PATH.
    """
    # 1. PATH
    found = shutil.which("ffprobe")
    if found:
        return found
    found_exe = shutil.which("ffprobe.exe")
    if found_exe:
        return found_exe

    # 2. WinGet Gyan.FFmpeg — procura recursiva sob Packages
    try:
        winget_base = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
        if winget_base.is_dir():
            # procura por Gyan.FFmpeg*
            for pkg in winget_base.glob("Gyan.FFmpeg*"):
                # procura ffprobe.exe dentro do pacote
                candidates = list(pkg.rglob("ffprobe.exe"))
                if candidates:
                    # prioriza o mais recente / menor path
                    candidates.sort(key=lambda p: len(str(p)))
                    return str(candidates[0])
                # também tenta ffmpeg*/bin
                for bin_dir in pkg.rglob("bin"):
                    candidate = bin_dir / "ffprobe.exe"
                    if candidate.is_file():
                        return str(candidate)
    except Exception as e:
        logger.debug(f"Erro ao procurar ffprobe em WinGet: {e}")

    # 3. Locais comuns
    common_paths = [
        Path(r"C:\ffmpeg\bin\ffprobe.exe"),
        Path(r"C:\ffmpeg\bin\ffprobe"),
        Path(r"C:\Program Files\ffmpeg\bin\ffprobe.exe"),
        Path(r"C:\tools\ffmpeg\bin\ffprobe.exe"),
        Path(r"C:\ProgramData\chocolatey\bin\ffprobe.exe"),
    ]
    for p in common_paths:
        if p.is_file():
            return str(p)

    return None


def get_ffprobe_path() -> Optional[str]:
    return _find_ffprobe()


def get_ffprobe_version() -> Optional[str]:
    """
    Retorna versão do ffprobe ou None se não encontrado.
    """
    ffprobe = _find_ffprobe()
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [ffprobe, "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # primeira linha: "ffprobe version 9.0.2 ..."
            first_line = (result.stdout or "").splitlines()[0] if result.stdout else ""
            return first_line.strip() or None
        return None
    except Exception as e:
        logger.debug(f"Erro ao obter versão ffprobe: {e}")
        return None


def _format_duration(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    try:
        total = int(float(seconds))
    except Exception:
        return None
    if total < 0:
        return None
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def probe_audio(path: Path, file_id: Optional[str] = None) -> AudioMetadata:
    """
    Executa ffprobe e retorna metadados normalizados.
    Levanta:
      FFProbeNotFoundError, FFProbeTimeoutError, InvalidAudioError, FFProbeError
    """
    ffprobe = _find_ffprobe()
    if not ffprobe:
        raise FFProbeNotFoundError("FFprobe não encontrado no sistema. Verifique se FFmpeg está no PATH.")

    if not path.is_file():
        raise InvalidAudioError("Arquivo não encontrado.")

    # Defesa: garantir que path é arquivo regular dentro de uploads (chamador já validou)
    # Não aceitamos uso de shell, argumentos como lista, path convertido com str(Path)
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration,format_name,size,bit_rate",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,codec_type",
        "-of",
        "json",
        str(path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FFPROBE_TIMEOUT,
        )
    except subprocess.TimeoutExpired as e:
        logger.error(f"ffprobe timeout file_id={file_id} path={path.name}: {e}")
        raise FFProbeTimeoutError("Tempo excedido ao analisar o áudio.")
    except FileNotFoundError as e:
        logger.error(f"ffprobe não encontrado ao executar: {e}")
        raise FFProbeNotFoundError("FFprobe não encontrado.")
    except Exception as e:
        logger.error(f"Erro inesperado ao executar ffprobe: {e}")
        raise FFProbeError("Erro ao executar análise de áudio.")

    if result.returncode != 0:
        # stderr contém detalhes — não expor ao usuário, apenas log
        stderr = (result.stderr or "").strip()[:500]
        logger.warning(f"ffprobe returncode={result.returncode} file_id={file_id} stderr={stderr}")
        raise InvalidAudioError("Arquivo de áudio inválido ou corrompido.")

    stdout = (result.stdout or "").strip()
    if not stdout:
        logger.warning(f"ffprobe stdout vazio file_id={file_id}")
        raise InvalidAudioError("Arquivo de áudio inválido ou corrompido.")

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        logger.error(f"ffprobe JSON inválido file_id={file_id}: {e} stdout={stdout[:500]}")
        raise FFProbeError("Falha ao interpretar resultado da análise.")

    # Valida existência de stream de áudio
    streams = data.get("streams") or []
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio_streams:
        logger.warning(f"ffprobe sem stream de áudio file_id={file_id} streams={streams}")
        raise InvalidAudioError("Arquivo de áudio inválido ou corrompido. Nenhum stream de áudio encontrado.")

    # Pega o primeiro stream de áudio (caso haja múltiplos, loga)
    if len(audio_streams) > 1:
        logger.info(f"ffprobe múltiplos audio streams file_id={file_id} count={len(audio_streams)}")

    audio = audio_streams[0]
    fmt = data.get("format") or {}

    # Extração normalizada com tolerância a ausência
    format_name = fmt.get("format_name")
    if isinstance(format_name, str):
        # ffprobe pode retornar "mov,mp4,m4a,3gp,3g2,mj2" — normalmente o primeiro é o container demuxer
        # Para melhor UX, prefere a extensão original se ela estiver na lista (ex: .m4a -> "m4a" em vez de "mov")
        if "," in format_name:
            parts = [p.strip().lower() for p in format_name.split(",")]
            ext_hint = path.suffix.lower().lstrip(".")  # ex: .m4a -> m4a
            if ext_hint and ext_hint in parts:
                format_name = ext_hint
            else:
                format_name = parts[0].strip()
        else:
            format_name = format_name.lower()
    elif format_name:
        format_name = str(format_name).lower()

    codec = audio.get("codec_name")
    if codec:
        codec = str(codec).lower()

    # duration — tenta format, senão stream
    duration_raw = fmt.get("duration") or audio.get("duration")
    duration: Optional[float] = None
    if duration_raw is not None:
        try:
            duration = float(duration_raw)
            if duration != duration:  # NaN
                duration = None
        except Exception:
            duration = None

    sample_rate: Optional[int] = None
    sr_raw = audio.get("sample_rate")
    if sr_raw is not None:
        try:
            sample_rate = int(sr_raw)
        except Exception:
            sample_rate = None

    channels: Optional[int] = None
    ch_raw = audio.get("channels")
    if ch_raw is not None:
        try:
            channels = int(ch_raw)
        except Exception:
            channels = None

    bitrate: Optional[int] = None
    br_raw = fmt.get("bit_rate") or audio.get("bit_rate")
    if br_raw is not None:
        try:
            bitrate = int(float(br_raw))
        except Exception:
            bitrate = None

    size_bytes: Optional[int] = None
    size_raw = fmt.get("size")
    if size_raw is not None:
        try:
            size_bytes = int(size_raw)
        except Exception:
            size_bytes = None
    # fallback para tamanho no disco
    if size_bytes is None:
        try:
            size_bytes = path.stat().st_size
        except Exception:
            size_bytes = None

    duration_formatted = _format_duration(duration)

    # file_id passado ou derivado do nome
    resolved_file_id = file_id or path.stem

    return AudioMetadata(
        file_id=resolved_file_id,
        duration=duration,
        duration_formatted=duration_formatted,
        format=format_name,
        codec=codec,
        sample_rate=sample_rate,
        channels=channels,
        bitrate=bitrate,
        size_bytes=size_bytes,
    )
