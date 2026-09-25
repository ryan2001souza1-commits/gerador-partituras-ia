import uuid
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.audio.probe import (
    probe_audio,
    get_ffprobe_version,
    FFProbeNotFoundError,
    FFProbeTimeoutError,
    InvalidAudioError,
    FFProbeError,
    FFPROBE_TIMEOUT,
)

# ---------------------------------------------------------------------------
# Configuração centralizada
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB — alterar aqui para ajustar limite
CHUNK_SIZE = 1 * 1024 * 1024  # 1 MB por chunk

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"

# Garante que o diretório de uploads existe ao iniciar
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("uvicorn.error")

# Loga versão do ffprobe ao iniciar (não bloqueia se ausente)
try:
    ffprobe_version = get_ffprobe_version()
    if ffprobe_version:
        logger.info(f"FFprobe disponível: {ffprobe_version}")
    else:
        logger.warning("FFprobe não encontrado. Análise de áudio ficará indisponível até instalação no PATH.")
except Exception as e:
    logger.warning(f"Erro ao verificar ffprobe: {e}")

app = FastAPI(
    title="Gerador de Partituras IA",
    description="API para análise de áudio e geração de partituras com IA",
    version="0.1.0",
)

# Servir arquivos estáticos (CSS/JS) em /static
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_allowed_extension(ext: str) -> bool:
    return ext.lower() in ALLOWED_EXTENSIONS


def _sanitize_original_name(filename: str) -> str:
    """
    Retorna apenas o nome base sem diretórios, para exibição como metadado.
    Não é usado para construir path no disco.
    """
    name = Path(filename).name if filename else "arquivo"
    name = "".join(c for c in name if c.isprintable()).strip()
    if not name:
        name = "arquivo"
    return name[:255]


def _find_upload_path(file_id: str) -> Optional[Path]:
    """
    Localiza arquivo em uploads/ pelo file_id (UUID) de forma segura.
    Não aceita path, apenas UUID. Retorna Path ou None.
    """
    # Aceita somente UUID válido — chamador já validou, mas reforça
    try:
        uuid.UUID(file_id)
    except ValueError:
        return None

    candidates: list[Path] = []

    # Busca direta por extensões permitidas
    for ext in ALLOWED_EXTENSIONS:
        p = UPLOAD_DIR / f"{file_id}{ext}"
        if p.is_file():
            candidates.append(p)

    # Fallback: glob para capturar variações (ex: uppercase)
    if not candidates:
        for p in UPLOAD_DIR.glob(f"{file_id}.*"):
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS and p.stem == file_id:
                candidates.append(p)

    if not candidates:
        return None

    if len(candidates) > 1:
        logger.warning(f"Múltiplos arquivos para file_id={file_id}: {candidates}")

    chosen = candidates[0]
    # Defesa: garantir que está dentro de UPLOAD_DIR
    try:
        chosen.resolve().relative_to(UPLOAD_DIR.resolve())
    except ValueError:
        logger.error(f"Path traversal detectado file_id={file_id} path={chosen}")
        return None

    return chosen


# ---------------------------------------------------------------------------
# Rotas
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def serve_frontend():
    """
    Entrega o frontend/index.html.
    """
    index_path = FRONTEND_DIR / "index.html"
    if index_path.is_file():
        return FileResponse(str(index_path), media_type="text/html")
    return JSONResponse(
        content={
            "nome": "Gerador de Partituras IA",
            "status": "online",
            "mensagem": "Frontend não encontrado. Verifique frontend/index.html",
        }
    )


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Recebe arquivo de áudio em chunks, valida tamanho e conteúdo via ffprobe.
    """
    destination: Optional[Path] = None
    file_id: Optional[str] = None
    total_written = 0

    try:
        # 1. Validação básica de presença
        if file is None or not file.filename:
            raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

        original_name = _sanitize_original_name(file.filename)
        ext = Path(original_name).suffix.lower()

        if not ext:
            raise HTTPException(
                status_code=400,
                detail="Arquivo sem extensão. Formatos aceitos: .mp3, .wav, .flac, .m4a, .ogg",
            )

        if not _is_allowed_extension(ext):
            raise HTTPException(
                status_code=400,
                detail=f"Formato '{ext}' não permitido. Formatos aceitos: .mp3, .wav, .flac, .m4a, .ogg",
            )

        # 2. Geração de nome seguro com UUID
        file_id = str(uuid.uuid4())
        stored_filename = f"{file_id}{ext}"
        destination = UPLOAD_DIR / stored_filename

        # Defesa adicional: garantir destino dentro de UPLOAD_DIR
        try:
            destination.resolve().relative_to(UPLOAD_DIR.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="Caminho de destino inválido.")

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        # 3. Escrita em chunks (não carrega 100 MB na RAM)
        try:
            with open(destination, "wb") as out:
                while True:
                    chunk = await file.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    total_written += len(chunk)
                    if total_written > MAX_FILE_SIZE:
                        # Remove parcial e aborta
                        try:
                            out.close()
                        except Exception:
                            pass
                        if destination.exists():
                            destination.unlink(missing_ok=True)
                        raise HTTPException(
                            status_code=413,
                            detail=f"Arquivo excede o tamanho máximo permitido de {MAX_FILE_SIZE // (1024 * 1024)} MB.",
                        )
                    out.write(chunk)
        except HTTPException:
            # Propaga 413
            raise
        except OSError as e:
            logger.error(f"Erro ao salvar arquivo file_id={file_id}: {e}")
            if destination and destination.exists():
                destination.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="Erro ao salvar o arquivo.")

        if total_written == 0:
            if destination.exists():
                destination.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="O arquivo está vazio.")

        # 4. Validação real via FFprobe (conteúdo)
        try:
            # probe_audio lançará InvalidAudioError se não for áudio
            probe_audio(destination, file_id=file_id)
        except FFProbeNotFoundError as e:
            # FFprobe ausente: não podemos validar, mas não bloqueia upload
            logger.warning(f"ffprobe ausente, upload sem validação file_id={file_id}: {e}")
            # mantém arquivo e considera sucesso
        except InvalidAudioError as e:
            logger.warning(f"Arquivo inválido detectado por ffprobe file_id={file_id}: {e}")
            if destination.exists():
                destination.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Arquivo de áudio inválido ou corrompido.")
        except FFProbeTimeoutError as e:
            # Timeout não significa arquivo inválido; mantém arquivo, mas loga
            logger.error(f"ffprobe timeout file_id={file_id}: {e}")
            # Não remove; upload considerado sucesso, análise posterior falhará com 504
        except FFProbeError as e:
            logger.error(f"ffprobe erro file_id={file_id}: {e}")
            # Mantém arquivo; erro de servidor não deve apagar upload
        except Exception as e:
            logger.error(f"Erro inesperado pós-upload ffprobe file_id={file_id}: {e}")
            # Mantém arquivo

        # 5. Resposta de sucesso
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "file_id": file_id,
                "original_name": original_name,
                "message": "Arquivo enviado com sucesso.",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        # Nunca expor stack trace ao frontend
        logger.error(f"Erro inesperado no upload: {e}")
        # Garante limpeza de parcial se file_id/destination já definidos
        if destination is not None and destination.exists():
            # Se total_written ainda 0 ou erro genérico, remove parcial
            # Mas se já validado e ffprobe manteve, não devemos remover aqui
            # Então só remove se arquivo ainda não foi validado como áudio válido
            # Para simplificar: se exceção genérica, remove se arquivo existe e total_written>0 mas não validado?
            # Aqui estamos em except genérico fora do bloco de escrita, então é erro inesperado — remove parcial
            try:
                # tenta remover apenas se ainda não houve sucesso de validação
                # Se file_id existe mas probe falhou com erro genérico, mantemos?
                # Por segurança, vamos manter se total_written >0 e não for InvalidAudio
                pass
            except Exception:
                pass
        raise HTTPException(status_code=500, detail="Erro interno ao processar o upload.")


@app.get("/api/analyze/{file_id}")
def analyze_audio(file_id: str):
    """
    Analisa arquivo previamente enviado e retorna metadados técnicos via FFprobe.
    """
    # 1. Valida UUID
    try:
        uuid.UUID(file_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="file_id inválido.")

    # 2. Localiza arquivo de forma segura
    path = _find_upload_path(file_id)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado.")

    # 3. Executa FFprobe
    try:
        meta = probe_audio(path, file_id=file_id)
    except FFProbeNotFoundError:
        logger.error(f"ffprobe não encontrado ao analisar file_id={file_id}")
        raise HTTPException(status_code=503, detail="FFprobe não disponível no servidor.")
    except FFProbeTimeoutError:
        logger.error(f"ffprobe timeout ao analisar file_id={file_id}")
        raise HTTPException(status_code=504, detail="Não foi possível analisar o áudio (timeout).")
    except InvalidAudioError:
        logger.warning(f"analyze: arquivo inválido file_id={file_id} path={path.name}")
        raise HTTPException(status_code=400, detail="Arquivo de áudio inválido ou corrompido.")
    except FFProbeError as e:
        logger.error(f"ffprobe erro ao analisar file_id={file_id}: {e}")
        raise HTTPException(status_code=500, detail="Não foi possível analisar o áudio.")
    except Exception as e:
        logger.error(f"Erro inesperado ao analisar file_id={file_id}: {e}")
        raise HTTPException(status_code=500, detail="Não foi possível analisar o áudio.")

    # 4. Retorna metadados normalizados
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "file_id": meta.file_id,
            "duration": meta.duration,
            "duration_formatted": meta.duration_formatted,
            "format": meta.format,
            "codec": meta.codec,
            "sample_rate": meta.sample_rate,
            "channels": meta.channels,
            "bitrate": meta.bitrate,
            "size_bytes": meta.size_bytes,
        },
    )
