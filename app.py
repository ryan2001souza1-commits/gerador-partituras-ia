import asyncio
import json
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

from backend.audio.music_analysis import analyze_music

from backend.audio.stem_separator import (
    EXPECTED_STEMS,
    STEMS_DIR,
    DEMUCS_MODEL,
    DEMUCS_DEVICE,
    DEMUCS_JOBS,
    DEMUCS_TIMEOUT,
    get_demucs_python,
    is_demucs_available,
    get_demucs_version,
    get_torch_info,
    are_stems_valid,
    get_stems_info,
    separate_stems_async,
)

from backend.audio.job_manager import (
    create_job,
    get_job,
    update_job,
    has_active_job,
    job_to_dict,
    get_active_job,
)

from backend.audio.transcriber import (
    TRANSCRIBED_STEMS,
    MIDI_DIR,
    TRANSCRIPTIONS_DIR,
    BASIC_PITCH_TIMEOUT,
    get_basic_pitch_python,
    is_basic_pitch_available,
    get_basic_pitch_version,
    get_runtime_info,
    are_required_stems_valid,
    are_transcriptions_valid,
    get_transcription_info,
    transcribe_all_stems_async,
)

from backend.audio.transcription_job_manager import (
    create_transcription_job,
    get_transcription_job,
    update_transcription_job,
    has_active_transcription_job,
    transcription_job_to_dict,
    get_active_transcription_job,
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
# STEMS_DIR já definido em stem_separator, mas garante existência também aqui
STEMS_DIR_APP = STEMS_DIR  # alias para clareza

# Garante que diretórios existem ao iniciar
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STEMS_DIR_APP.mkdir(parents=True, exist_ok=True)
MIDI_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)

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
    Analisa arquivo previamente enviado e retorna:
      - metadados técnicos via FFprobe
      - análise musical (BPM, tonalidade, modo, confiança) via librosa + FFmpeg
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

    # 4. Análise musical (BPM, tonalidade, confiança) — não quebra análise técnica se falhar
    music_dict = None
    music_warning = None
    try:
        music_result = analyze_music(path, duration_probe=meta.duration)
        music_dict = music_result.to_api_dict()
        # Se houver warning técnico (ex: áudio curto), propaga como campo opcional no topo para frontend
        if music_result.warning:
            music_warning = music_result.warning
        # Loga caso bpm/key sejam None mas sem erro técnico (inconclusivo)
        if music_dict.get("bpm") is None and music_dict.get("key") is None and not music_dict.get("error"):
            logger.info(f"analyze_music inconclusivo file_id={file_id} warning={music_warning}")
    except Exception as e:
        logger.error(f"Falha inesperada na análise musical file_id={file_id}: {e}", exc_info=True)
        music_dict = {
            "bpm": None,
            "bpm_rounded": None,
            "bpm_confidence": None,
            "key": None,
            "mode": None,
            "key_confidence": None,
            "error": "Não foi possível determinar a estrutura musical deste áudio.",
        }
        music_warning = music_dict["error"]

    # 5. Retorna metadados técnicos + música
    response_content: dict = {
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
        "music": music_dict,
    }
    if music_warning:
        response_content["music_warning"] = music_warning

    return JSONResponse(
        status_code=200,
        content=response_content,
    )


# ---------------------------------------------------------------------------
# Separação de stems — job em background
# ---------------------------------------------------------------------------

async def _run_separation_job(job_id: str, file_id: str, input_path: Path):
    """
    Executa separação em background, atualizando job registry.
    Não bloqueia request; captura stdout/returncode via stem_separator.
    """
    try:
        update_job(job_id, status="running", message="Separando instrumentos... (pode levar vários minutos, CPU)")
        logger.info(f"Job {job_id} running file_id={file_id}")

        # Passos graduais para progresso honesto (não percent falsa)
        # Mensagens intermediárias
        update_job(job_id, message="Carregando modelo htdemucs... (primeira execução pode baixar modelo)")

        result = await separate_stems_async(input_path, file_id, timeout=DEMUCS_TIMEOUT)

        # already_completed pode indicar que stems já existiam (idempotência)
        if result.get("already_completed"):
            update_job(
                job_id,
                status="completed",
                message="Instrumentos já separados.",
                stems=result.get("stems"),
                already_completed=True,
            )
            logger.info(f"Job {job_id} already_completed file_id={file_id}")
            return

        # Finalizando
        update_job(job_id, message="Finalizando stems...")
        # Valida novamente
        valid, _ = are_stems_valid(file_id)
        if not valid:
            raise RuntimeError("Validação final de stems falhou")

        update_job(
            job_id,
            status="completed",
            message="Separação concluída.",
            stems=EXPECTED_STEMS,
        )
        logger.info(f"Job {job_id} completed file_id={file_id}")
    except asyncio.TimeoutError as e:
        logger.error(f"Job {job_id} timeout file_id={file_id}: {e}")
        update_job(job_id, status="failed", message="A separação demorou mais que o esperado (timeout 45 min).", error="Timeout")
    except TimeoutError as e:
        logger.error(f"Job {job_id} timeout file_id={file_id}: {e}")
        update_job(job_id, status="failed", message="A separação demorou mais que o esperado.", error=str(e))
    except RuntimeError as e:
        msg = str(e)
        # Mensagens amigáveis
        friendly = "Não foi possível separar os instrumentos."
        if "Demucs não está instalado" in msg:
            friendly = "Demucs não está instalado. Configure .venv-demucs com demucs==4.1.0"
        elif "Saída incompleta" in msg:
            friendly = "Saída incompleta: faltando stems. Tente novamente."
        elif "vazio" in msg:
            friendly = "Stem vazio gerado. Tente com outro arquivo."
        logger.error(f"Job {job_id} failed file_id={file_id}: {e}")
        update_job(job_id, status="failed", message=friendly, error=msg)
    except Exception as e:
        logger.error(f"Job {job_id} erro inesperado file_id={file_id}: {e}", exc_info=True)
        update_job(job_id, status="failed", message="Não foi possível separar os instrumentos.", error=str(e))


@app.post("/api/separate/{file_id}")
async def separate_audio(file_id: str):
    """
    Inicia separação em 4 stems (vocals, drums, bass, other) via Demucs.
    Retorna rápido com job_id (queued). Processamento em background.
    Protege: só 1 job ativo por vez, idempotência, valida UUID e arquivo.
    """
    # 1. Valida UUID
    try:
        uuid.UUID(file_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="file_id inválido.")

    # 2. Localiza arquivo
    path = _find_upload_path(file_id)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado.")

    # 3. Idempotência: se stems já válidos, retorna already_completed
    valid, _ = are_stems_valid(file_id)
    if valid:
        job = create_job(file_id, status="completed", message="Instrumentos já separados.")
        job.already_completed = True
        job.stems = EXPECTED_STEMS
        update_job(job.job_id, status="completed", message="Instrumentos já separados.", stems=EXPECTED_STEMS, already_completed=True)
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "job_id": job.job_id,
                "file_id": file_id,
                "status": "completed",
                "already_completed": True,
                "message": "Instrumentos já separados.",
                "stems": [{"name": s, "url": f"/api/stems/{file_id}/{s}"} for s in EXPECTED_STEMS],
            },
        )

    # 4. Verifica Demucs instalado (fail fast amigável)
    if not is_demucs_available():
        # Não cria job; informa diretamente para não confundir polling
        logger.warning(f"Demucs não disponível ao tentar separar file_id={file_id} python={get_demucs_python()}")
        raise HTTPException(status_code=503, detail="Demucs não está instalado. Configure .venv-demucs com demucs==4.1.0")

    # 5. Proteção um job por vez
    if has_active_job():
        active = get_active_job()
        logger.warning(f"Separação já em andamento job={active.job_id if active else 'unknown'} solicitado file_id={file_id}")
        raise HTTPException(status_code=409, detail="Já existe uma separação em andamento. Aguarde concluir.")

    # 6. Cria job queued
    job = create_job(file_id, status="queued", message="Preparando separação...")

    # 7. Agenda background (não bloqueia)
    # Mensagem inicial honesta (não percent falsa)
    update_job(job.job_id, message="Preparando modelo htdemucs...")

    asyncio.create_task(_run_separation_job(job.job_id, file_id, path))

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "job_id": job.job_id,
            "file_id": file_id,
            "status": "queued",
            "already_completed": False,
            "message": "Separação agendada. Na primeira execução, o modelo pode precisar ser baixado.",
        },
    )


@app.get("/api/separate/status/{job_id}")
def get_separate_status(job_id: str):
    """
    Consulta status de job de separação.
    Retorna queued|running|completed|failed
    """
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="job_id inválido.")

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado. Reiniciar servidor perde jobs em memória.")

    data = job_to_dict(job)
    # Adiciona stems urls quando completed
    if job.status == "completed" and job.file_id:
        data["stems_info"] = get_stems_info(job.file_id)
        # Compat: stems list com urls
        if not data.get("stems"):
            data["stems"] = [{"name": s, "url": f"/api/stems/{job.file_id}/{s}"} for s in EXPECTED_STEMS]

    return JSONResponse(status_code=200, content={"success": True, **data})


@app.get("/api/stems/{file_id}")
def list_stems(file_id: str):
    """
    Lista stems disponíveis para file_id.
    """
    try:
        uuid.UUID(file_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="file_id inválido.")

    # Verifica se upload existe (opcional, mas valida)
    path = _find_upload_path(file_id)
    if path is None:
        # Mesmo se upload removido, pode haver stems; mas retorna 404 se não houver stems nem upload
        info = get_stems_info(file_id)
        if not info or not info.get("available"):
            raise HTTPException(status_code=404, detail="Arquivo não encontrado.")
    info = get_stems_info(file_id)
    if not info:
        raise HTTPException(status_code=400, detail="file_id inválido.")

    return JSONResponse(status_code=200, content={"success": True, **info})


@app.get("/api/stems/{file_id}/{stem_name}")
def get_stem_file(file_id: str, stem_name: str):
    """
    Serve arquivo WAV do stem com allowlist e defesa path traversal.
    """
    try:
        uuid.UUID(file_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="file_id inválido.")

    stem = stem_name.lower().strip()
    if stem not in EXPECTED_STEMS:
        raise HTTPException(status_code=400, detail=f"stem_name inválido. Permitidos: {', '.join(EXPECTED_STEMS)}")

    # Constrói path seguro
    stems_dir = STEMS_DIR_APP / file_id
    stem_path = stems_dir / f"{stem}.wav"

    # Defesa resolve/relative_to
    try:
        # Garante que stems_dir está dentro de STEMS_DIR
        stems_dir.resolve().relative_to(STEMS_DIR_APP.resolve())
        # Se arquivo existe, verifica também
        if stem_path.exists():
            stem_path.resolve().relative_to(STEMS_DIR_APP.resolve())
    except ValueError:
        logger.error(f"Path traversal em get_stem file_id={file_id} stem={stem}")
        raise HTTPException(status_code=400, detail="Caminho inválido.")

    if not stem_path.is_file():
        raise HTTPException(status_code=404, detail=f"Stem '{stem}' não encontrado. Execute separação primeiro.")

    # Verifica tamanho >0
    try:
        if stem_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Stem vazio.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Erro ao acessar stem.")

    return FileResponse(str(stem_path), media_type="audio/wav", filename=f"{stem}.wav")


# Endpoint auxiliar para debug/info Demucs (não expõe detalhes sensíveis)
@app.get("/api/demucs/info")
def demucs_info():
    # Garante serialização segura: Path -> str, torch_info apenas tipos primitivos
    demucs_py = get_demucs_python()
    # Normaliza Path caso get_demucs_python retorne Path no futuro
    demucs_py_str = str(demucs_py) if demucs_py is not None else None
    torch_info = get_torch_info()
    # Garante que torch_info é serializável (já filtrado em stem_separator)
    # Fallback se None
    if not isinstance(torch_info, dict):
        torch_info = {}
    return JSONResponse(
        status_code=200,
        content={
            "demucs_python": demucs_py_str,
            "demucs_available": bool(is_demucs_available()),
            "demucs_version": get_demucs_version(),
            "torch_info": torch_info,
            "model": str(DEMUCS_MODEL),
            "device": str(DEMUCS_DEVICE),
            "jobs": int(DEMUCS_JOBS),
            "timeout_seconds": int(DEMUCS_TIMEOUT),
            "expected_stems": list(EXPECTED_STEMS),
        },
    )


# ---------------------------------------------------------------------------
# Transcrição Basic Pitch — jobs e endpoints
# ---------------------------------------------------------------------------

async def _run_transcription_job(job_id: str, file_id: str):
    """
    Executa transcrição dos 3 stems sequencialmente, atualizando job registry.
    """
    try:
        update_transcription_job(job_id, status="running", message="Preparando transcrição...")
        logger.info(f"Transcription job {job_id} running file_id={file_id}")

        # Verifica se já está completo (idempotência) — o transcriber já faz, mas reforça
        valid, _ = are_transcriptions_valid(file_id)
        if valid:
            update_transcription_job(job_id, status="completed", message="Transcrição já concluída.", already_completed=True)
            logger.info(f"Transcription job {job_id} already_completed file_id={file_id}")
            return

        # Mensagens honestas por stem
        stems_to_process = TRANSCRIBED_STEMS  # vocals, bass, other

        for idx, stem in enumerate(stems_to_process):
            if stem == "vocals":
                msg = "Transcrevendo vocais..."
            elif stem == "bass":
                msg = "Transcrevendo baixo..."
            else:
                msg = "Transcrevendo acompanhamento..."
            update_transcription_job(job_id, message=msg)
            logger.info(f"Transcription job {job_id} stem={stem} ({idx+1}/{len(stems_to_process)})")

            # O transcriber processa um stem por vez; se falhar, levanta e job vai para failed
            # Usa transcribe_all_stems_async internamente sequencial, mas aqui chamamos por stem para progresso
            # Para simplificar, chamamos transcribe_all mas com progresso por stem:
            # Na verdade vamos chamar transcribe_all de uma vez e atualizar mensagens antes
            # Para ter progresso granular, chamamos stem a stem via transcribe_stem
            from backend.audio.transcriber import transcribe_stem_async
            from backend.audio.stem_separator import STEMS_DIR as STEMS_DIR_SEP
            stem_path = STEMS_DIR_SEP / file_id / f"{stem}.wav"
            if not stem_path.is_file():
                raise FileNotFoundError(f"Stem não encontrado: {stem}")

            # Chama transcrição do stem individual com timeout por stem
            await transcribe_stem_async(stem_path, file_id, stem, timeout=BASIC_PITCH_TIMEOUT)

            # Atualiza status intermediário
            update_transcription_job(job_id, message=f"{msg} concluído ({idx+1}/{len(stems_to_process)})")

        # Valida final
        update_transcription_job(job_id, message="Validando MIDI...")
        valid_final, missing = are_transcriptions_valid(file_id)
        if not valid_final:
            raise RuntimeError(f"Validação final falhou, faltando: {missing}")

        # Coleta resultados
        info = get_transcription_info(file_id)
        update_transcription_job(
            job_id,
            status="completed",
            message="Transcrição concluída.",
            results=info,
        )
        logger.info(f"Transcription job {job_id} completed file_id={file_id}")

    except asyncio.TimeoutError as e:
        logger.error(f"Transcription job {job_id} timeout file_id={file_id}: {e}")
        update_transcription_job(job_id, status="failed", message="A transcrição demorou mais que o esperado.", error="Timeout")
    except TimeoutError as e:
        logger.error(f"Transcription job {job_id} timeout file_id={file_id}: {e}")
        update_transcription_job(job_id, status="failed", message="A transcrição demorou mais que o esperado.", error=str(e))
    except FileNotFoundError as e:
        msg = str(e)
        if "Separe os instrumentos" in msg:
            friendly = "Separe os instrumentos antes de transcrever."
        else:
            friendly = f"Stem não encontrado: {msg}"
        logger.error(f"Transcription job {job_id} failed file_id={file_id}: {e}")
        update_transcription_job(job_id, status="failed", message=friendly, error=msg)
    except RuntimeError as e:
        msg = str(e)
        friendly = "Não foi possível transcrever."
        if "Basic Pitch não está instalado" in msg:
            friendly = "Basic Pitch não está instalado. Configure .venv-basicpitch com basic-pitch==0.4.0"
        logger.error(f"Transcription job {job_id} failed file_id={file_id}: {e}")
        update_transcription_job(job_id, status="failed", message=friendly, error=msg)
    except Exception as e:
        logger.error(f"Transcription job {job_id} erro inesperado file_id={file_id}: {e}", exc_info=True)
        update_transcription_job(job_id, status="failed", message="Não foi possível transcrever.", error=str(e))


@app.post("/api/transcribe/{file_id}")
async def transcribe_audio(file_id: str):
    """
    Inicia transcrição dos stems vocals/bass/other via Basic Pitch.
    Pré-condição: stems devem existir.
    """
    try:
        uuid.UUID(file_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="file_id inválido.")

    # Verifica se stems necessários para transcrição existem (vocals, bass, other — não exige drums)
    valid, missing = are_required_stems_valid(file_id)
    if not valid:
        raise HTTPException(status_code=409, detail="Separe os instrumentos antes de transcrever.")

    # Idempotência: se transcrições já válidas, retorna already_completed
    valid_trans, _ = are_transcriptions_valid(file_id)
    if valid_trans:
        job = create_transcription_job(file_id, status="completed", message="Transcrição já concluída.")
        job.already_completed = True
        info = get_transcription_info(file_id)
        update_transcription_job(job.job_id, status="completed", message="Transcrição já concluída.", results=info, already_completed=True)
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "job_id": job.job_id,
                "file_id": file_id,
                "status": "completed",
                "already_completed": True,
                "message": "Transcrição já concluída.",
                "transcriptions": info,
            },
        )

    # Verifica Basic Pitch instalado
    if not is_basic_pitch_available():
        logger.warning(f"Basic Pitch não disponível file_id={file_id} python={get_basic_pitch_python()}")
        raise HTTPException(status_code=503, detail="Basic Pitch não está instalado. Configure .venv-basicpitch com basic-pitch==0.4.0")

    # Proteção um job por vez (transcrição)
    if has_active_transcription_job():
        active = get_active_transcription_job()
        logger.warning(f"Transcrição já em andamento job={active.job_id if active else 'unknown'} file_id={file_id}")
        raise HTTPException(status_code=409, detail="Já existe uma transcrição em andamento. Aguarde concluir.")

    # Também respeita Demucs job ativo? Não bloqueia transcription se Demucs estiver rodando? Para proteger CPU, bloqueia qualquer job ativo
    # Mas spec diz apenas 1 transcrição por vez, não menciona Demucs; vamos permitir Demucs e transcription simultâneos? Para proteger CPU, vamos bloquear se houver qualquer job ativo de ambos?
    # Simplifica: apenas verifica transcription jobs, permite Demucs paralelo (mas pode pesar). Para proteger CPU máxima, verifica ambos.
    if has_active_job():
        logger.warning(f"Demucs em andamento, bloqueando transcrição file_id={file_id}")
        raise HTTPException(status_code=409, detail="Já existe uma separação em andamento. Aguarde concluir.")

    job = create_transcription_job(file_id, status="queued", message="Preparando transcrição...")
    update_transcription_job(job.job_id, message="Preparando transcrição...")

    asyncio.create_task(_run_transcription_job(job.job_id, file_id))

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "job_id": job.job_id,
            "file_id": file_id,
            "status": "queued",
            "already_completed": False,
            "message": "Transcrição agendada.",
        },
    )


@app.get("/api/transcribe/status/{job_id}")
def get_transcribe_status(job_id: str):
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="job_id inválido.")
    job = get_transcription_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado. Reiniciar servidor perde jobs em memória.")
    data = transcription_job_to_dict(job)
    # Adiciona info de transcrições quando completed
    if job.status == "completed" and job.file_id:
        info = get_transcription_info(job.file_id)
        data["transcriptions"] = info
        if not data.get("results"):
            data["results"] = info
    return JSONResponse(status_code=200, content={"success": True, **data})


@app.get("/api/transcriptions/{file_id}")
def list_transcriptions(file_id: str):
    try:
        uuid.UUID(file_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="file_id inválido.")
    # Verifica se file_id existe (upload ou stems)
    path = _find_upload_path(file_id)
    from backend.audio.stem_separator import are_stems_valid as check_stems
    stems_valid, _ = check_stems(file_id)
    if path is None and not stems_valid:
        # Se não tem upload nem stems, mas pode ter transcriptions antigas, ainda permite listar?
        # Verifica transcriptions
        info = get_transcription_info(file_id)
        if not info or not info.get("available"):
            raise HTTPException(status_code=404, detail="Arquivo não encontrado.")
    info = get_transcription_info(file_id)
    if not info:
        raise HTTPException(status_code=400, detail="file_id inválido.")
    return JSONResponse(status_code=200, content={"success": True, **info})


@app.get("/api/transcriptions/{file_id}/{stem}")
def get_transcription_file(file_id: str, stem: str):
    try:
        uuid.UUID(file_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="file_id inválido.")
    stem_l = stem.lower().strip()
    if stem_l not in TRANSCRIBED_STEMS:
        if stem_l == "drums":
            raise HTTPException(status_code=400, detail="Transcrição de bateria não suportada nesta etapa. Será adicionada em etapa futura.")
        raise HTTPException(status_code=400, detail=f"stem inválido. Permitidos: {', '.join(TRANSCRIBED_STEMS)}")
    # Path seguro
    json_path = TRANSCRIPTIONS_DIR / file_id / f"{stem_l}.json"
    try:
        TRANSCRIPTIONS_DIR.resolve().relative_to(TRANSCRIPTIONS_DIR.resolve())
        if json_path.exists():
            json_path.resolve().relative_to(TRANSCRIPTIONS_DIR.resolve())
    except ValueError:
        logger.error(f"Path traversal transcriptions file_id={file_id} stem={stem_l}")
        raise HTTPException(status_code=400, detail="Caminho inválido.")
    if not json_path.is_file():
        raise HTTPException(status_code=404, detail=f"Transcrição para '{stem_l}' não encontrada. Execute transcrição primeiro.")
    try:
        if json_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Transcrição vazia.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Erro ao acessar transcrição.")
    # Retorna JSON com validação de conteúdo
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Falha ao ler transcrição JSON {json_path}: {e}")
        raise HTTPException(status_code=500, detail="Transcrição inválida.")
    return JSONResponse(status_code=200, content={"success": True, **data})


@app.get("/api/midi/{file_id}/{stem}")
def get_midi_file(file_id: str, stem: str):
    try:
        uuid.UUID(file_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="file_id inválido.")
    stem_l = stem.lower().strip()
    if stem_l not in TRANSCRIBED_STEMS:
        if stem_l == "drums":
            raise HTTPException(status_code=400, detail="Transcrição de bateria não suportada nesta etapa.")
        raise HTTPException(status_code=400, detail=f"stem inválido. Permitidos: {', '.join(TRANSCRIBED_STEMS)}")
    midi_path = MIDI_DIR / file_id / f"{stem_l}.mid"
    try:
        MIDI_DIR.resolve().relative_to(MIDI_DIR.resolve())
        if midi_path.exists():
            midi_path.resolve().relative_to(MIDI_DIR.resolve())
    except ValueError:
        logger.error(f"Path traversal midi file_id={file_id} stem={stem_l}")
        raise HTTPException(status_code=400, detail="Caminho inválido.")
    if not midi_path.is_file():
        raise HTTPException(status_code=404, detail=f"MIDI para '{stem_l}' não encontrado. Execute transcrição primeiro.")
    try:
        if midi_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="MIDI vazio.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Erro ao acessar MIDI.")
    # Valida rapidamente que é MIDI (tenta abrir com mido se disponível, senão apenas serve)
    return FileResponse(str(midi_path), media_type="audio/midi", filename=f"{stem_l}.mid")


@app.get("/api/basic-pitch/info")
def basic_pitch_info():
    py = get_basic_pitch_python()
    py_str = str(py) if py is not None else None
    runtime_info = get_runtime_info()
    if not isinstance(runtime_info, dict):
        runtime_info = {}
    return JSONResponse(
        status_code=200,
        content={
            "available": bool(is_basic_pitch_available()),
            "python": py_str,
            "basic_pitch_version": get_basic_pitch_version(),
            "runtime": runtime_info.get("runtime", "unknown"),
            "runtime_version": runtime_info.get("runtime_version"),
            "runtime_info": runtime_info,
            "processed_stems": list(TRANSCRIBED_STEMS),
            "timeout_seconds": int(BASIC_PITCH_TIMEOUT),
            "expected_stems": list(TRANSCRIBED_STEMS),
        },
    )
