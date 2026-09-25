import uuid
import logging
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Configuração centralizada
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB — alterar aqui para ajustar limite

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"

# Garante que o diretório de uploads existe ao iniciar
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("uvicorn.error")

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
    # Path().name remove qualquer tentativa de path traversal
    # Limita tamanho para evitar header overflow / logs gigantes
    name = Path(filename).name if filename else "arquivo"
    # Remove caracteres de controle e limita a 255 chars
    name = "".join(c for c in name if c.isprintable()).strip()
    if not name:
        name = "arquivo"
    return name[:255]


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
    Mantém compatibilidade: se o arquivo não existir, retorna JSON informativo.
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
    Recebe arquivo de áudio, valida e armazena com nome seguro (UUID).
    """
    try:
        # 1. Validação básica de presença
        if file is None or not file.filename:
            raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

        original_name = _sanitize_original_name(file.filename)
        ext = Path(original_name).suffix.lower()

        # 2. Validação de extensão (não confia apenas no Content-Type)
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

        # 3. Leitura do conteúdo com validação de tamanho e vazio
        # python-multipart já faz buffering; lemos de forma controlada
        content = await file.read()

        if not content or len(content) == 0:
            raise HTTPException(status_code=400, detail="O arquivo está vazio.")

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Arquivo excede o tamanho máximo permitido de {MAX_FILE_SIZE // (1024 * 1024)} MB.",
            )

        # 4. Geração de nome seguro com UUID (sem usar nome do usuário no disco)
        file_id = str(uuid.uuid4())
        stored_filename = f"{file_id}{ext}"
        destination = UPLOAD_DIR / stored_filename

        # Defesa adicional: garantir que o destino permanece dentro de UPLOAD_DIR
        # (mesmo usando UUID, resolve para evitar qualquer manipulação)
        try:
            destination.resolve().relative_to(UPLOAD_DIR.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="Caminho de destino inválido.")

        # Garante diretório existe (caso tenha sido removido em runtime)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        # 5. Escrita no disco
        try:
            destination.write_bytes(content)
        except OSError as e:
            logger.error(f"Erro ao salvar arquivo {file_id}: {e}")
            raise HTTPException(status_code=500, detail="Erro ao salvar o arquivo.")

        # 6. Resposta de sucesso — não expõe path interno, apenas metadados
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
        raise HTTPException(status_code=500, detail="Erro interno ao processar o upload.")
