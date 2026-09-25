# Gerador de Partituras IA

Aplicação FastAPI para análise de áudio e geração de partituras editáveis no MuseScore 4.

## Etapa 2 — Análise técnica com FFprobe

- Upload seguro em `uploads/` com UUID e validação real via FFprobe
- Extração de duração, formato, codec, sample rate, canais, bitrate e tamanho

## Pré-requisitos

- Python 3.11
- FFmpeg / FFprobe disponíveis no `PATH`

Windows (winget):
```powershell
winget install Gyan.FFmpeg
ffmpeg -version
ffprobe -version
```

## Como executar

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
# http://127.0.0.1:8000
# health: http://127.0.0.1:8000/health
```

Endpoints:

- `GET /` — frontend
- `GET /health` — status
- `POST /api/upload` — upload de áudio (multipart, 100 MB max, chunk 1 MB)
- `GET /api/analyze/{file_id}` — metadados técnicos via FFprobe

Formatos: `.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg`

## Segurança

- `pathlib` + UUID, sem `shell=True`, `subprocess` com lista, timeout 15s, sem expor stderr/stack trace
