# Gerador de Partituras IA

Aplicação FastAPI para análise de áudio e geração de partituras editáveis no MuseScore 4.

## Etapa 4 — Separação de stems / instrumentos base (Demucs)

- Upload seguro em `uploads/` com UUID e validação real via FFprobe
- Extração de duração, formato, codec, sample rate, canais, bitrate e tamanho
- Detecção de BPM / andamento via librosa (HPSS percussivo + onset + beat tracking)
- Detecção de tonalidade (12 tônicas × maior/menor = 24 possibilidades) via chroma + perfis Krumhansl-Kessler
- **Novidade Etapa 4:**
  - Separação real em 4 stems com **Demucs `htdemucs`** (CPU): `vocals.wav`, `drums.wav`, `bass.wav`, `other.wav`
  - Pipeline seguro: arquivo original → Demucs (`-d cpu -j 1 -n htdemucs -o <tmp>`) → validação FFprobe → normalização para `stems/<file_id>/`
  - Job assíncrono (`POST /api/separate/{file_id}` → `queued`, polling `GET /api/separate/status/{job_id}`) — protege CPU/RAM (1 job por vez, timeout 45 min, sem bloquear request)
  - Endpoints seguros para listar e reproduzir stems: `GET /api/stems/{file_id}` e `GET /api/stems/{file_id}/{stem}` (allowlist, `resolve().relative_to()`)
  - Frontend com botão “Separar instrumentos”, estados honestos (Preparando → Carregando modelo → Separando → Finalizando) e players `<audio controls>` para cada stem
  - Idempotência (reuso de `stems/<file_id>/` já válidos), limpeza de temporários (`try/finally`), sem expor stderr/traceback

> Resultados são automáticos e heurísticos — podem exigir revisão musical. Na primeira separação, Demucs pode baixar o modelo (~80–100 MB). Processamento em CPU pode levar vários minutos dependendo da duração.

## Pré-requisitos

- Python 3.11
- FFmpeg / FFprobe disponíveis no `PATH`

Windows (winget):
```powershell
winget install Gyan.FFmpeg
ffmpeg -version
ffprobe -version
```

## Como executar (backend principal)

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
# http://127.0.0.1:8000
# health: http://127.0.0.1:8000/health
```

## Setup Demucs (ambiente separado — protege .venv principal)

Demucs `4.1.0` é pesado (PyTorch). Para não quebrar o `.venv` principal, as dependências ficam em um ambiente separado:

```powershell
# Criar ambiente dedicado (Windows)
py -3.11 -m venv .venv-demucs

.\.venv-demucs\Scripts\python.exe -m pip install --upgrade pip

.\.venv-demucs\Scripts\python.exe -m pip install -r requirements-demucs.txt
# requirements-demucs.txt contém: demucs==4.1.0  (e numpy==2.4.6 se necessário para torch)

# Validar
.\.venv-demucs\Scripts\python.exe -c "import demucs; print('demucs ok')"
.\.venv-demucs\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Esperado: torch 2.x+cpu, cuda_available False (CPU nesta etapa, sem wheel CUDA)
```

O backend principal resolve automaticamente o Python do Demucs nesta ordem:
1. variável `DEMUCS_PYTHON` (se definida)
2. `.venv-demucs/Scripts/python.exe` (Windows)
3. `.venv-demucs/bin/python` (Linux/Mac)
4. fallback `sys.executable` (apenas para testes)

Não é necessário configurar CUDA nesta etapa (`-d cpu` sempre).

Endpoints:

- `GET /` — frontend
- `GET /health` — status
- `POST /api/upload` — upload de áudio (multipart, 100 MB max, chunk 1 MB)
- `GET /api/analyze/{file_id}` — técnico (FFprobe) + musical (`music: {bpm,...}`)
- `POST /api/separate/{file_id}` — inicia separação (retorna `job_id`, `queued`; 409 se já há job ativo; 503 se Demucs não instalado)
- `GET /api/separate/status/{job_id}` — estado `queued|running|completed|failed` (memória; reiniciar servidor perde jobs — documentado)
- `GET /api/stems/{file_id}` — lista stems disponíveis
- `GET /api/stems/{file_id}/{stem}` — serve `vocals|drums|bass|other` WAV (allowlist, sem path livre)
- `GET /api/demucs/info` — debug: versão demucs/torch, device, modelo

Formatos: `.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg`

## Arquitetura Etapa 4

- `backend/audio/stem_separator.py` — `get_demucs_python()`, `are_stems_valid()`, `_build_demucs_command()`, `_run_demucs_async()` (asyncio.create_subprocess_exec), `separate_stems_async()` com timeout 45 min, validação FFprobe/duração, normalização para `stems/<file_id>/` e limpeza `try/finally`/`shutil.rmtree`
- `backend/audio/job_manager.py` — registry em memória `JOBS={}` com `threading.Lock`, `has_active_job()` (1 por vez), `create_job/update_job/job_to_dict`, limitação documentada (perda em restart)
- `app.py` — endpoints async `POST /api/separate/{file_id}` → `asyncio.create_task(_run_separation_job)`, `GET /api/separate/status`, `GET /api/stems/...` com validação UUID/allowlist/`resolve().relative_to()`, sem `shell=True`
- Frontend: botão “Separar instrumentos” após análise, polling 2 s, estados honestos sem porcentagem falsa, players `<audio controls>` seguros

## Segurança

- `pathlib` + UUID, sem `shell=True`, `subprocess` lista, timeout (FFprobe 15s, FFmpeg 30s, Demucs 45 min), sem expor stderr/stack trace
- Upload chunks 1 MB, limite 100 MB, validação real via FFprobe
- Stems: `file_id` UUID, `stem_name` allowlist, `resolve().relative_to(STEMS_DIR)`, nunca aceitar `../../` nem filename do usuário
- Temporários limpos mesmo em exceção/timeout; `stems/` e `.venv-demucs/` ignorados no Git
