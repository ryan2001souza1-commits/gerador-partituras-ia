# Gerador de Partituras IA

Aplicação FastAPI para análise de áudio e geração de partituras editáveis no MuseScore 4.

## Etapa 5 — Transcrição dos stems para notas e MIDI (Basic Pitch)

- Upload seguro em `uploads/` com UUID e validação real via FFprobe
- Extração de duração, formato, codec, sample rate, canais, bitrate e tamanho
- Detecção de BPM / andamento via librosa (HPSS percussivo + onset + beat tracking)
- Detecção de tonalidade (12 tônicas × maior/menor = 24 possibilidades) via chroma + perfis Krumhansl-Kessler
- Separação real em 4 stems com **Demucs `htdemucs`** (CPU): `vocals.wav`, `drums.wav`, `bass.wav`, `other.wav`
- **Novidade Etapa 5:**
  - Transcrição dos stems melódicos **vocals, bass, other** para eventos de notas e **MIDI** com **Basic Pitch `0.4.0`** (ONNX runtime, CPU)
  - Worker isolado `backend/workers/basic_pitch_worker.py` executado via `.venv-basicpitch` (sem importar no backend principal)
  - Saída: `midi/<file_id>/{vocals,bass,other}.mid` + `transcriptions/<file_id>/{vocals,bass,other}.json` com `start,end,duration,pitch,note,velocity,amplitude/confidence, pitch_bends`
  - Job assíncrono `POST /api/transcribe/{file_id}` → `queued` → `GET /api/transcribe/status/{job_id}` (1 transcrição por vez, 3 stems sequenciais, timeout 20 min/stem, sem bloquear)
  - Endpoints seguros `GET /api/midi/{file_id}/{stem}` e `GET /api/transcriptions/{file_id}/{stem}` (allowlist, `resolve().relative_to()`)
  - Frontend com botão “Transcrever para notas” após stems, polling, card com `notes_count` e `Baixar MIDI` (bateria: “será adicionada em etapa futura”)
  - Validação MIDI (pretty_midi/mido, tamanho >0, notes), idempotência, limpeza temporários, sem quantização ainda (timing bruto)

> Resultados são automáticos e heurísticos — podem exigir revisão musical. Na primeira transcrição, o modelo ONNX é carregado; CPU pode levar ~12s por stem (3s de áudio → 3 notas C4/E4/G4 detectadas). Bateria não é processada com algoritmo melódico.

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

```powershell
py -3.11 -m venv .venv-demucs
.\.venv-demucs\Scripts\python.exe -m pip install --upgrade pip
.\.venv-demucs\Scripts\python.exe -m pip install -r requirements-demucs.txt
# requirements-demucs.txt: demucs==4.1.0  + numpy==2.4.6 se necessário
.\.venv-demucs\Scripts\python.exe -c "import demucs; print('demucs ok')"
.\.venv-demucs\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Esperado: torch 2.x+cpu, cuda False
```

O backend resolve o Python do Demucs nesta ordem: `DEMUCS_PYTHON` env → `.venv-demucs/Scripts/python.exe` → `bin/python` → fallback `sys.executable`. Não usar CUDA (`-d cpu`).

## Setup Basic Pitch (ambiente separado — protege .venv e .venv-demucs)

```powershell
py -3.11 -m venv .venv-basicpitch
.\.venv-basicpitch\Scripts\python.exe -m pip install --upgrade pip
.\.venv-basicpitch\Scripts\python.exe -m pip install -r requirements-basicpitch.txt
# requirements-basicpitch.txt: basic-pitch==0.4.0  (pip resolve onnxruntime/tensorflow)
# Para ONNX (recomendado CPU Windows, evita DLL TensorFlow):
# pip install onnxruntime já incluído via pip; se necessário: pip install onnxruntime==1.30.0

.\.venv-basicpitch\Scripts\python.exe -c "import basic_pitch; print('Basic Pitch OK')"
.\.venv-basicpitch\Scripts\python.exe -c "import onnxruntime; print(onnxruntime.__version__)"
# ou, se TF disponível: import tensorflow as tf; print(tf.__version__)
# Esperado: Basic Pitch OK, onnx 1.30.0 (ou tf 2.15.0), sem CUDA
```

O backend resolve o Python do Basic Pitch nesta ordem: `BASIC_PITCH_PYTHON` env → `.venv-basicpitch/Scripts/python.exe` → `bin/python` → fallback `sys.executable`. O worker `backend/workers/basic_pitch_worker.py` é executado isoladamente via esse Python, com `shell=False`, caminhos absolutos, `cwd=BASE_DIR`, `timeout 20min/stem`, `asyncio.to_thread` (evita limitação `WindowsSelectorEventLoop` do Uvicorn).

Endpoints:

- `GET /` — frontend
- `GET /health` — status
- `POST /api/upload` — upload
- `GET /api/analyze/{file_id}` — técnico + musical
- `POST /api/separate/{file_id}` — Demucs 4 stems
- `GET /api/separate/status/{job_id}` — Demucs status
- `GET /api/stems/{file_id}` / `GET /api/stems/{file_id}/{stem}` — stems
- `GET /api/demucs/info` — Demucs debug
- `POST /api/transcribe/{file_id}` — inicia transcrição vocals/bass/other (409 se sem stems, 503 se Basic Pitch ausente, 409 se já há job)
- `GET /api/transcribe/status/{job_id}` — status transcrição `queued|running|completed|failed`
- `GET /api/transcriptions/{file_id}` — lista transcrições + notes_count
- `GET /api/transcriptions/{file_id}/{stem}` — JSON eventos
- `GET /api/midi/{file_id}/{stem}` — MIDI `audio/midi` (vocals/bass/other, drums 400)
- `GET /api/basic-pitch/info` — Basic Pitch debug

Formatos: `.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg`

## Arquitetura Etapa 5

- `backend/workers/basic_pitch_worker.py` — `argparse` (`--input`, `--output-midi`, `--output-json`, `--stem`, `--file-id`, thresholds, frequências), `midi_pitch_to_note_name`, `predict()` com `minimum_frequency`/`maximum_frequency` por stem (vocals 70-2000, bass 30-500, other amplo), defaults `onset 0.5/frame 0.3/min_note 127.7/tempo 120`, salva MIDI via `pretty_midi`, serializa `note_events` → JSON `start,end,duration,pitch,note,velocity,amplitude/confidence/strength,pitch_bends`, valida MIDI/JSON, imprime `{"notes_count":...}` em `stdout`
- `backend/audio/transcriber.py` — `TRANSCRIBED_STEMS=[vocals,bass,other]`, `STEM_FREQ_RANGES`, `BASIC_PITCH_DEFAULTS`, `MIDI_DIR`, `TRANSCRIPTIONS_DIR`, `BASIC_PITCH_TIMEOUT=1200`, `get_basic_pitch_python()`, `is_basic_pitch_available()`, `get_runtime_info()` (onnx/tf), `are_required_stems_valid()`, `are_transcriptions_valid()`, `_build_worker_command()` (absolutos, lista, `repr` log), `_run_basic_pitch_sync()` (`subprocess.run` `PIPE` `cwd` `timeout` `errors="replace"` últimos 6000), `_run_basic_pitch_async()` (`to_thread` + log `event loop`), `transcribe_stem_async()` (valida, idempotência, worker, valida MIDI/JSON, `notes_count 0` → warning), `transcribe_all_stems_async()` sequencial 3 stems
- `backend/audio/transcription_job_manager.py` — `JOBS` separado de Demucs, `has_active_transcription_job()` (1 por vez), `create/update/get`, documentado perda em restart
- `app.py` — `POST /api/transcribe/{file_id}` (valida UUID, `are_required_stems_valid` → 409, `are_transcriptions_valid` idempotência, `is_basic_pitch_available` → 503, `has_active_transcription_job`/`has_active_job` → 409, `create_transcription_job` + `asyncio.create_task(_run_transcription_job)`), `_run_transcription_job` atualiza `Transcrevendo vocais/bass/acompanhamento` sequencial, `GET` endpoints com `UUID`/`allowlist`/`resolve().relative_to()`, `shell=False`, `audio/midi`/`application/json`
- Frontend: após `stems-info` mostra `transcribe-section` (“Transcrever para notas” roxo #7c3aed), polling 2s, `transcription-info` grid com `notes_count` + `Baixar MIDI` + `Ver eventos` (vocals/bass/other) e `drums` com aviso futuro, sem bloquear upload/análise

## Segurança

- `pathlib` + UUID, sem `shell=True`, `subprocess` lista, timeout (FFprobe 15s, FFmpeg 30s, Demucs 45min, Basic Pitch 20min/stem), sem expor stderr/stack trace
- Upload chunks 1 MB, limite 100 MB, validação real via FFprobe
- Stems: `file_id` UUID, `stem_name` allowlist, `resolve().relative_to(STEMS_DIR)`; Transcrições/MIDI: `TRANSCRIBED_STEMS` allowlist, `resolve().relative_to(MIDI_DIR/TRANSCRIPTIONS_DIR)`, nunca `../../` nem filename original
- Temporários limpos mesmo em exceção/timeout; `stems/`, `midi/`, `transcriptions/`, `.venv-*` ignorados no Git
