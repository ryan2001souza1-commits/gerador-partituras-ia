# Gerador de Partituras IA

Aplicação FastAPI para análise de áudio e geração de partituras editáveis no MuseScore 4.

## Etapa 3 — Análise musical básica (BPM e tonalidade)

- Upload seguro em `uploads/` com UUID e validação real via FFprobe
- Extração de duração, formato, codec, sample rate, canais, bitrate e tamanho
- **Novidade Etapa 3:**
  - Detecção de BPM / andamento via librosa (HPSS percussivo + onset + beat tracking)
  - Detecção de tonalidade (12 tônicas × maior/menor = 24 possibilidades) via chroma + perfis Krumhansl-Kessler
  - Estimativas de confiança (`bpm_confidence`, `key_confidence` 0–1) como heurística de estabilidade/ambiguidade
  - Nomes de tonalidade em português no frontend (Dó, Ré, Mi ... + maior/menor)
  - Avisos para áudio curto (<3s), silêncio, harmonia ambígua ou ritmo instável

> Resultados são automáticos e heurísticos — podem exigir revisão musical. Baixa confiança indica necessidade de validação humana. Precisão não é 100%.

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
- `GET /api/analyze/{file_id}` — metadados técnicos (FFprobe) + análise musical (`music: {bpm, bpm_rounded, bpm_confidence, key, mode, key_confidence}`) — decodificação via FFmpeg → WAV mono 22050 → librosa

Formatos: `.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg`

## Arquitetura Etapa 3

- `backend/audio/music_analysis.py` — decodificação FFmpeg segura, `MusicAnalysisResult`, `analyze_music()`; BPM via `librosa.effects.hpss` + `onset_strength` + `beat_track` com confiança por regularidade de intervalos; tonalidade via `chroma_cqt` + correlação Pearson com perfis Krumhansl major/minor transpostos; confiança por comparação melhor vs segundo melhor.
- Decodificação recomendada: arquivo original → FFmpeg → WAV temporário PCM mono 22050 (nome aleatório `tempfile`, removido em `try/finally`, sem `shell=True`).

## Segurança

- `pathlib` + UUID, sem `shell=True`, `subprocess` com lista, timeout (FFprobe 15s, FFmpeg 30s), sem expor stderr/stack trace
- Upload em chunks 1 MB, limite 100 MB, validação real de áudio
- Temporários limpos mesmo em exceção
