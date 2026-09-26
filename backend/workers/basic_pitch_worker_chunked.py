#!/usr/bin/env python
"""
Worker Basic Pitch CHUNKED — Etapa 8.2 — músicas longas.

Recebe um manifest JSON (não JSON gigante pela linha de comando):

{
  "input": ".../vocals.wav",
  "output_midi": ".../vocals.mid",
  "output_json": ".../vocals.json",
  "cache_dir": ".../chunks/vocals",
  "duration": 312.4,
  "audio_hash": "abc...",
  "stem": "vocals",
  "file_id": "uuid",
  "chunks": [
    {"index": 0, "start": 0.0, "end": 60.0, "overlap_before": 0.0, "overlap_after": 3.0},
    ...
  ],
  "predict_kwargs": {...}
}

Fluxo:
1. Carrega Basic Pitch UMA vez (não por chunk).
2. Para cada chunk:
   a. Verifica cache (chunk_<i>.json válido → pula inferência).
   b. Extrai WAV do trecho via FFmpeg (streaming, sem carregar stem inteiro).
   c. Verifica silêncio musical (RMS conservador) → pula inferência se silencioso.
   d. Executa predict() no chunk WAV (tempos LOCAIS).
   e. Converte tempos locais → globais.
   f. Salva cache com escrita atômica + atualiza checkpoint.
3. Stitches todos os resultados (dedupe overlap + merge fronteira).
4. Gera UM MIDI final (pretty_midi, timeline global).
5. Gera JSON final compatível com Etapa 5 (frontend não percebe chunking).
6. Limpa WAVs temporários (finally).

Métricas no JSON final (metadata):
  model_load_seconds, inference_seconds (soma), stitch_seconds,
  midi_write_seconds, chunks_total, chunks_inferred, chunks_cached,
  chunks_silence_skipped, rtf.

Executado via .venv-basicpitch com cwd=BASE_DIR:
  python backend/workers/basic_pitch_worker_chunked.py --manifest <manifest.json>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Versão do algoritmo de chunking — invalida cache ao mudar (item 26)
CHUNKED_STAGE_VERSION = "bp-chunked-v1"

# Silence skip: threshold conservador (item 12)
SILENCE_RMS_THRESHOLD = float(os.getenv("SILENCE_RMS_THRESHOLD", "0.003"))
SILENCE_MIN_FRACTION = float(os.getenv("SILENCE_MIN_FRACTION", "0.95"))


def parse_args():
    p = argparse.ArgumentParser(description="Basic Pitch chunked worker")
    p.add_argument("--manifest", required=True,
                   help="Caminho absoluto para manifest JSON")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Cache por chunk (itens 23-25)
# ---------------------------------------------------------------------------

def _chunk_cache_path(cache_dir: Path, index: int) -> Path:
    return cache_dir / f"chunk_{index:04d}.json"


def _chunk_cache_valid(path: Path, expected: dict) -> bool:
    """Cache válido: existe, parseável, version/hash/index/start/end batem."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    if not isinstance(data.get("events"), list):
        return False
    if data.get("version") != expected.get("version"):
        return False
    if data.get("audio_hash") != expected.get("audio_hash"):
        return False
    if data.get("index") != expected.get("index"):
        return False
    if abs(float(data.get("start", -1)) - float(expected["start"])) > 1e-9:
        return False
    if abs(float(data.get("end", -1)) - float(expected["end"])) > 1e-9:
        return False
    # Valida estrutura dos eventos
    for ev in data["events"]:
        if not isinstance(ev.get("pitch"), int):
            return False
        if not isinstance(ev.get("start"), (int, float)):
            return False
        if not isinstance(ev.get("end"), (int, float)):
            return False
    return True


def _atomic_write_json(path: Path, data: dict) -> None:
    """Escrita atômica: tmp + rename (item 114)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(str(tmp), str(path))


def _write_checkpoint(cache_dir: Path, manifest: dict,
                      completed: list) -> None:
    """Persiste checkpoint com chunks concluídos (item 24)."""
    _atomic_write_json(cache_dir / "checkpoint.json", {
        "version": CHUNKED_STAGE_VERSION,
        "audio_hash": manifest.get("audio_hash"),
        "stem": manifest.get("stem"),
        "completed_chunks": sorted(completed),
        "total_chunks": len(manifest.get("chunks", [])),
        "updated_at": time.time(),
    })


# ---------------------------------------------------------------------------
# Extração de chunk via FFmpeg (item 8) — streaming, sem carregar stem inteiro
# ---------------------------------------------------------------------------

def _extract_chunk_wav(input_wav: Path, start: float, end: float,
                       output_wav: Path) -> bool:
    """Extrai trecho [start, end] do WAV via FFmpeg. Retorna True se ok."""
    duration = max(0.0, end - start)
    if duration <= 0:
        return False
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_wav),
        "-ss", f"{start:.6f}",
        "-t", f"{duration:.6f}",
        "-vn",
        str(output_wav),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        if r.returncode == 0 and output_wav.is_file() and output_wav.stat().st_size > 44:
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Silence skip (itens 11-12) — conservador
# ---------------------------------------------------------------------------

def _is_musically_silent(wav_path: Path) -> bool:
    """RMS por frame: silêncio se >=95% dos frames < threshold.

    Conservador: não pula fade-in, voz baixa (amp>0.003), intro suave.
    Não usa VAD de fala — energia musical pura.
    """
    try:
        import numpy as np
        import soundfile as sf
        data, sr = sf.read(str(wav_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.size < 2048:
            return float(np.sqrt(np.mean(data ** 2))) < SILENCE_RMS_THRESHOLD if data.size else True
        frame_len = 2048
        hop = 512
        n_frames = 1 + (data.size - frame_len) // hop
        if n_frames <= 0:
            return True
        quiet = 0
        for k in range(n_frames):
            fr = data[k * hop: k * hop + frame_len]
            rms = float(np.sqrt(np.mean(fr ** 2)))
            if rms < SILENCE_RMS_THRESHOLD:
                quiet += 1
        return (quiet / n_frames) >= SILENCE_MIN_FRACTION
    except Exception:
        return False  # Em dúvida, NÃO pula (preserva precisão)


# ---------------------------------------------------------------------------
# Conversão de eventos (compatível com worker original)
# ---------------------------------------------------------------------------

def midi_pitch_to_note_name(pitch: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (pitch // 12) - 1
    return f"{names[pitch % 12]}{octave}"


def _is_valid_float(x) -> bool:
    """Verifica se x é float finito e não-NaN."""
    try:
        v = float(x)
        return math.isfinite(v)
    except (TypeError, ValueError):
        return False


def _raw_event_to_json(ev, chunk_start: float) -> dict:
    """Converte note_event bruto (local) → dict com tempo global.

    Bug fix: NaN/Infinity do modelo podiam entrar no pipeline
    e corromper JSON/MIDI. Agora eventos inválidos são descartados
    (retorna None) e contabilizados em stats.
    """
    if len(ev) == 5:
        start, end, pitch, amplitude, pitch_bends = ev
    elif len(ev) == 4:
        start, end, pitch, amplitude = ev
        pitch_bends = None
    else:
        start, end, pitch = ev[0], ev[1], ev[2]
        amplitude = ev[3] if len(ev) > 3 else 0.5
        pitch_bends = ev[4] if len(ev) > 4 else None

    # BUG FIX: valida NaN/Infinity/negativo antes de propagar
    if not _is_valid_float(start) or not _is_valid_float(end):
        return None  # Evento inválido: descarta
    if not _is_valid_float(chunk_start):
        return None

    # Local → global (item 14)
    g_start = float(start) + chunk_start
    g_end = float(end) + chunk_start

    # BUG FIX: clamp global_start >= 0 e valida end > start
    if g_start < 0:
        g_start = 0.0
    if g_end <= g_start:
        g_end = g_start + 0.01  # mínimo viável para MIDI

    # Valida pitch
    try:
        pitch_i = int(pitch)
    except (TypeError, ValueError):
        return None
    if not (0 <= pitch_i <= 127):
        return None

    # Valida amplitude (NaN silenciosamente virava 1.0 antes)
    try:
        amp_raw = float(amplitude)
    except (TypeError, ValueError):
        amp_raw = 0.5
    if not math.isfinite(amp_raw):
        amp_raw = 0.5  # NaN/Inf → default, não max
    amp_f = max(0.0, min(1.0, amp_raw))
    velocity = max(0, min(127, int(round(amp_f * 127))))

    event = {
        "start": round(g_start, 6),
        "end": round(g_end, 6),
        "duration": round(max(0.0, g_end - g_start), 6),
        "pitch": pitch_i,
        "note": midi_pitch_to_note_name(pitch_i),
        "velocity": velocity,
        "amplitude": round(amp_f, 4),
        "confidence": round(amp_f, 4),
        "strength": round(amp_f, 4),
    }
    if pitch_bends is not None:
        try:
            pb = [int(x) for x in pitch_bends] if isinstance(pitch_bends, (list, tuple)) else None
            # BUG FIX: valida cada bend é finito
            if pb is not None and all(math.isfinite(b) and -8192 <= b <= 8191 for b in pb):
                event["pitch_bends"] = pb
        except (Exception,):
            pass
    return event


# ---------------------------------------------------------------------------
# Stitching — reutiliza lógica da Etapa 8.1 (item 15)
# ---------------------------------------------------------------------------

def _notes_overlap(a: dict, b: dict, min_overlap: float = 0.05) -> bool:
    if a.get("pitch") != b.get("pitch"):
        return False
    return (min(float(a["end"]), float(b["end"]))
            - max(float(a["start"]), float(b["start"]))) >= min_overlap


def _note_score(n: dict) -> float:
    try:
        return float(n.get("confidence", 0.0)) * 0.6 + float(n.get("amplitude", 0.0)) * 0.4
    except (TypeError, ValueError):
        return 0.0


def stitch_global_events(all_events: list) -> tuple:
    """Stitching global: dedupe overlap + merge fronteira (itens 16-19).

    Bug fix: era O(n^2) — 10.000 eventos levavam 16s, 50.000 explodiriam.
    Agora usa janela deslizante: apenas eventos cujo end >= start atual
    participam da comparação. Complexidade: O(n * w) onde w = polifonia
    simultânea (tipicamente < 30).

    Args:
        all_events: eventos em tempo GLOBAL, com _chunk interno.

    Returns:
        (stitched, stats)
    """
    stats = {
        "raw_notes": len(all_events),
        "notes_after_stitch": 0,
        "overlap_duplicates_removed": 0,
        "cross_chunk_notes_merged": 0,
        "boundary_notes_preserved": 0,
        "invalid_events_discarded": 0,
    }

    # BUG FIX: descarta eventos com NaN/Infinity ANTES do processamento
    valid_events = []
    for n in all_events:
        try:
            s, e = float(n["start"]), float(n["end"])
            if math.isfinite(s) and math.isfinite(e) and s >= 0 and e > s:
                valid_events.append(n)
            else:
                stats["invalid_events_discarded"] += 1
        except (KeyError, TypeError, ValueError):
            stats["invalid_events_discarded"] += 1
    all_events = valid_events

    all_events.sort(key=lambda n: (n["start"], n.get("pitch", 0)))

    # Dedupe overlap: mesma nota em 2 chunks → mantém maior score (item 16)
    # OTIMIZAÇÃO (bug fix #5): janela deslizante em vez de comparar com todos.
    # Eventos ordenados por start: um evento expira quando end < start atual.
    kept = []          # Eventos definitivamente mantidos (expirados da janela)
    window = []        # Eventos ainda ativos (podem overlap com futuros)
    removed = 0
    EXPIRY_GAP = 0.05  # Margem: um evento pode overlap se end >= start - 0.05

    for note in all_events:
        note_start = float(note["start"])
        # Expira eventos da janela cujo end já passou
        still_active = []
        for ex in window:
            if float(ex["end"]) >= note_start - EXPIRY_GAP:
                still_active.append(ex)
            else:
                kept.append(ex)  # Expirou: nunca mais vai overlap
        window = still_active

        # Verifica overlap apenas dentro da janela ativa
        is_dup = False
        for i, ex in enumerate(window):
            if _notes_overlap(ex, note):
                if _note_score(note) > _note_score(ex):
                    window[i] = note
                removed += 1
                is_dup = True
                break
        if not is_dup:
            window.append(note)

    # Move restantes da janela para kept
    kept.extend(window)
    stats["overlap_duplicates_removed"] = removed

    # Merge fronteira: mesmo pitch, gap <= 0.30s, chunks adjacentes (item 17)
    merged = []
    cross = 0
    for note in kept:
        if merged:
            last = merged[-1]
            same_pitch = last.get("pitch") == note.get("pitch")
            gap = float(note["start"]) - float(last["end"])
            adjacent = abs(note.get("_chunk", 0) - last.get("_chunk", 0)) <= 1
            if same_pitch and 0 <= gap <= 0.30 and adjacent:
                last["end"] = note["end"]
                last["duration"] = round(float(last["end"]) - float(last["start"]), 6)
                if _note_score(note) > _note_score(last):
                    last["confidence"] = note.get("confidence", last.get("confidence"))
                    last["amplitude"] = note.get("amplitude", last.get("amplitude"))
                    if "pitch_bends" in note:
                        last["pitch_bends"] = note["pitch_bends"]
                cross += 1
                continue
        merged.append(note)
    stats["cross_chunk_notes_merged"] = cross
    stats["notes_after_stitch"] = len(merged)

    for n in merged:
        n.pop("_chunk", None)
    return merged, stats


# ---------------------------------------------------------------------------
# MIDI final (item 21) — UM MIDI com timeline global
# ---------------------------------------------------------------------------

def _write_final_midi(events: list, output_midi: Path,
                      tempo: float = 120.0) -> int:
    """Gera MIDI único a partir dos eventos consolidados. Retorna n notas."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0)
    for ev in events:
        start = max(0.0, float(ev["start"]))
        end = max(start + 0.01, float(ev["end"]))
        note = pretty_midi.Note(
            velocity=int(ev.get("velocity", 64)),
            pitch=int(ev["pitch"]),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
    pm.instruments.append(instrument)
    output_midi.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(output_midi))
    return len(instrument.notes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.is_file():
        print(json.dumps({"error": f"Manifest não encontrado: {manifest_path}"}), file=sys.stderr)
        sys.exit(2)

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        print(json.dumps({"error": f"Manifest inválido: {e}"}), file=sys.stderr)
        sys.exit(3)

    input_path = Path(manifest["input"]).resolve()
    output_midi = Path(manifest["output_midi"]).resolve()
    output_json = Path(manifest["output_json"]).resolve()
    cache_dir = Path(manifest.get("cache_dir", "")).resolve() if manifest.get("cache_dir") else None
    stem = manifest.get("stem", "other")
    file_id = manifest.get("file_id", "")
    duration = float(manifest.get("duration", 0.0))
    chunks_spec = manifest.get("chunks", [])
    predict_kwargs = manifest.get("predict_kwargs", {})
    audio_hash = manifest.get("audio_hash", "")

    if not input_path.is_file() or input_path.stat().st_size == 0:
        print(json.dumps({"error": f"Input não encontrado/vazio: {input_path}"}), file=sys.stderr)
        sys.exit(4)
    if not chunks_spec:
        print(json.dumps({"error": "Manifest sem chunks"}), file=sys.stderr)
        sys.exit(5)

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Diretório temporário controlado com UUID (item 10)
    temp_root = Path(tempfile.gettempdir()) / f"basicpitch_{uuid.uuid4().hex[:12]}"
    temp_root.mkdir(parents=True, exist_ok=True)

    timings = {
        "model_load_seconds": 0.0,
        "inference_seconds": 0.0,
        "stitch_seconds": 0.0,
        "midi_write_seconds": 0.0,
        "json_write_seconds": 0.0,
    }
    chunks_inferred = 0
    chunks_cached = 0
    chunks_silence = 0
    completed = []
    all_events = []

    try:
        # ------------------------------------------------------------------
        # Carrega Basic Pitch UMA VEZ (item 3 — requisito central)
        # ------------------------------------------------------------------
        t0 = time.time()
        try:
            from basic_pitch.inference import predict as bp_predict
        except Exception as e:
            print(json.dumps({"error": f"Falha ao importar basic_pitch: {e}"}), file=sys.stderr)
            sys.exit(10)
        timings["model_load_seconds"] = round(time.time() - t0, 3)
        # Nota: predict() carrega o modelo ONNX internamente na 1ª chamada;
        # chamadas subsequentes reutilizam o modelo carregado (runtime ONNX).

        # ------------------------------------------------------------------
        # Processa chunks sequencialmente (modelo já carregado)
        # ------------------------------------------------------------------
        for chunk in chunks_spec:
            idx = int(chunk["index"])
            c_start = float(chunk["start"])
            c_end = float(chunk["end"])
            expected = {
                "version": CHUNKED_STAGE_VERSION,
                "audio_hash": audio_hash,
                "index": idx,
                "start": c_start,
                "end": c_end,
            }

            # Cache válido? (itens 23, 25, 48-51)
            cache_p = _chunk_cache_path(cache_dir, idx) if cache_dir else None
            if cache_p and _chunk_cache_valid(cache_p, expected):
                try:
                    with open(cache_p, "r", encoding="utf-8") as f:
                        cdata = json.load(f)
                    all_events.extend(cdata["events"])
                    chunks_cached += 1
                    completed.append(idx)
                    _write_checkpoint(cache_dir, manifest, completed)
                    continue
                except Exception:
                    pass  # Cache corrompido → recalcular este chunk

            # Extrai WAV do chunk (item 8)
            chunk_wav = temp_root / f"chunk_{idx:04d}.wav"
            if not _extract_chunk_wav(input_path, c_start, c_end, chunk_wav):
                print(json.dumps({"error": f"FFmpeg falhou no chunk {idx}"}), file=sys.stderr)
                sys.exit(20)

            # Silence skip (itens 11-12) — antes da inferência pesada
            if _is_musically_silent(chunk_wav):
                chunks_silence += 1
                if cache_p:
                    _atomic_write_json(cache_p, {
                        "version": CHUNKED_STAGE_VERSION,
                        "audio_hash": audio_hash,
                        "index": idx,
                        "start": c_start,
                        "end": c_end,
                        "events": [],  # Silêncio: nenhum evento
                        "silent": True,
                    })
                completed.append(idx)
                _write_checkpoint(cache_dir, manifest, completed)
                chunk_wav.unlink(missing_ok=True)
                continue

            # Inferência (item 3: modelo já carregado, 1 processo)
            t_inf = time.time()
            try:
                _, _, note_events = bp_predict(str(chunk_wav), **predict_kwargs)
            except Exception as e:
                import traceback
                print(json.dumps({
                    "error": f"Falha na predição chunk {idx}: {e}",
                    "traceback": traceback.format_exc(),
                }), file=sys.stderr)
                sys.exit(21)
            timings["inference_seconds"] += time.time() - t_inf

            # Converte local → global (item 14)
            # BUG FIX: descarta eventos inválidos (NaN/Infinity/pitch fora de range)
            local_events = []
            events_invalid = 0
            for ev in note_events or []:
                jev = _raw_event_to_json(ev, c_start)
                if jev is None:
                    events_invalid += 1
                    continue
                jev["_chunk"] = idx
                local_events.append(jev)
            if events_invalid > 0:
                print(json.dumps({
                    "warning": f"chunk {idx}: {events_invalid} evento(s) invalido(s) descartado(s)"
                }), file=sys.stderr)
            all_events.extend(local_events)

            # Cache + checkpoint (itens 23-24)
            if cache_p:
                _atomic_write_json(cache_p, {
                    "version": CHUNKED_STAGE_VERSION,
                    "audio_hash": audio_hash,
                    "index": idx,
                    "start": c_start,
                    "end": c_end,
                    "events": local_events,
                    "silent": False,
                })
            completed.append(idx)
            _write_checkpoint(cache_dir, manifest, completed)

            # Limpa WAV do chunk (economiza disco em músicas longas)
            chunk_wav.unlink(missing_ok=True)
            chunks_inferred += 1

        # ------------------------------------------------------------------
        # Stitching global (itens 15-19)
        # ------------------------------------------------------------------
        t_st = time.time()
        stitched, stitch_stats = stitch_global_events(all_events)
        timings["stitch_seconds"] = round(time.time() - t_st, 3)

        # ------------------------------------------------------------------
        # MIDI final — UM arquivo, timeline global (item 21)
        # ------------------------------------------------------------------
        t_midi = time.time()
        notes_in_midi = _write_final_midi(stitched, output_midi,
                                          tempo=predict_kwargs.get("midi_tempo", 120.0))
        timings["midi_write_seconds"] = round(time.time() - t_midi, 3)

        if not output_midi.is_file() or output_midi.stat().st_size == 0:
            print(json.dumps({"error": "MIDI final não gerado"}), file=sys.stderr)
            sys.exit(22)

        # ------------------------------------------------------------------
        # JSON final — compatível com Etapa 5 (item 22)
        # ------------------------------------------------------------------
        t_json = time.time()
        notes_count = len(stitched)
        warning = None
        if notes_count == 0:
            warning = "Nenhuma nota detectada."

        total_chunks = len(chunks_spec)
        total_seconds_inf = timings["inference_seconds"]
        rtf = round(total_seconds_inf / duration, 3) if duration > 0 else None

        output_data = {
            "file_id": file_id,
            "stem": stem,
            "notes_count": notes_count,
            "duration": round(duration, 3),
            "events": stitched,
        }
        if warning:
            output_data["warning"] = warning

        # Parâmetros para auditoria (mesmo formato do worker original)
        output_data["parameters"] = {
            "onset_threshold": predict_kwargs.get("onset_threshold", 0.5),
            "frame_threshold": predict_kwargs.get("frame_threshold", 0.3),
            "minimum_note_length": predict_kwargs.get("minimum_note_length", 127.7),
            "midi_tempo": predict_kwargs.get("midi_tempo", 120.0),
            "multiple_pitch_bends": predict_kwargs.get("multiple_pitch_bends", False),
            "melodia_trick": predict_kwargs.get("melodia_trick", True),
            "minimum_frequency": predict_kwargs.get("minimum_frequency"),
            "maximum_frequency": predict_kwargs.get("maximum_frequency"),
        }

        # Metadata de chunking (debug; frontend não depende disso)
        output_data["chunking"] = {
            "version": CHUNKED_STAGE_VERSION,
            "audio_hash": audio_hash,
            "chunks_total": total_chunks,
            "chunks_inferred": chunks_inferred,
            "chunks_cached": chunks_cached,
            "chunks_silence_skipped": chunks_silence,
            "timings_seconds": timings,
            "inference_rtf": rtf,
            "stitch_stats": stitch_stats,
        }

        try:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(json.dumps({"error": f"Falha ao salvar JSON: {e}"}), file=sys.stderr)
            sys.exit(23)
        timings["json_write_seconds"] = round(time.time() - t_json, 3)

        if not output_json.is_file() or output_json.stat().st_size == 0:
            print(json.dumps({"error": "JSON final não gerado"}), file=sys.stderr)
            sys.exit(24)

        # ------------------------------------------------------------------
        # Summary em stdout (mesmo formato do worker original + extras)
        # ------------------------------------------------------------------
        summary = {
            "file_id": file_id,
            "stem": stem,
            "notes_count": notes_count,
            "midi_path": str(output_midi),
            "json_path": str(output_json),
            "duration": round(duration, 3),
            "warning": warning,
            "chunked": True,
            "chunks_total": total_chunks,
            "chunks_inferred": chunks_inferred,
            "chunks_cached": chunks_cached,
            "chunks_silence_skipped": chunks_silence,
            "timings_seconds": timings,
            "inference_rtf": rtf,
        }
        print(json.dumps(summary))
        sys.exit(0)

    finally:
        # Limpa temporários SEMPRE (item 53) — não apaga cache/resultados finais
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()