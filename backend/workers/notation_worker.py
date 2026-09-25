#!/usr/bin/env python
"""Worker isolado de notação — executado via .venv-notation (music21).

Pipeline: eventos Basic Pitch (JSON) -> limpeza -> grade de tempo ->
quantização -> figuras rítmicas -> pausas -> compassos -> armadura ->
andamento -> partes -> MusicXML -> MuseScore 4.

NÃO importar este módulo no processo FastAPI principal.
Uso:
  .venv-notation/Scripts/python.exe backend/workers/notation_worker.py
    --file-id <uuid> --transcriptions-dir <abs> --output-musicxml <abs>
    --output-model <abs> --tempo 89 --time-signature 4/4
    --quantization 1/16 --key-mode auto [--key G# --mode major
    --key-confidence 0.28] [--beat-offset 0.12]

Decisões documentadas:
- Grade binária limpa; sem detecção automática de tercinas/tuplets
  ("Tuplets automáticos serão aprimorados futuramente").
- pitch bends preservados no MIDI da Etapa 5; MusicXML usa pitch discreto.
- Uma única clave por parte (sem trocas automáticas no meio da parte).
- Bateria não incluída.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Permite `from backend.notation.score_utils import ...` com cwd=BASE_DIR.
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.notation.score_utils import (  # noqa: E402
    PART_NAMES,
    beats_per_measure,
    check_parts_nonempty,
    choose_clef_other,
    clean_events,
    grid_step_beats,
    group_chords_other,
    merge_same_pitch,
    quantize_note,
    resolve_key_signature,
    resolve_monophonic_overlaps,
    seconds_to_beats,
    config_key,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Notation worker (music21)")
    p.add_argument("--file-id", required=True)
    p.add_argument("--transcriptions-dir", required=True)
    p.add_argument("--output-musicxml", required=True)
    p.add_argument("--output-model", required=True)
    p.add_argument("--tempo", required=True, type=float)
    p.add_argument("--time-signature", required=True)
    p.add_argument("--quantization", required=True)
    p.add_argument("--key-mode", required=True)
    p.add_argument("--key", default=None)
    p.add_argument("--mode", default=None)
    p.add_argument("--key-confidence", default=None, type=float)
    p.add_argument("--beat-offset", default=0.0, type=float)
    return p.parse_args()


def load_events(trans_dir: Path, stem: str) -> dict:
    path = trans_dir / f"{stem}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    events = data.get("events", [])
    if not isinstance(events, list):
        raise ValueError(f"JSON de {stem} sem lista 'events'")
    return {"events": events}


def process_monophonic(
    events: list, tempo: float, beat_offset: float, grid: float
) -> tuple[list, dict]:
    """Limpeza -> beats -> merge -> quantização -> overlaps. Retorna (notes, stats)."""
    stats: dict = {}
    raw = len(events)
    cleaned, clean_stats = clean_events(events)
    stats.update({f"clean_{k}": v for k, v in clean_stats.items()})

    notes_beats: list = []
    pickups = 0
    for ev in cleaned:
        start = float(ev["start"])
        end = float(ev["end"])
        if start < beat_offset:
            pickups += 1
        sb = seconds_to_beats(start, tempo, beat_offset)
        eb = seconds_to_beats(end, tempo, beat_offset)
        if eb <= sb:
            eb = sb + grid
        notes_beats.append({
            "start": sb, "end": eb,
            "pitch": int(ev["pitch"]),
            "velocity": int(ev.get("velocity", 64)),
        })
    stats["pickups_clamped"] = pickups

    merged, merged_count = merge_same_pitch(notes_beats)
    stats["merged"] = merged_count

    quantized: list = []
    for n in merged:
        qs, qe = quantize_note(n["start"], n["end"], grid)
        quantized.append({
            "start": qs, "end": qe,
            "pitch": int(n["pitch"]),
            "velocity": int(n.get("velocity", 64)),
        })
    resolved, ov_stats = resolve_monophonic_overlaps(quantized, grid)
    stats.update(ov_stats)
    stats["raw"] = raw
    stats["cleaned"] = len(cleaned)
    stats["quantized"] = len(resolved)
    return resolved, stats


def process_other(
    events: list, tempo: float, beat_offset: float, grid: float
) -> tuple[list, list, dict]:
    """Retorna (items chord/note, voices assignment, stats)."""
    stats: dict = {}
    raw = len(events)
    cleaned, clean_stats = clean_events(events)
    stats.update({f"clean_{k}": v for k, v in clean_stats.items()})

    notes_beats: list = []
    pickups = 0
    for ev in cleaned:
        start = float(ev["start"])
        end = float(ev["end"])
        if start < beat_offset:
            pickups += 1
        sb = seconds_to_beats(start, tempo, beat_offset)
        eb = seconds_to_beats(end, tempo, beat_offset)
        if eb <= sb:
            eb = sb + grid
        notes_beats.append({
            "start": sb, "end": eb,
            "pitch": int(ev["pitch"]),
            "velocity": int(ev.get("velocity", 64)),
        })
    stats["pickups_clamped"] = pickups

    merged, merged_count = merge_same_pitch(notes_beats)
    stats["merged"] = merged_count

    quantized: list = []
    for n in merged:
        qs, qe = quantize_note(n["start"], n["end"], grid)
        quantized.append({
            "start": qs, "end": qe,
            "pitch": int(n["pitch"]),
            "velocity": int(n.get("velocity", 64)),
        })
    items, chord_stats = group_chords_other(quantized)
    stats.update(chord_stats)

    # Distribui itens em voices monofônicas (greedy, determinístico).
    voices: list[list] = []
    for item in sorted(items, key=lambda i: (i["start"], i["end"])):
        placed = False
        for voice in voices:
            if float(item["start"]) >= float(voice[-1]["end"]) - 1e-9:
                voice.append(item)
                placed = True
                break
        if not placed:
            if len(voices) < 4:
                voices.append([item])
            else:
                # Estratégia determinística: anexa à voice com fim mais antigo.
                voices.sort(key=lambda v: v[-1]["end"])
                voices[0].append(item)
                stats["voice_overflow"] = stats.get("voice_overflow", 0) + 1
    stats["voices"] = len(voices)
    stats["raw"] = raw
    stats["cleaned"] = len(cleaned)
    stats["quantized"] = len(items)
    return items, voices, stats


def build_score(args, processed: dict, key_norm, mode_norm, warnings: list):
    from music21 import clef as m21clef
    from music21 import instrument as m21instrument
    from music21 import key as m21key
    from music21 import metadata as m21metadata
    from music21 import meter as m21meter
    from music21 import note as m21note
    from music21 import chord as m21chord
    from music21 import stream as m21stream
    from music21 import tempo as m21tempo

    score = m21stream.Score()
    md = m21metadata.Metadata()
    md.title = "Gerador de Partituras IA"
    score.metadata = md
    metronome = m21tempo.MetronomeMark(number=int(args.tempo))

    if key_norm and mode_norm:
        try:
            sharps = m21key.Key(key_norm, mode_norm).sharps
        except Exception:
            sharps = 0
            warnings.append(f"Armadura {key_norm} {mode_norm} inválida; usando neutra.")
    else:
        sharps = 0

    mlen = beats_per_measure(args.time_signature)
    clef_map = {
        "vocals": "treble",
        "bass": "bass",
        "other": processed["other_clef"],
    }
    instr_map = {
        "vocals": m21instrument.Vocalist,
        "bass": m21instrument.ElectricBass,
        "other": m21instrument.Piano,
    }

    for idx, stem in enumerate(("vocals", "bass", "other")):
        part = m21stream.Part()
        pname, pabbr = PART_NAMES[stem]
        part.partName = pname
        part.partAbbreviation = pabbr
        if idx == 0:
            # Andamento explícito na primeira parte (exporta p/ MusicXML;
            # MetronomeMark direto no Score não é exportado).
            part.insert(0, metronome)
        try:
            part.insert(0, instr_map[stem]())
        except Exception:
            pass
        clef_name = clef_map[stem]
        part.insert(0, m21clef.TrebleClef() if clef_name == "treble" else m21clef.BassClef())
        part.insert(0, m21meter.TimeSignature(args.time_signature))
        part.insert(0, m21key.KeySignature(sharps))

        if stem in ("vocals", "bass"):
            _fill_monophonic(part, processed[stem]["notes"], mlen,
                             m21note, m21stream)
        else:
            _fill_other(part, processed[stem]["voices"], mlen,
                        m21note, m21chord, m21stream)

        try:
            part.makeMeasures(inPlace=True)
        except Exception as e:
            warnings.append(f"makeMeasures falhou em {stem}: {e}")
        try:
            part.makeTies(inPlace=True)
        except Exception as e:
            warnings.append(f"makeTies falhou em {stem}: {e}")
        _renumber_voices(part)
        score.insert(0, part)

    return score


def _renumber_voices(part) -> None:
    """Numera Voices 1..N dentro de cada Measure (pós-makeMeasures/makeTies).

    makeMeasures/makeTies criam containers Voice com id default 0, exportado
    como <voice>0</voice> — inválido (vozes MusicXML começam em 1) e o
    MuseScore ignora essas notas. Renumerar por compasso é seguro: voices
    em MusicXML são por medida.
    """
    from music21 import stream as m21stream
    for m in part.getElementsByClass(m21stream.Measure):
        for i, v in enumerate(list(m.voices)):
            v.id = i + 1


def _make_note(m21note, pitch: int, qlen: float, velocity: int):
    n = m21note.Note()
    n.pitch.midi = int(pitch)
    n.quarterLength = float(qlen)
    try:
        n.volume.velocity = max(1, min(127, int(velocity)))
    except Exception:
        pass
    return n


def _fill_monophonic(part, notes: list, mlen: float, m21note, m21stream) -> None:
    """Preenche parte monofônica com notas + pausas reais + trailing rest."""
    cursor = 0.0
    for n in sorted(notes, key=lambda x: (x["start"], x["end"])):
        start = round(float(n["start"]), 6)
        end = round(float(n["end"]), 6)
        if start > cursor + 1e-9:
            r = m21note.Rest()
            r.quarterLength = round(start - cursor, 6)
            part.insert(cursor, r)
            cursor = start
        dur = round(end - start, 6)
        if dur <= 0:
            continue
        part.insert(start, _make_note(m21note, int(n["pitch"]), dur, int(n.get("velocity", 64))))
        cursor = max(cursor, end)
    # Completa último compasso para duração consistente por measure.
    if cursor > 0:
        remainder = cursor % mlen
        if remainder > 1e-6:
            r = m21note.Rest()
            r.quarterLength = round(mlen - remainder, 6)
            part.insert(cursor, r)
    else:
        r = m21note.Rest()
        r.quarterLength = mlen
        part.insert(0, r)


def _fill_other(part, voices: list, mlen: float, m21note, m21chord, m21stream) -> None:
    """Preenche `other`: 1 voice -> flat; N voices -> Voices explícitas."""
    if not voices or not voices[0]:
        r = m21note.Rest()
        r.quarterLength = mlen
        part.insert(0, r)
        return
    if len(voices) == 1:
        cursor = 0.0
        for item in sorted(voices[0], key=lambda i: (i["start"], i["end"])):
            start = round(float(item["start"]), 6)
            end = round(float(item["end"]), 6)
            if start > cursor + 1e-9:
                r = m21note.Rest()
                r.quarterLength = round(start - cursor, 6)
                part.insert(cursor, r)
                cursor = start
            dur = round(end - start, 6)
            if dur <= 0:
                continue
            if item["kind"] == "chord":
                c = m21chord.Chord([int(p) for p in item["pitches"]])
                c.quarterLength = dur
                try:
                    c.volume.velocity = max(1, min(127, int(item.get("velocity", 64))))
                except Exception:
                    pass
                part.insert(start, c)
            else:
                part.insert(start, _make_note(
                    m21note, int(item["pitches"][0]), dur, int(item.get("velocity", 64))))
            cursor = max(cursor, end)
        if cursor > 0:
            remainder = cursor % mlen
            if remainder > 1e-6:
                r = m21note.Rest()
                r.quarterLength = round(mlen - remainder, 6)
                part.insert(cursor, r)
        return
    # Múltiplas voices: cada voice monofônica com rests, mesma duração total.
    total = 0.0
    for voice_items in voices:
        for item in voice_items:
            total = max(total, round(float(item["end"]), 6))
    if total <= 0:
        total = mlen
    else:
        n = int(total // mlen)
        if total % mlen > 1e-6:
            n += 1
        total = round(n * mlen, 6)
    for vi, voice_items in enumerate(voices):
        v = m21stream.Voice()
        # Voice id 1-based: MusicXML <voice>0</voice> é inválido e o
        # MuseScore ignora as notas (parte aparece vazia).
        v.id = vi + 1
        cursor = 0.0
        for item in sorted(voice_items, key=lambda i: (i["start"], i["end"])):
            start = round(float(item["start"]), 6)
            end = round(float(item["end"]), 6)
            if start > cursor + 1e-9:
                r = m21note.Rest()
                r.quarterLength = round(start - cursor, 6)
                v.insert(cursor, r)
                cursor = start
            dur = round(end - start, 6)
            if dur <= 0:
                continue
            if item["kind"] == "chord":
                c = m21chord.Chord([int(p) for p in item["pitches"]])
                c.quarterLength = dur
                v.insert(start, c)
            else:
                v.insert(start, _make_note(
                    m21note, int(item["pitches"][0]), dur, int(item.get("velocity", 64))))
            cursor = max(cursor, end)
        if total > cursor + 1e-9:
            r = m21note.Rest()
            r.quarterLength = round(total - cursor, 6)
            v.insert(cursor, r)
        part.insert(0, v)


def _count_elements(part) -> dict:
    """Conta Note/Chord/Rest de uma Part (pré-write ou pós-reparse).

    music21 é importado localmente para não vazar para o processo FastAPI.
    Chord é testado antes de Note para contagem exata por tipo.
    """
    from music21 import chord as m21chord
    from music21 import note as m21note
    nn = cc = rr = 0
    for el in part.flatten().notesAndRests:
        if isinstance(el, m21chord.Chord):
            cc += 1
        elif isinstance(el, m21note.Note):
            nn += 1
        elif isinstance(el, m21note.Rest):
            rr += 1
    return {"notes": nn, "chords": cc, "rests": rr}


def validate_musicxml(musicxml_path: Path) -> dict:
    """Validação estrutural crítica (10 checagens da spec + detalhe por parte)."""
    from music21 import chord as m21chord
    from music21 import converter, meter, note as m21note, stream, tempo
    info: dict = {}
    if not musicxml_path.is_file():
        raise ValueError("MusicXML não existe")
    size = musicxml_path.stat().st_size
    if size <= 0:
        raise ValueError("MusicXML vazio")
    info["size"] = size
    ET.parse(str(musicxml_path))  # XML parseável ou lança
    info["xml_ok"] = True
    parsed = converter.parse(str(musicxml_path))  # reabre no music21 ou lança
    parts = list(parsed.parts)
    if len(parts) != 3:
        raise ValueError(f"Score deve conter 3 Parts, encontrado {len(parts)}")
    info["parts"] = len(parts)
    ts = list(parsed.recurse().getElementsByClass(meter.TimeSignature))
    if not ts:
        raise ValueError("TimeSignature ausente")
    info["time_signature"] = ts[0].ratioString
    mm = list(parsed.recurse().getElementsByClass(tempo.MetronomeMark))
    if not mm:
        raise ValueError("Andamento (MetronomeMark) ausente")
    info["tempo"] = mm[0].number
    measures = 0
    parts_detail = []
    for pt in parts:
        measures += len(list(pt.getElementsByClass(stream.Measure)))
        counts = _count_elements(pt)
        parts_detail.append({
            "name": pt.partName or "",
            "notes": counts["notes"],
            "chords": counts["chords"],
            "rests": counts["rests"],
        })
    info["parts_detail"] = parts_detail
    if measures == 0:
        raise ValueError("Measures ausentes")
    info["measures"] = measures
    notes = list(parsed.recurse().notes)
    if not notes:
        raise ValueError("Nenhuma nota na partitura")
    info["notes"] = len(notes)
    for el in parsed.recurse().notes:
        pitches = el.pitches if hasattr(el, "pitches") else [el.pitch]
        for p in pitches:
            if p.midi < 0 or p.midi > 127:
                raise ValueError(f"Pitch fora de 0-127: {p.midi}")
    return info


def main() -> None:
    args = parse_args()
    warnings: list = []

    tempo = float(args.tempo)
    grid = grid_step_beats(args.quantization)
    mlen = beats_per_measure(args.time_signature)
    beat_offset = float(args.beat_offset or 0.0)

    trans_dir = Path(args.transcriptions_dir)
    out_xml = Path(args.output_musicxml)
    out_model = Path(args.output_model)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    out_model.parent.mkdir(parents=True, exist_ok=True)

    raw_counts: dict = {}
    vocals_data = load_events(trans_dir, "vocals")
    bass_data = load_events(trans_dir, "bass")
    other_data = load_events(trans_dir, "other")
    raw_counts = {
        "vocals": len(vocals_data["events"]),
        "bass": len(bass_data["events"]),
        "other": len(other_data["events"]),
    }

    vocals_notes, vocals_stats = process_monophonic(
        vocals_data["events"], tempo, beat_offset, grid)
    bass_notes, bass_stats = process_monophonic(
        bass_data["events"], tempo, beat_offset, grid)
    other_items, other_voices, other_stats = process_other(
        other_data["events"], tempo, beat_offset, grid)

    other_pitches = [int(ev["pitch"]) for ev in other_data["events"]
                     if isinstance(ev.get("pitch"), (int, float))]
    other_clef = choose_clef_other(other_pitches)

    if vocals_stats.get("pickups_clamped"):
        warnings.append(
            f"{vocals_stats['pickups_clamped']} nota(s) de vocais antes do beat 0; "
            "leading rest adicionado (anacruse simplificada)."
        )
    if other_stats.get("voice_overflow"):
        warnings.append(
            f"Polifonia de `other` excedeu {4} vozes em "
            f"{other_stats['voice_overflow']} trecho(s); estratégia determinística aplicada. "
            "Tuplets automáticos serão aprimorados futuramente."
        )
    if bass_stats.get("large_overlaps") or vocals_stats.get("large_overlaps"):
        warnings.append("Overlaps grandes resolvidos por encurtamento; revisar musicalmente.")

    key_norm, mode_norm, key_warning = resolve_key_signature(
        args.key, args.mode, args.key_confidence, args.key_mode)
    if key_warning:
        warnings.append(key_warning)

    processed = {
        "vocals": {"notes": vocals_notes},
        "bass": {"notes": bass_notes},
        "other": {"voices": other_voices},
        "other_clef": other_clef,
    }
    score = build_score(args, processed, key_norm, mode_norm, warnings)

    # Contagem pré-write por parte (ordem de inserção: vocals, bass, other).
    written_counts: dict = {}
    for stem, part in zip(("vocals", "bass", "other"), list(score.parts)):
        written_counts[stem] = _count_elements(part)

    score.write("musicxml", fp=str(out_xml))

    validation = validate_musicxml(out_xml)

    # Validação forte por parte: transcrição com notas -> parte não vazia.
    reparsed_counts = {
        d["name"]: {"notes": d["notes"], "chords": d["chords"]}
        for d in validation.get("parts_detail", [])
    }
    cleaned_counts = {
        "vocals": vocals_stats.get("cleaned", 0),
        "bass": bass_stats.get("cleaned", 0),
        "other": other_stats.get("cleaned", 0),
    }
    parts_ok, parts_errors, parts_warnings = check_parts_nonempty(
        raw_counts, cleaned_counts, reparsed_counts)
    warnings.extend(parts_warnings)
    if not parts_ok:
        raise ValueError("; ".join(parts_errors))

    if other_stats.get("duplicate_pitches_removed"):
        warnings.append(
            f"{other_stats['duplicate_pitches_removed']} pitch(es) duplicado(s) "
            f"removido(s) de acordes em `other` (colapso de grade)."
        )

    def _rep(name: str, field: str) -> int:
        return int(reparsed_counts.get(name, {}).get(field, 0))

    model = {
        "file_id": args.file_id,
        "tempo": int(tempo) if float(tempo).is_integer() else tempo,
        "time_signature": args.time_signature,
        "quantization": args.quantization,
        "key_mode": args.key_mode,
        "key": key_norm,
        "mode": mode_norm,
        "key_confidence": args.key_confidence,
        "key_warning": key_warning,
        "beat_offset": beat_offset,
        "config_key": config_key(
            int(tempo) if float(tempo).is_integer() else tempo,
            args.time_signature, args.quantization, args.key_mode),
        "parts": {
            "vocals": {
                "raw_events": raw_counts["vocals"],
                "cleaned_events": vocals_stats.get("cleaned", 0),
                "quantized_events": vocals_stats.get("quantized", 0),
                "merged_notes": vocals_stats.get("merged", 0),
                "voices": 1,
                "chords": 0,
                "voice_overflow": 0,
                "duplicate_pitches_removed": 0,
                "written_notes": written_counts["vocals"]["notes"],
                "written_chords": written_counts["vocals"]["chords"],
                "reparsed_notes": _rep("Vocais", "notes"),
                "reparsed_chords": _rep("Vocais", "chords"),
                "clef": "treble",
            },
            "bass": {
                "raw_events": raw_counts["bass"],
                "cleaned_events": bass_stats.get("cleaned", 0),
                "quantized_events": bass_stats.get("quantized", 0),
                "merged_notes": bass_stats.get("merged", 0),
                "voices": 1,
                "chords": 0,
                "voice_overflow": 0,
                "duplicate_pitches_removed": 0,
                "written_notes": written_counts["bass"]["notes"],
                "written_chords": written_counts["bass"]["chords"],
                "reparsed_notes": _rep("Baixo", "notes"),
                "reparsed_chords": _rep("Baixo", "chords"),
                "clef": "bass",
            },
            "other": {
                "raw_events": raw_counts["other"],
                "cleaned_events": other_stats.get("cleaned", 0),
                "quantized_events": other_stats.get("quantized", 0),
                "merged_notes": other_stats.get("merged", 0),
                "voices": other_stats.get("voices", 0),
                "chords": other_stats.get("chords", 0),
                "voice_overflow": other_stats.get("voice_overflow", 0),
                "duplicate_pitches_removed": other_stats.get("duplicate_pitches_removed", 0),
                "written_notes": written_counts["other"]["notes"],
                "written_chords": written_counts["other"]["chords"],
                "reparsed_notes": _rep("Outros", "notes"),
                "reparsed_chords": _rep("Outros", "chords"),
                "clef": other_clef,
            },
        },
        "measures": validation.get("measures"),
        "notes": validation.get("notes"),
        "warnings": warnings,
    }
    with open(out_model, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

    summary = {
        "file_id": args.file_id,
        "tempo": model["tempo"],
        "time_signature": args.time_signature,
        "quantization": args.quantization,
        "key": key_norm,
        "parts": 3,
        "measures": validation.get("measures"),
        "notes": validation.get("notes"),
        "musicxml": str(out_xml),
        "model": str(out_model),
        "warnings": warnings,
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
