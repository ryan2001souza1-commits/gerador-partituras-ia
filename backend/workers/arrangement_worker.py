#!/usr/bin/env python
"""Worker de arranjo — executado via .venv-notation (music21 já instalado).

Re-deriva as linhas quantizadas da base (determinístico, mesma config da
Etapa 6 lida do score.json), aplica o arranjador (concert pitch) e exporta
MusicXML com partes originais (opcional) + sopros transpostos.

Uso: ver arrangement_generator._build_arrange_command.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.arrangement.arranger import (  # noqa: E402
    arrange,
    extract_main_melody,
    written_line,
)
from backend.arrangement.instrument_definitions import (  # noqa: E402
    REGISTER_ORDER,
    concert_to_written,
    fit_range,
    get_instrument,
)
from backend.musical.cleanup import validate_cleanup_profile  # noqa: E402
from backend.notation.score_utils import (  # noqa: E402
    PART_NAMES,
    beats_per_measure,
    check_parts_nonempty,
    choose_clef_other,
    config_key,
    grid_step_beats,
    normalize_key,
    resolve_key_signature,
)
from backend.workers.notation_worker import (  # noqa: E402
    _count_elements,
    _fill_monophonic,
    _renumber_voices,
    build_score,
    process_monophonic,
    process_other,
)

# Escala ascendente (som -> escrito) por transposição em semitons.
_ASC_INTERVAL = {9: "M6", 14: "M9", 2: "M2", 0: "P1"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Arrangement worker (music21)")
    p.add_argument("--file-id", required=True)
    p.add_argument("--transcriptions-dir", required=True)
    p.add_argument("--output-musicxml", required=True)
    p.add_argument("--output-model", required=True)
    p.add_argument("--tempo", required=True, type=float)
    p.add_argument("--time-signature", required=True)
    p.add_argument("--quantization", required=True)
    p.add_argument("--key-mode", required=True)
    p.add_argument("--key", default=None)
    p.add_argument("--concert-mode", default=None)
    p.add_argument("--key-confidence", default=None, type=float)
    p.add_argument("--beat-offset", default=0.0, type=float)
    p.add_argument("--instruments", required=True)
    p.add_argument("--mode", default="automatic")
    p.add_argument("--base-config-key", default="")
    p.add_argument("--cleanup-profile", default="detailed")
    p.add_argument("--include-original-parts", action="store_true")
    return p.parse_args()


def load_events(trans_dir: Path, stem: str) -> list:
    with open(trans_dir / f"{stem}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    events = data.get("events", [])
    if not isinstance(events, list):
        raise ValueError(f"JSON de {stem} sem lista 'events'")
    return events


def written_key_sharps(key_norm, mode_norm, trans_semitones: int):
    """Armadura escrita: transpõe a concert key para cima (som -> escrito)."""
    from music21 import key as m21key
    if not key_norm or not mode_norm:
        return 0
    try:
        interval_name = _ASC_INTERVAL.get(int(trans_semitones), "P1")
        wk = m21key.Key(key_norm, mode_norm).transpose(interval_name)
        nk, nm = normalize_key(wk.tonic.name, wk.mode)
        return m21key.Key(nk or wk.tonic.name, nm or wk.mode).sharps
    except Exception:
        return 0


def build_wind_part(inst_id: str, concert_line: list, args, key_norm,
                    mode_norm, mlen: float, first_part: bool):
    """Parte transposta (notas em written pitch + instrumento + armadura)."""
    from music21 import clef as m21clef
    from music21 import instrument as m21instrument
    from music21 import key as m21key
    from music21 import meter as m21meter
    from music21 import note as m21note
    from music21 import stream as m21stream
    from music21 import tempo as m21tempo

    definition = get_instrument(inst_id)
    part = m21stream.Part()
    part.partName = definition.name
    part.partAbbreviation = definition.short_name
    inst_cls = getattr(m21instrument, definition.music21_instrument)
    part.insert(0, inst_cls())
    if first_part:
        part.insert(0, m21tempo.MetronomeMark(number=int(args.tempo)))
    part.insert(0, m21clef.TrebleClef() if definition.clef == "treble" else m21clef.BassClef())
    part.insert(0, m21meter.TimeSignature(args.time_signature))
    part.insert(0, m21key.KeySignature(
        written_key_sharps(key_norm, mode_norm, definition.transposition_semitones)))
    written = [{"start": n["start"], "end": n["end"],
                "pitch": concert_to_written(int(n["pitch"]), definition),
                "velocity": int(n.get("velocity", 64))}
               for n in concert_line]
    _fill_monophonic(part, written, mlen, m21note, m21stream)
    try:
        part.makeMeasures(inPlace=True)
    except Exception as e:
        raise ValueError(f"makeMeasures falhou em {inst_id}: {e}")
    try:
        part.makeTies(inPlace=True)
    except Exception:
        pass
    _renumber_voices(part)
    return part


def validate_arrangement(musicxml_path: Path, expected: list, concert_lines: dict,
                         include_originals: bool) -> dict:
    """Reabre e valida por parte (nomes), transposição, ranges e voice 0."""
    from music21 import converter, meter, stream, tempo
    info: dict = {}
    if not musicxml_path.is_file():
        raise ValueError("MusicXML do arranjo não existe")
    if musicxml_path.stat().st_size <= 0:
        raise ValueError("MusicXML do arranjo vazio")
    xml_text = musicxml_path.read_text(encoding="utf-8")
    ET.fromstring(xml_text)
    if "<voice>0</voice>" in xml_text:
        raise ValueError("Voice 0 inválida no MusicXML do arranjo")
    parsed = converter.parse(str(musicxml_path))
    parts = list(parsed.parts)
    by_name = {pt.partName or "": pt for pt in parts}
    if include_originals:
        for pname in ("Vocais", "Baixo", "Outros"):
            if pname not in by_name:
                raise ValueError(f"Parte original ausente: {pname}")
    for inst_id in expected:
        definition = get_instrument(inst_id)
        pt = by_name.get(definition.name)
        if pt is None:
            raise ValueError(f"Parte ausente: {definition.name}")
        counts = _count_elements(pt)
        if counts["notes"] + counts["chords"] == 0:
            raise ValueError(f"Parte '{definition.name}' vazia no reparse")
        # Transposição declarada (som -> escrito confere).
        insts = list(pt.getElementsByClass(__import__("music21").instrument.Instrument))
        trans = insts[0].transposition if insts else None
        if definition.transposition_semitones != 0:
            if trans is None:
                raise ValueError(f"Transposição ausente em {definition.name}")
            if int(trans.semitones) != -int(definition.transposition_semitones):
                raise ValueError(
                    f"Transposição incorreta em {definition.name}: {trans.semitones}")
        # Ranges escritos (absolutos) + pitches válidos.
        for el in pt.flatten().notesAndRests:
            pitches = getattr(el, "pitches", None) or [getattr(el, "pitch", None)]
            for px in pitches:
                if px is None:
                    continue
                if px.midi < 0 or px.midi > 127:
                    raise ValueError(f"Pitch inválido em {definition.name}")
                if px.midi < definition.written_low or px.midi > definition.written_high:
                    raise ValueError(
                        f"Pitch {px.midi} fora da tessitura de {definition.name}")
        # Som real da primeira nota confere com a linha concert.
        line = concert_lines.get(inst_id, [])
        if line:
            first = None
            for el in pt.flatten().notesAndRests:
                if hasattr(el, "pitches") and len(getattr(el, "pitches", [])) > 0:
                    first = el.pitches[0].midi
                    break
                if hasattr(el, "pitch") and el.pitch is not None:
                    first = el.pitch.midi
                    break
            if first is not None:
                sounding = first - definition.transposition_semitones
                if sounding != int(line[0]["pitch"]):
                    raise ValueError(
                        f"Concert pitch divergente em {definition.name}")
    ts = list(parsed.recurse().getElementsByClass(meter.TimeSignature))
    if not ts:
        raise ValueError("TimeSignature ausente no arranjo")
    mm = list(parsed.recurse().getElementsByClass(tempo.MetronomeMark))
    if not mm:
        raise ValueError("Andamento ausente no arranjo")
    info.update(parts=len(parts), tempo=mm[0].number,
                time_signature=ts[0].ratioString,
                measures=sum(len(list(pt.getElementsByClass(stream.Measure))) for pt in parts),
                notes=len(list(parsed.recurse().notes)))
    return info


def main() -> None:
    args = parse_args()
    warnings: list = []

    tempo = float(args.tempo)
    grid = grid_step_beats(args.quantization)
    mlen = beats_per_measure(args.time_signature)
    beat_offset = float(args.beat_offset or 0.0)
    profile = validate_cleanup_profile(getattr(args, "cleanup_profile", "detailed"))
    inst_ids = [s for s in args.instruments.split(",") if s]
    if not inst_ids:
        raise ValueError("Nenhum instrumento selecionado")
    for i in inst_ids:
        if get_instrument(i) is None:
            raise ValueError(f"Instrumento inválido: {i}")

    trans_dir = Path(args.transcriptions_dir)
    out_xml = Path(args.output_musicxml)
    out_model = Path(args.output_model)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    out_model.parent.mkdir(parents=True, exist_ok=True)

    vocals_notes, vocals_stats = process_monophonic(
        load_events(trans_dir, "vocals"), tempo, beat_offset, grid,
        profile=profile, smooth=True)
    bass_notes, bass_stats = process_monophonic(
        load_events(trans_dir, "bass"), tempo, beat_offset, grid, profile=profile)
    other_items, other_voices, other_stats = process_other(
        load_events(trans_dir, "other"), tempo, beat_offset, grid, profile=profile,
        bass_notes=bass_notes if profile == "natural" else None)
    other_flat = load_events(trans_dir, "other")
    other_pitches = [int(ev["pitch"]) for ev in other_flat
                     if isinstance(ev.get("pitch"), (int, float))]
    other_clef = choose_clef_other(other_pitches)

    key_norm, mode_norm, key_warning = resolve_key_signature(
        args.key, args.concert_mode, args.key_confidence, args.key_mode)
    if key_warning:
        warnings.append(key_warning)
        warnings.append("Tonalidade original possui baixa confiança; "
                        "harmonia baseada principalmente nas notas detectadas.")

    melody, mel_stats = extract_main_melody(
        vocals_notes,
        [{"start": it["start"], "end": it["end"], "pitches": it["pitches"],
          "velocity": it.get("velocity", 64)} for it in other_items],
    )
    if not melody:
        raise ValueError("Melodia vazia: sem material em vocals/other.")
    defs = [get_instrument(i) for i in inst_ids]
    lines, report = arrange(melody, other_items, defs, mode=args.mode,
                            simplify=True, profile=profile,
                            bass_notes=bass_notes)
    for inst_id, st in report.get("stats", {}).items():
        if inst_id not in report.get("roles", {}) or not isinstance(st, dict):
            continue  # ex. stats de simplify: não é instrumento
        if st.get("range_adjustments"):
            d = get_instrument(inst_id)
            warnings.append(
                f"{st['range_adjustments']} nota(s) deslocada(s) uma oitava "
                f"para caber na tessitura de {d.name if d else inst_id}.")
        if st.get("dropped_out_of_range") or st.get("dropped"):
            warnings.append(
                f"{(st.get('dropped_out_of_range') or 0) + (st.get('dropped') or 0)} "
                f"nota(s) fora da tessitura em {inst_id} viraram pausa.")
    if report.get("voice_crossings"):
        warnings.append(f"{report['voice_crossings']} cruzamento(s) de vozes no arranjo; revisar.")

    # Monta score: base (opcional) + sopros em concert->written.
    if args.include_original_parts:
        base_args = SimpleNamespace(tempo=args.tempo, time_signature=args.time_signature)
        score = build_score(base_args, {
            "vocals": {"notes": vocals_notes},
            "bass": {"notes": bass_notes},
            "other": {"voices": other_voices},
            "other_clef": other_clef,
        }, key_norm, mode_norm, warnings)
        first_wind = False
    else:
        from music21 import metadata as m21metadata
        from music21 import stream as m21stream
        score = m21stream.Score()
        md = m21metadata.Metadata()
        md.title = "Gerador de Partituras IA — Arranjo"
        score.metadata = md
        first_wind = True
    for inst_id, role in sorted(report["roles"].items(),
                                key=lambda kv: REGISTER_ORDER.index(kv[0])
                                if kv[0] in REGISTER_ORDER else 99):
        part = build_wind_part(inst_id, lines[inst_id], args, key_norm, mode_norm, mlen,
                               first_part=first_wind)
        first_wind = False
        score.insert(0, part)

    score.write("musicxml", fp=str(out_xml))
    validation = validate_arrangement(out_xml, inst_ids, lines, args.include_original_parts)

    # Checagem das partes originais reaproveitada da Etapa 6.
    if args.include_original_parts:
        from music21 import converter
        parsed = converter.parse(str(out_xml))
        rep = {}
        for pt in parsed.parts:
            c = _count_elements(pt)
            rep[pt.partName or ""] = {"notes": c["notes"], "chords": c["chords"]}
        raw_counts = {"vocals": len(load_events(trans_dir, "vocals")),
                      "bass": len(load_events(trans_dir, "bass")),
                      "other": len(other_flat)}
        cleaned_counts = {"vocals": vocals_stats.get("cleaned", 0),
                          "bass": bass_stats.get("cleaned", 0),
                          "other": other_stats.get("cleaned", 0)}
        ok, errors, ckw = check_parts_nonempty(raw_counts, cleaned_counts, rep)
        warnings.extend(ckw)
        if not ok:
            raise ValueError("; ".join(errors))

    instruments_model = []
    for inst_id in inst_ids:
        d = get_instrument(inst_id)
        st = report.get("stats", {}).get(inst_id, {})
        line = lines.get(inst_id, [])
        instruments_model.append({
            "id": inst_id, "name": d.name, "role": report["roles"][inst_id],
            "notes_count": len(line),
            "source_notes": int(st.get("source_notes", len(line))),
            "final_notes": int(st.get("final_notes", len(line))),
            "large_leaps_before": int(st.get("large_leaps_before", 0)),
            "large_leaps_after": int(st.get("large_leaps_after", 0)),
            "average_interval_before": st.get("average_interval_before", 0.0),
            "average_interval_after": st.get("average_interval_after", 0.0),
            "harmonic_changes_before": int(st.get("harmonic_changes_before", 0)),
            "harmonic_changes_after": int(st.get("harmonic_changes_after", 0)),
            "range_adjustments": int(st.get("range_adjustments", 0)),
            "breath_adjustments": int(st.get("breath", {}).get("breath_adjustments", 0)
                                      if isinstance(st.get("breath"), dict) else 0),
        })
    from backend.arrangement.arrangement_generator import arrangement_config_key as _ck  # noqa
    model = {
        "file_id": args.file_id,
        "tempo": int(tempo) if float(tempo).is_integer() else tempo,
        "time_signature": args.time_signature,
        "quantization": args.quantization,
        "cleanup_profile": profile,
        "concert_key": (f"{key_norm} {mode_norm}" if key_norm else None),
        "key_warning": key_warning,
        "beat_offset": beat_offset,
        "mode": args.mode,
        "include_original_parts": bool(args.include_original_parts),
        "config_key": _ck(inst_ids, args.mode, bool(args.include_original_parts),
                          cleanup_profile=profile),
        "base_config_key": args.base_config_key or "",
        "instruments": instruments_model,
        "melody_source": mel_stats.get("source"),
        "voice_crossings": report.get("voice_crossings", 0),
        "crossings_fixed": report.get("crossings_fixed", 0),
        "naturalness": report.get("naturalness", {}),
        "parts": validation.get("parts"),
        "measures": validation.get("measures"),
        "notes": validation.get("notes"),
        "warnings": warnings,
    }
    with open(out_model, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)
    print(json.dumps({"file_id": args.file_id, "parts": validation.get("parts"),
                      "measures": validation.get("measures"),
                      "notes": validation.get("notes"),
                      "instruments": [i["id"] for i in instruments_model],
                      "warnings": warnings}))


if __name__ == "__main__":
    main()
