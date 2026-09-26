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
from typing import List, Optional

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
from backend.arrangement.styles import (  # noqa: E402
    apply_style_to_drums,
    get_style,
    validate_style,
)
from backend.drums.drum_transcriber import get_drums_json_path  # noqa: E402
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
    p.add_argument("--arrangement-style", default="automatic")
    p.add_argument("--dynamics", default="automatic")
    p.add_argument("--drums-json", default=None)
    p.add_argument("--include-original-parts", action="store_true")
    p.add_argument("--include-drums", action="store_true")
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


# ---------------------------------------------------------------------------
# Bateria / percussão (Etapa 8)
# ---------------------------------------------------------------------------

# displayStep, displayOctave, notehead|None, quarterLength, storedInstrument|None.
# Posições em pauta de percussão: kick inferior, snare central, hats/pratos X.
DRUM_DISPLAY = {
    "kick": ("F", 4, None, 0.5, "BassDrum"),
    "snare": ("C", 5, None, 0.5, "SnareDrum"),
    "closed_hihat": ("G", 5, "x", 0.25, None),
    "open_hihat": ("A", 5, "x", 0.5, None),
    "crash": ("B", 5, "x", 1.0, "CrashCymbals"),
    "tom_low": ("A", 4, None, 0.5, None),
    "tom_mid": ("D", 5, None, 0.5, None),
    "tom_high": ("F", 5, None, 0.5, None),
}
# Voice 1 (pratos/caixa/toms) x Voice 2 (bumbo). Sem voice 0 (Etapa 6).
DRUM_UPPER = {"snare", "closed_hihat", "open_hihat", "crash",
              "tom_low", "tom_mid", "tom_high"}
DRUM_LOWER = {"kick"}


def _drum_hit(inst: str, qlen: float, strength: float = 0.6):
    """Uma cabeça de percussão (sempre instância nova)."""
    from music21 import instrument as m21instrument
    from music21 import note as m21note
    step, octv, head, _, stored = DRUM_DISPLAY[inst]
    n = m21note.Unpitched()
    n.displayStep = step
    n.displayOctave = octv
    n.quarterLength = float(qlen)
    if head:
        n.notehead = head
    if stored:
        try:
            n.storedInstrument = getattr(m21instrument, stored)()
        except Exception:
            pass
    try:
        n.volume.velocity = max(1, min(127, int(32 + float(strength) * 95)))
    except Exception:
        pass
    return n


def build_drum_part(drum_events: list, args, mlen: float, first_part: bool):
    """Parte Bateria: PercussionClef, 2 voices (1=pratos/caixa/toms, 2=bumbo).

    Hits simultâneos de classes distintas (ex.: snare + closed_hihat no mesmo
    tempo) caem na mesma voice (UPPER) e são preservados. Sem KeySignature,
    sem ties.
    """
    from music21 import clef as m21clef
    from music21 import instrument as m21instrument
    from music21 import meter as m21meter
    from music21 import note as m21note
    from music21 import stream as m21stream
    from music21 import tempo as m21tempo

    part = m21stream.Part()
    part.partName = "Bateria"
    part.partAbbreviation = "Bat."
    part.insert(0, m21instrument.UnpitchedPercussion())
    if first_part:
        part.insert(0, m21tempo.MetronomeMark(number=int(args.tempo)))
    part.insert(0, m21clef.PercussionClef())
    part.insert(0, m21meter.TimeSignature(args.time_signature))
    total = 0.0
    for e in drum_events:
        total = max(total, round(float(e.get("beat", 0)), 6) + 0.5)
    if total <= 0:
        raise ValueError("Eventos de bateria vazios")
    n_meas = max(1, int(total // mlen) + (1 if total % mlen > 1e-6 else 0))
    total = round(n_meas * mlen, 6)
    for voice_id, classes in ((1, DRUM_UPPER), (2, DRUM_LOWER)):
        v = m21stream.Voice()
        v.id = voice_id
        cursor = 0.0
        for e in sorted(drum_events, key=lambda x: float(x.get("beat", 0))):
            if e.get("instrument") not in classes:
                continue
            start = round(float(e.get("beat", 0)), 6)
            ql = DRUM_DISPLAY[e["instrument"]][3]
            if start > cursor + 1e-9:
                r = m21note.Rest()
                r.quarterLength = round(start - cursor, 6)
                v.insert(cursor, r)
                cursor = start
            # Permite notas simultâneas na mesma voice (ex.: snare + hat no
            # mesmo tempo). Não pula se start <= cursor; apenas insere.
            v.insert(start, _drum_hit(e["instrument"], ql,
                                      float(e.get("strength", 0.6))))
            cursor = max(cursor, start + ql)
        if total > cursor + 1e-9:
            r = m21note.Rest()
            r.quarterLength = round(total - cursor, 6)
            v.insert(cursor, r)
        part.insert(0, v)
    try:
        part.makeMeasures(inPlace=True)
    except Exception as e:
        raise ValueError(f"makeMeasures falhou na bateria: {e}")
    _renumber_voices(part)
    return part


# ---------------------------------------------------------------------------
# Dinâmica e articulações (Etapa 8)
# ---------------------------------------------------------------------------

DYNAMIC_LEVELS = ["pp", "p", "mp", "mf", "f", "ff"]


def _velocity_levels(velocities: List[float], bias: int = 0) -> List[int]:
    """Percentis p20/p40/p60/p80 por parte -> níveis 0..5 + bias (relativo)."""
    vs = sorted(velocities)
    if not vs:
        return []
    if max(vs) - min(vs) < 1e-9:
        return [max(0, min(5, 2 + bias))] * len(vs)  # uniforme -> mp
    def pct(p: float) -> float:
        return vs[min(len(vs) - 1, int(len(vs) * p / 100.0))]
    cuts = [pct(20), pct(40), pct(60), pct(80)]

    def level(v: float) -> int:
        lv = 5
        for i, c in enumerate(cuts):
            if v < c:
                lv = i
                break
        else:
            lv = 4 if v < (cuts[3] + (max(vs) - cuts[3]) / 2.0) else 5
        return max(0, min(5, lv + bias))
    return [level(v) for v in velocities]


def _part_note_items(part):
    """[(offset_absoluto, quarterLength, elemento)] ordenados (pós-measures)."""
    from music21 import stream as m21stream
    items = []
    for m in part.getElementsByClass(m21stream.Measure):
        base = float(part.elementOffset(m))
        for el in m.notesAndRests:
            if el.isRest:
                continue
            items.append((round(base + float(el.offset), 6), float(el.quarterLength), el))
    items.sort(key=lambda t: t[0])
    return items


def _insert_at_absolute(part, offset: float, el) -> None:
    """Insere elemento em offset absoluto (dentro do Measure correto)."""
    from music21 import stream as m21stream
    for m in part.getElementsByClass(m21stream.Measure):
        base = float(part.elementOffset(m))
        if base <= offset < base + float(m.quarterLength) + 1e-9:
            m.insert(max(0.0, offset - base), el)
            return
    part.insert(offset, el)


def apply_dynamics_articulations(part, part_name: str, is_melody: bool,
                                 is_harmony: bool, dynamics_mode: str,
                                 dyn_bias: int, accent_downbeats: bool) -> dict:
    """Dinâmica por blocos (percentis da parte) + acentos/staccato/tenuto.

    Não altera pitches. Slurs: omitidos (regras frágeis — etapa futura).
    """
    from music21 import articulations as m21art
    from music21 import chord as m21chord
    from music21 import dynamics as m21dyn
    from music21 import note as m21note
    stats = {"dynamics_marks": 0, "accents": 0, "staccatos": 0, "tenutos": 0}
    items = _part_note_items(part)
    if not items:
        return stats
    vels = []
    for _, _, el in items:
        try:
            if isinstance(el, m21chord.Chord):
                vels.append(float(el.volume.velocity))
            elif isinstance(el, m21note.Note):
                vels.append(float(el.volume.velocity))
            else:  # Unpitched
                vels.append(float(getattr(el.volume, "velocity", 64)))
        except Exception:
            vels.append(64.0)
    levels = _velocity_levels(vels, bias=dyn_bias) if dynamics_mode == "automatic" else [2] * len(items)
    # Dinâmica: marca só quando o nível muda (blocos, sem poluir).
    if dynamics_mode == "automatic":
        current = None
        for (off, _, _), lv in zip(items, levels):
            if lv != current:
                _insert_at_absolute(part, off, m21dyn.Dynamic(DYNAMIC_LEVELS[lv]))
                stats["dynamics_marks"] += 1
                current = lv
    # Articulações (mutação direta, sem re-split: pós-makeMeasures/makeTies).
    for idx, (off, ql, el) in enumerate(items):
        tied = getattr(el, "tie", None)
        if tied is not None and tied.type in ("continue", "stop"):
            continue  # só cabeça do tie recebe articulação
        is_strong = abs(off - round(off)) < 1e-6
        vel = vels[idx]
        if vel >= sorted(vels)[min(len(vels) - 1, int(len(vels) * 0.9))]:
            if is_strong or (accent_downbeats and el.__class__.__name__ != "Rest"):
                try:
                    el.articulations.append(m21art.Accent())
                    stats["accents"] += 1
                    continue
                except Exception:
                    pass
        if is_melody and ql <= 0.25 + 1e-9 and tied is None:
            nxt = items[idx + 1][0] if idx + 1 < len(items) else off + ql
            if nxt - (off + ql) >= 0.25 - 1e-9 and el.__class__.__name__ == "Note":
                try:
                    el.articulations.append(m21art.Staccato())
                    stats["staccatos"] += 1
                except Exception:
                    pass
        elif is_harmony and ql >= 2.0 - 1e-9:
            try:
                el.articulations.append(m21art.Tenuto())
                stats["tenutos"] += 1
            except Exception:
                pass
    return stats


def validate_arrangement(musicxml_path: Path, expected: list, concert_lines: dict,
                         include_originals: bool, drums_expected: bool = False,
                         drums_input_beats: Optional[list] = None) -> dict:
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
    if drums_expected:
        from music21 import clef as _m21clef
        dpt = by_name.get("Bateria")
        if dpt is None:
            raise ValueError("Parte Bateria ausente")
        dunpitched = [e for e in dpt.flatten().notesAndRests
                      if type(e).__name__ == "Unpitched"]
        if not dunpitched:
            raise ValueError("Parte Bateria vazia no reparse")
        clefs = list(dpt.recurse().getElementsByClass(_m21clef.PercussionClef))
        if not clefs:
            raise ValueError("PercussionClef ausente na bateria")
        nvoices = sum(len(list(m.voices)) for m in
                      dpt.getElementsByClass(stream.Measure))
        if nvoices == 0:
            raise ValueError("Bateria sem voices")
        info["drums_notes"] = len(dunpitched)
        if drums_input_beats:
            sim_in = sum(1 for b in set(drums_input_beats)
                         if drums_input_beats.count(b) >= 2)
            if sim_in and nvoices < 2:
                raise ValueError("Simultaneidade da bateria perdida (1 voice)")
            info["drums_simultaneous_beats"] = sim_in
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
    style_name = validate_style(getattr(args, "arrangement_style", "automatic"))
    style = get_style(style_name)
    dynamics_mode = getattr(args, "dynamics", "automatic")
    if dynamics_mode not in ("automatic", "none"):
        raise ValueError("dynamics inválido. Permitidos: automatic, none.")
    lines, report = arrange(melody, other_items, defs, mode=args.mode,
                            simplify=True, profile=profile,
                            bass_notes=bass_notes,
                            style_params={"breath_mult": style["breath_mult"],
                                          "harm_min_dur_mult": style["harm_min_dur_mult"]})
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

    # Bateria (Etapa 8): parte de percussão se solicitada e transcrita.
    drum_events: list = []
    drums_wanted = bool(getattr(args, "include_drums", False))
    drums_path = getattr(args, "drums_json", None) or str(
        Path(__file__).resolve().parents[2] / "drums" / args.file_id / "drums.json")
    if drums_wanted:
        try:
            with open(drums_path, "r", encoding="utf-8") as f:
                drums_data = json.load(f)
            drum_events = apply_style_to_drums(
                drums_data.get("events", []) or [], style_name)
            removed = len(drums_data.get("events", []) or []) - len(drum_events)
            if removed:
                warnings.append(f"Estilo {style_name}: {removed} hit(s) de bateria filtrados.")
        except FileNotFoundError:
            warnings.append("Bateria solicitada mas transcrição indisponível; "
                            "transcreva a bateria primeiro.")
            drum_events = []
        except Exception as e:
            warnings.append(f"Bateria ignorada (JSON inválido): {e}")
            drum_events = []
    drum_part_built = False
    if drum_events:
        try:
            dpart = build_drum_part(drum_events, args, mlen, first_part=first_wind)
            first_wind = False
            score.insert(0, dpart)
            drum_part_built = True
        except Exception as e:
            warnings.append(f"Parte de bateria não gerada: {e}")

    # Dinâmica/articulações (pós-measures; não altera pitches).
    melody_names = {"Vocais"}
    for inst_id, role in report.get("roles", {}).items():
        if role == "melody":
            d = get_instrument(inst_id)
            if d:
                melody_names.add(d.name)
    harmony_names = {"Outros"}
    for inst_id in ("alto_sax", "trombone"):
        d = get_instrument(inst_id)
        if d:
            harmony_names.add(d.name)
    dyn_totals = {"dynamics_marks": 0, "accents": 0, "staccatos": 0, "tenutos": 0}
    if dynamics_mode == "automatic":
        for pt in score.parts:
            pname = pt.partName or ""
            res = apply_dynamics_articulations(
                pt, pname, is_melody=pname in melody_names,
                is_harmony=pname in harmony_names, dynamics_mode=dynamics_mode,
                dyn_bias=int(style["dynamics_bias"]),
                accent_downbeats=bool(style["accent_downbeats"]))
            for k in dyn_totals:
                dyn_totals[k] += res.get(k, 0)

    score.write("musicxml", fp=str(out_xml))
    validation = validate_arrangement(
        out_xml, inst_ids, lines, args.include_original_parts,
        drums_expected=drum_part_built,
        drums_input_beats=[round(float(e.get("beat", 0)), 6) for e in drum_events])

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
        "arrangement_style": style_name,
        "include_drums": drums_wanted,
        "dynamics": dynamics_mode,
        "drums": ({
            "events": len(drum_events),
            "part_built": drum_part_built,
            "notes": validation.get("drums_notes", 0),
        } if (drums_wanted or drum_part_built) else None),
        "dynamics_stats": dyn_totals,
        "config_key": _ck(inst_ids, args.mode, bool(args.include_original_parts),
                          cleanup_profile=profile, arrangement_style=style_name,
                          include_drums=drums_wanted, dynamics=dynamics_mode),
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
