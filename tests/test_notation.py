"""
Testes Etapa 6 — quantização musical + compassos + MusicXML (MuseScore 4).

- Lógica pura (score_utils) e API: rápidos, sem music21 no processo principal.
- Validação MusicXML: via worker em .venv-notation (subprocess), com cache
  modular (1 execução cobre pausas, ties, acordes, claves, partes, tempo).
- Teste real (15s, vocals 48/bass 34/other 96): executa se as transcrições
  existirem no ambiente; senão, skip.
"""
import json
import shutil
import subprocess
import tempfile
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parents[1]

from backend.notation.score_generator import (
    get_notation_python,
    is_notation_available,
    get_music21_version,
)
from backend.notation.score_utils import (
    beats_per_measure,
    check_parts_nonempty,
    choose_clef_other,
    clean_events,
    config_key,
    grid_step_beats,
    group_chords_other,
    merge_same_pitch,
    normalize_key,
    quantize_note,
    quantize_value,
    resolve_key_signature,
    resolve_monophonic_overlaps,
    seconds_to_beats,
    should_use_key,
    validate_score_config,
)

WORKER_PATH = BASE_DIR / "backend" / "workers" / "notation_worker.py"


# ---------------------------------------------------------------------------
# Helpers de worker (subprocess .venv-notation)
# ---------------------------------------------------------------------------

def _notation_ok() -> bool:
    py = get_notation_python()
    return bool(py and Path(py).is_file() and is_notation_available())


def _run_worker(trans_dir: Path, out_xml: Path, out_model: Path, extra: list | None = None) -> dict:
    """Executa notation_worker via .venv-notation. Retorna summary (stdout)."""
    py = get_notation_python()
    cmd = [
        str(Path(py).resolve()),
        str(WORKER_PATH.resolve()),
        "--file-id", "synth-test",
        "--transcriptions-dir", str(trans_dir.resolve()),
        "--output-musicxml", str(out_xml.resolve()),
        "--output-model", str(out_model.resolve()),
        "--tempo", "120",
        "--time-signature", "4/4",
        "--quantization", "1/16",
        "--key-mode", "none",
        "--beat-offset", "0.0",
    ]
    if extra:
        cmd.extend(extra)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(BASE_DIR))
    assert result.returncode == 0, f"worker rc={result.returncode} stderr={result.stderr[-800:]}"
    lines = [ln for ln in (result.stdout or "").strip().splitlines() if ln.strip()]
    assert lines, "worker sem stdout"
    return json.loads(lines[-1])


def _dump_parsed(musicxml_path: Path) -> dict:
    """Parseia MusicXML via .venv-notation e retorna estrutura JSON."""
    py = get_notation_python()
    script = (
        "import json, sys\n"
        "from music21 import converter, clef, key, meter, tempo, note, chord, stream\n"
        f"p = converter.parse(r'''{musicxml_path}''')\n"
        "parts = list(p.parts)\n"
        "out = {'parts': [], 'tempo': None, 'time_signature': None}\n"
        "mm = list(p.recurse().getElementsByClass(tempo.MetronomeMark))\n"
        "out['tempo'] = mm[0].number if mm else None\n"
        "ts = list(p.recurse().getElementsByClass(meter.TimeSignature))\n"
        "out['time_signature'] = ts[0].ratioString if ts else None\n"
        "ks = list(p.recurse().getElementsByClass(key.KeySignature))\n"
        "out['key_fifths'] = ks[0].sharps if ks else None\n"
        "for pt in parts:\n"
        "    cl = list(pt.recurse().getElementsByClass(clef.Clef))\n"
        "    els = []\n"
        "    for el in pt.flatten().notesAndRests:\n"
        "        if isinstance(el, note.Rest):\n"
        "            els.append({'t': 'Rest', 'ql': el.quarterLength})\n"
        "        elif isinstance(el, chord.Chord):\n"
        "            els.append({'t': 'Chord', 'ql': el.quarterLength, 'pitches': sorted(px.midi for px in el.pitches)})\n"
        "        else:\n"
        "            tie = []\n"
        "            if el.tie is not None:\n"
        "                tie = [el.tie.type]\n"
        "            els.append({'t': 'Note', 'ql': el.quarterLength, 'pitches': [el.pitch.midi], 'tie': tie})\n"
        "    out['parts'].append({'name': pt.partName, 'clef': type(cl[0]).__name__ if cl else None,\n"
        "        'measures': len(list(pt.getElementsByClass(stream.Measure))), 'els': els})\n"
        "print(json.dumps(out))\n"
    )
    result = subprocess.run([str(Path(py).resolve()), "-c", script],
                            capture_output=True, text=True, timeout=120, cwd=str(BASE_DIR))
    assert result.returncode == 0, f"dump rc={result.returncode} stderr={result.stderr[-800:]}"
    return json.loads(result.stdout.strip().splitlines()[-1])


_SYNTH: dict = {}


def _ensure_synth() -> dict:
    """Gera 1 partitura sintética via worker e cacheia (xml, model, dump)."""
    if _SYNTH:
        return _SYNTH
    if not _notation_ok():
        pytest.skip("music21/.venv-notation indisponível")
    tmp = Path(tempfile.mkdtemp(prefix="synth_score_"))
    trans_dir = tmp / "trans"
    trans_dir.mkdir(parents=True, exist_ok=True)

    def ev(start, end, pitch, velocity=80):
        return {"start": start, "end": end, "duration": round(end - start, 6),
                "pitch": pitch, "note": "X", "velocity": velocity,
                "amplitude": 0.8, "confidence": 0.8, "strength": 0.8}

    # vocals: C4 D4 E4 F4 semínimas (0.5s = 1 beat @120) — 1 compasso 4/4
    vocals = [ev(0.00, 0.50, 60), ev(0.50, 1.00, 62), ev(1.00, 1.50, 64), ev(1.50, 2.00, 65)]
    # bass: G2 começa beat 3.5, dura 1 beat — atravessa barra -> tie
    bass = [ev(1.75, 2.25, 43)]
    # other: acorde C+E+G mesmo onset/duração
    other = [ev(0.00, 0.50, 60), ev(0.00, 0.50, 64), ev(0.00, 0.50, 67)]
    for stem, events in (("vocals", vocals), ("bass", bass), ("other", other)):
        with open(trans_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump({"file_id": "synth-test", "stem": stem,
                       "notes_count": len(events), "events": events}, f)
    out_xml = tmp / "score.musicxml"
    out_model = tmp / "score.json"
    summary = _run_worker(trans_dir, out_xml, out_model)
    model = json.loads(out_model.read_text(encoding="utf-8"))
    xml_text = out_xml.read_text(encoding="utf-8")
    dump = _dump_parsed(out_xml)
    _SYNTH.update({"tmp": tmp, "xml": out_xml, "model": model,
                   "xml_text": xml_text, "summary": summary, "dump": dump})
    return _SYNTH


# ---------------------------------------------------------------------------
# 1-3. Ambiente / info
# ---------------------------------------------------------------------------

def test_01_notation_python_found():
    py = get_notation_python()
    assert py is not None
    assert "python" in py.lower()
    assert Path(py).is_file()


def test_02_music21_importable():
    assert is_notation_available() is True
    assert get_music21_version() == "10.5.0"


def test_03_notation_info():
    from app import app
    client = TestClient(app)
    r = client.get("/api/notation/info")
    assert r.status_code == 200
    data = r.json()  # serializável
    assert data["available"] is True
    assert data["music21_version"] == "10.5.0"
    assert "python" in data
    assert data["supported_time_signatures"] == ["4/4", "3/4", "6/8"]
    assert data["supported_quantization"] == ["1/8", "1/16", "1/32"]


# ---------------------------------------------------------------------------
# 4-13. Validação de config
# ---------------------------------------------------------------------------

def test_04_invalid_tempo():
    for bad in (10, 29, 301, 500, 0, -5, "abc", "rápido"):
        with pytest.raises(ValueError, match="BPM inválido"):
            validate_score_config(bad, "4/4", "1/16", "auto")


def test_05_valid_tempo():
    assert validate_score_config(89, "4/4", "1/16", "auto")["tempo"] == 89
    assert validate_score_config(30, "4/4", "1/16", "auto")["tempo"] == 30
    assert validate_score_config(300, "3/4", "1/8", "none")["tempo"] == 300
    assert validate_score_config(None, "4/4", "1/16", "auto")["tempo"] is None


def test_06_invalid_time_signature():
    with pytest.raises(ValueError, match="[Cc]ompasso"):
        validate_score_config(120, "5/4", "1/16", "auto")


def test_07_time_signature_4_4():
    assert validate_score_config(120, "4/4", "1/16", "auto")["time_signature"] == "4/4"
    assert beats_per_measure("4/4") == 4.0


def test_08_time_signature_3_4():
    assert validate_score_config(100, "3/4", "1/16", "auto")["time_signature"] == "3/4"
    assert beats_per_measure("3/4") == 3.0


def test_09_time_signature_6_8():
    assert validate_score_config(90, "6/8", "1/8", "auto")["time_signature"] == "6/8"
    assert beats_per_measure("6/8") == 3.0


def test_10_invalid_quantization():
    with pytest.raises(ValueError, match="[Qq]uantização"):
        validate_score_config(120, "4/4", "1/64", "auto")


def test_11_quantization_1_8():
    assert validate_score_config(120, "4/4", "1/8", "auto")["quantization"] == "1/8"
    assert grid_step_beats("1/8") == 0.5


def test_12_quantization_1_16():
    assert validate_score_config(120, "4/4", "1/16", "auto")["quantization"] == "1/16"
    assert grid_step_beats("1/16") == 0.25


def test_13_quantization_1_32():
    assert validate_score_config(120, "4/4", "1/32", "auto")["quantization"] == "1/32"
    assert grid_step_beats("1/32") == 0.125


# ---------------------------------------------------------------------------
# 14-15. API: UUID / pré-condição
# ---------------------------------------------------------------------------

def test_14_invalid_uuid():
    from app import app
    client = TestClient(app)
    r = client.post("/api/score/not-a-uuid", json={"tempo": 120})
    assert r.status_code == 400
    r2 = client.get("/api/score/not-a-uuid")
    assert r2.status_code == 400
    r3 = client.get("/api/score/not-a-uuid/musicxml")
    assert r3.status_code == 400
    r4 = client.get("/api/score/status/not-a-uuid")
    assert r4.status_code == 400


def test_15_missing_transcription():
    from app import app
    client = TestClient(app)
    fid = str(uuid.uuid4())
    r = client.post(f"/api/score/{fid}", json={"tempo": 89, "time_signature": "4/4",
                                               "quantization": "1/16", "key_mode": "auto"})
    assert r.status_code == 409
    assert "Transcreva" in r.json()["detail"]


# ---------------------------------------------------------------------------
# 16-18. Armadura / enarmonia
# ---------------------------------------------------------------------------

def test_16_low_key_confidence_neutral():
    assert should_use_key(0.28) is False
    k, m, w = resolve_key_signature("G#", "major", 0.28, "auto")
    assert k is None and m is None
    assert w == "Tonalidade detectada com baixa confiança; armadura neutra utilizada."


def test_17_high_key_confidence_applied():
    assert should_use_key(0.9) is True
    assert should_use_key(0.45) is True
    k, m, w = resolve_key_signature("D", "major", 0.9, "auto")
    assert (k, m, w) == ("D", "major", None)


def test_18_gs_major_to_ab_major():
    assert normalize_key("G#", "major") == ("Ab", "major")
    assert normalize_key("D#", "major") == ("Eb", "major")
    assert normalize_key("A#", "major") == ("Bb", "major")
    assert normalize_key("C#", "major") == ("Db", "major")
    k, m, _ = resolve_key_signature("G#", "major", 0.95, "auto")
    assert (k, m) == ("Ab", "major")


# ---------------------------------------------------------------------------
# 19-21. Limpeza / merge
# ---------------------------------------------------------------------------

def _ev(start, end, pitch, amp=0.8):
    return {"start": start, "end": end, "duration": end - start, "pitch": pitch,
            "note": "X", "velocity": 80, "amplitude": amp, "confidence": amp, "strength": amp}


def test_19_invalid_events_removed():
    events = [_ev(0.0, 0.5, 60), _ev(1.0, 0.5, 62), _ev(2.0, 2.5, 200),
              _ev(3.0, 3.5, -5), _ev(4.0, 4.5, 128)]
    cleaned, stats = clean_events(events)
    assert len(cleaned) == 1
    assert cleaned[0]["pitch"] == 60
    assert stats["invalid"] == 4


def test_20_zero_duration_removed():
    events = [_ev(0.5, 0.5, 60), _ev(1.0, 1.0, 62), _ev(0.0, 0.5, 64)]
    cleaned, stats = clean_events(events)
    assert [e["pitch"] for e in cleaned] == [64]
    assert stats["invalid"] == 2


def test_21_merge_same_pitch():
    notes = [{"start": 0.0, "end": 0.48, "pitch": 60, "velocity": 80},
             {"start": 0.51, "end": 1.00, "pitch": 60, "velocity": 90},
             {"start": 2.0, "end": 2.5, "pitch": 62, "velocity": 80}]
    merged, count = merge_same_pitch(notes)
    assert count == 1
    assert len(merged) == 2
    assert merged[0]["start"] == 0.0 and merged[0]["end"] == 1.00
    assert merged[0]["velocity"] == 90


# ---------------------------------------------------------------------------
# 22-23. Overlaps monofônicos
# ---------------------------------------------------------------------------

def test_22_vocals_overlap_resolved():
    notes = [{"start": 0.0, "end": 1.2, "pitch": 60, "velocity": 80},
             {"start": 1.0, "end": 2.0, "pitch": 62, "velocity": 80}]
    resolved, stats = resolve_monophonic_overlaps(notes, grid=0.25)
    assert stats["overlaps_resolved"] == 1
    assert resolved[0]["end"] == 1.0
    assert resolved[1]["start"] == 1.0


def test_23_bass_overlap_resolved():
    notes = [{"start": 0.0, "end": 0.75, "pitch": 40, "velocity": 80},
             {"start": 0.5, "end": 1.5, "pitch": 43, "velocity": 80}]
    resolved, stats = resolve_monophonic_overlaps(notes, grid=0.25)
    assert stats["overlaps_resolved"] == 1
    assert resolved[0]["end"] == 0.5


# ---------------------------------------------------------------------------
# 24-25. Acorde other
# ---------------------------------------------------------------------------

def test_24_other_preserves_chord():
    notes = [{"start": 0.0, "end": 1.0, "pitch": 60, "velocity": 80},
             {"start": 0.0, "end": 1.0, "pitch": 64, "velocity": 85},
             {"start": 2.0, "end": 3.0, "pitch": 67, "velocity": 80}]
    items, stats = group_chords_other(notes)
    assert stats["chords"] == 1 and stats["notes"] == 1
    assert items[0]["kind"] == "chord"


def test_25_chord_c_e_g():
    notes = [{"start": 0.0, "end": 1.0, "pitch": 67, "velocity": 80},
             {"start": 0.0, "end": 1.0, "pitch": 60, "velocity": 80},
             {"start": 0.0, "end": 1.0, "pitch": 64, "velocity": 80}]
    items, _ = group_chords_other(notes)
    assert len(items) == 1
    assert items[0]["kind"] == "chord"
    assert items[0]["pitches"] == [60, 64, 67]


# ---------------------------------------------------------------------------
# 26-36. Partitura sintética via worker (cache único)
# ---------------------------------------------------------------------------

def test_26_rests_present():
    s = _ensure_synth()
    dump = s["dump"]
    bass = dump["parts"][1]
    rests = [e for e in bass["els"] if e["t"] == "Rest"]
    assert len(rests) >= 1  # leading + trailing do tie


def test_27_leading_rest():
    s = _ensure_synth()
    bass = s["dump"]["parts"][1]
    assert bass["els"][0]["t"] == "Rest"
    assert bass["els"][0]["ql"] == pytest.approx(3.5)


def test_28_tie_across_barline():
    s = _ensure_synth()
    bass_notes = [e for e in s["dump"]["parts"][1]["els"] if e["t"] == "Note"]
    assert len(bass_notes) == 2
    types = [tuple(n.get("tie", [])) for n in bass_notes]
    assert ("start",) in types and ("stop",) in types
    assert all(n["ql"] == pytest.approx(0.5) for n in bass_notes)
    assert s["xml_text"].count("<tie") >= 2


def test_29_vocal_clef_treble():
    s = _ensure_synth()
    assert s["dump"]["parts"][0]["clef"] == "TrebleClef"


def test_30_bass_clef_bass():
    s = _ensure_synth()
    assert s["dump"]["parts"][1]["clef"] == "BassClef"


def test_31_other_clef_by_median():
    assert choose_clef_other([60, 64, 67]) == "treble"
    assert choose_clef_other([36, 38, 40]) == "bass"
    s = _ensure_synth()
    assert s["dump"]["parts"][2]["clef"] == "TrebleClef"  # mediana 64


def test_32_three_parts():
    s = _ensure_synth()
    dump = s["dump"]
    assert len(dump["parts"]) == 3
    assert [p["name"] for p in dump["parts"]] == ["Vocais", "Baixo", "Outros"]


def test_33_explicit_tempo():
    s = _ensure_synth()
    assert s["dump"]["tempo"] == 120
    assert s["model"]["tempo"] == 120
    assert "<sound tempo=" in s["xml_text"] or "<metronome" in s["xml_text"]


def test_34_explicit_time_signature():
    s = _ensure_synth()
    assert s["dump"]["time_signature"] == "4/4"
    assert "<time>" in s["xml_text"]


def test_35_valid_xml():
    s = _ensure_synth()
    ET.fromstring(s["xml_text"])  # parseável ou lança
    assert s["xml"].stat().st_size > 0


def test_36_musicxml_reopens():
    s = _ensure_synth()
    dump = s["dump"]
    total_notes = sum(1 for p in dump["parts"] for e in p["els"] if e["t"] in ("Note", "Chord"))
    assert total_notes >= 6  # 4 vocais + 2 tie baixo + 1 acorde
    assert all(p["measures"] >= 1 for p in dump["parts"])


# ---------------------------------------------------------------------------
# Sintéticos S1-S4 (esperado -> resultado)
# ---------------------------------------------------------------------------

def test_synth_s1_quarter_notes():
    """BPM 120: C4/D4/E4/F4 de 0.5s -> 4 semínimas em 1 compasso 4/4."""
    s = _ensure_synth()
    vocals = [e for e in s["dump"]["parts"][0]["els"] if e["t"] == "Note"]
    assert len(vocals) == 4
    assert [n["pitches"][0] for n in vocals] == [60, 62, 64, 65]
    assert all(n["ql"] == pytest.approx(1.0) for n in vocals)
    assert s["dump"]["parts"][0]["measures"] == 1


def test_synth_s2_jitter_quantized():
    """Onsets 0.03/0.48/1.04/1.51s @120 -> grid 0/1/2/3 beats, sem figuras absurdas."""
    grid = grid_step_beats("1/16")
    got = [quantize_value(seconds_to_beats(t, 120, 0.0), grid) for t in (0.03, 0.48, 1.04, 1.51)]
    assert got == [pytest.approx(0.0), pytest.approx(1.0), pytest.approx(2.0), pytest.approx(3.0)]
    qs, qe = quantize_note(seconds_to_beats(0.03, 120, 0.0),
                           seconds_to_beats(0.50, 120, 0.0), grid)
    assert (qs, qe) == (pytest.approx(0.0), pytest.approx(1.0))


def test_synth_s3_chord_not_sequential():
    """other C+E+G mesmo onset -> 1 Chord, não 3 notas sequenciais."""
    s = _ensure_synth()
    other = s["dump"]["parts"][2]["els"]
    chords = [e for e in other if e["t"] == "Chord"]
    assert len(chords) == 1
    assert chords[0]["pitches"] == [60, 64, 67]
    assert chords[0]["ql"] == pytest.approx(1.0)


def test_synth_s4_tie_expected():
    """Nota beat 3.5 dur 1 em 4/4 -> tie atravessando barra."""
    qs, qe = quantize_note(3.5, 4.5, grid_step_beats("1/16"))
    assert (qs, qe) == (pytest.approx(3.5), pytest.approx(4.5))
    assert qe > beats_per_measure("4/4")  # ultrapassa compasso -> worker gera tie
    test_28_tie_across_barline()  # confirma no MusicXML real


# ---------------------------------------------------------------------------
# 37-40. Download / traversal / idempotência / regressão
# ---------------------------------------------------------------------------

def _stage_score_copy() -> str:
    """Copia artefatos sintéticos para scores/<uuid>/ (download + idempotência).

    Também monta transcriptions/<uuid>/ para satisfazer a pré-condição da API.
    """
    from backend.notation.score_generator import get_score_paths
    from backend.audio.transcriber import TRANSCRIPTIONS_DIR
    s = _ensure_synth()
    fid = str(uuid.uuid4())
    xml_p, model_p = get_score_paths(fid)
    xml_p.parent.mkdir(parents=True, exist_ok=True)
    model_p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(s["xml"], xml_p)
    model = dict(s["model"])
    model["file_id"] = fid
    with open(model_p, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)
    trans_target = TRANSCRIPTIONS_DIR / fid
    trans_target.mkdir(parents=True, exist_ok=True)
    for stem in ("vocals", "bass", "other"):
        shutil.copyfile(s["tmp"] / "trans" / f"{stem}.json", trans_target / f"{stem}.json")
    return fid


def _unstage_score_copy(fid: str) -> None:
    from backend.notation.score_generator import get_score_paths
    from backend.audio.transcriber import TRANSCRIPTIONS_DIR
    xml_p, model_p = get_score_paths(fid)
    shutil.rmtree(xml_p.parent, ignore_errors=True)
    shutil.rmtree(model_p.parent, ignore_errors=True)
    shutil.rmtree(TRANSCRIPTIONS_DIR / fid, ignore_errors=True)


def test_37_fileresponse_musicxml():
    from app import app
    client = TestClient(app)
    fid = _stage_score_copy()
    try:
        r = client.get(f"/api/score/{fid}/musicxml")
        assert r.status_code == 200
        assert "musicxml" in r.headers["content-type"]
        assert "partitura.musicxml" in r.headers.get("content-disposition", "")
        assert len(r.content) > 0
    finally:
        _unstage_score_copy(fid)


def test_38_path_traversal():
    from app import app
    client = TestClient(app)
    fid = _stage_score_copy()
    try:
        r = client.get(f"/api/score/{fid}/musicxml/../../etc/passwd")
        assert r.status_code in (400, 404, 422)
        r2 = client.get("/api/score/../app/musicxml")
        assert r2.status_code in (400, 404, 422)
    finally:
        _unstage_score_copy(fid)


def test_39_idempotency_config_regen():
    # Mesma config -> mesma key; config distinta -> key distinta
    k1 = config_key(89, "4/4", "1/16", "auto")
    assert config_key(89, "4/4", "1/16", "auto") == k1
    assert config_key(90, "4/4", "1/16", "auto") != k1
    assert config_key(89, "3/4", "1/16", "auto") != k1
    # API: score existente + mesma config -> already_completed
    from app import app
    client = TestClient(app)
    fid = _stage_score_copy()  # config: 120/4/4/1/16/none
    try:
        r = client.post(f"/api/score/{fid}", json={"tempo": 120, "time_signature": "4/4",
                                                   "quantization": "1/16", "key_mode": "none"})
        assert r.status_code == 200
        assert r.json()["already_completed"] is True
    finally:
        _unstage_score_copy(fid)


def test_40_etapas_1_5_regression():
    from app import app
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/api/demucs/info").status_code == 200
    assert client.get("/api/basic-pitch/info").status_code == 200
    assert client.get("/api/notation/info").status_code == 200


# ---------------------------------------------------------------------------
# Regressão bugfix: other com notas -> parte Outros não vazia no MusicXML
# ---------------------------------------------------------------------------

def _ev_sec(start, end, pitch, velocity=80, amp=0.8):
    return {"start": start, "end": end, "duration": round(end - start, 6),
            "pitch": pitch, "note": "X", "velocity": velocity,
            "amplitude": amp, "confidence": amp, "strength": amp}


def _run_custom_other(other_events, vocals_events=None, bass_events=None):
    """Worker via .venv-notation com fixture custom; retorna (xml, model, dump, xml_text)."""
    if not _notation_ok():
        pytest.skip("music21/.venv-notation indisponível")
    tmp = Path(tempfile.mkdtemp(prefix="other_fix_"))
    trans_dir = tmp / "trans"
    trans_dir.mkdir(parents=True, exist_ok=True)
    for stem, events in (("vocals", vocals_events or []),
                         ("bass", bass_events or []),
                         ("other", other_events)):
        with open(trans_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump({"file_id": "other-fix", "stem": stem,
                       "notes_count": len(events), "events": events}, f)
    out_xml = tmp / "score.musicxml"
    out_model = tmp / "score.json"
    _run_worker(trans_dir, out_xml, out_model)
    model = json.loads(out_model.read_text(encoding="utf-8"))
    xml_text = out_xml.read_text(encoding="utf-8")
    ET.fromstring(xml_text)
    dump = _dump_parsed(out_xml)
    outros = [p for p in dump["parts"] if p["name"] == "Outros"][0]
    return {"tmp": tmp, "xml": out_xml, "model": model,
            "dump": dump, "outros": outros, "xml_text": xml_text}


def test_other_monophonic_musicxml():
    """other C4 D4 E4 F4 sequenciais -> 4 semínimas em 1 compasso (reparse)."""
    r = _run_custom_other([
        _ev_sec(0.0, 0.5, 60), _ev_sec(0.5, 1.0, 62),
        _ev_sec(1.0, 1.5, 64), _ev_sec(1.5, 2.0, 65),
    ])
    notes = [e for e in r["outros"]["els"] if e["t"] == "Note"]
    assert len(notes) == 4
    assert [n["pitches"][0] for n in notes] == [60, 62, 64, 65]
    assert all(n["ql"] == pytest.approx(1.0) for n in notes)
    assert r["outros"]["measures"] == 1
    assert r["model"]["parts"]["other"]["reparsed_notes"] == 4


def test_other_chord_musicxml():
    """other C+E+G mesmo onset -> 1 Chord com 3 pitches (reparse)."""
    r = _run_custom_other([
        _ev_sec(0.0, 0.5, 60), _ev_sec(0.0, 0.5, 64), _ev_sec(0.0, 0.5, 67),
    ])
    chords = [e for e in r["outros"]["els"] if e["t"] == "Chord"]
    assert len(chords) == 1
    assert chords[0]["pitches"] == [60, 64, 67]
    assert r["model"]["parts"]["other"]["reparsed_chords"] == 1


def test_other_polyphonic_voices_musicxml():
    """C4 0-2 + E4 1-3 + G4 2-4 beats: nenhuma nota desaparece; voices 1-based."""
    r = _run_custom_other([
        _ev_sec(0.0, 1.0, 60), _ev_sec(0.5, 1.5, 64), _ev_sec(1.0, 2.0, 67),
    ])
    got = set()
    for e in r["outros"]["els"]:
        if e["t"] in ("Note", "Chord"):
            got.update(e["pitches"])
    assert {60, 64, 67} <= got
    assert "<voice>0</voice>" not in r["xml_text"]  # voz 0 quebra o MuseScore
    assert "<voice>1</voice>" in r["xml_text"]
    assert r["model"]["parts"]["other"]["voices"] >= 2


def test_other_nonempty_transcription_produces_nonempty_part():
    """Cluster com duplicatas de grade -> parte contém notas, sem pitch repetido."""
    r = _run_custom_other([
        _ev_sec(0.0, 0.5, 60), _ev_sec(0.03, 0.53, 60),  # mesma grade, mesmo pitch
        _ev_sec(0.0, 0.5, 64), _ev_sec(1.0, 1.5, 67),
    ])
    total = sum(1 for e in r["outros"]["els"] if e["t"] in ("Note", "Chord"))
    assert total >= 1
    for e in r["outros"]["els"]:
        if e["t"] == "Chord":
            assert len(e["pitches"]) == len(set(e["pitches"]))
    m = r["model"]["parts"]["other"]
    assert m["reparsed_notes"] + m["reparsed_chords"] >= 1
    assert m["written_notes"] + m["written_chords"] >= 1


def test_validation_rejects_empty_other_when_input_has_notes():
    """check_parts_nonempty: falha se parte reaberta vazia com entrada válida."""
    ok, errors, warns = check_parts_nonempty(
        {"vocals": 1, "bass": 0, "other": 5},
        {"vocals": 1, "bass": 0, "other": 5},
        {"Vocais": {"notes": 1, "chords": 0},
         "Baixo": {"notes": 0, "chords": 0},
         "Outros": {"notes": 0, "chords": 0}})
    assert ok is False
    assert any("Outros" in e for e in errors)
    # Descarte total documentado -> warning, sem falha
    ok2, _, warns2 = check_parts_nonempty(
        {"vocals": 0, "bass": 0, "other": 3},
        {"vocals": 0, "bass": 0, "other": 0},
        {"Vocais": {"notes": 0, "chords": 0},
         "Baixo": {"notes": 0, "chords": 0},
         "Outros": {"notes": 0, "chords": 0}})
    assert ok2 is True
    assert any("Outros" in w for w in warns2)
    # Parte preenchida -> ok
    ok3, _, _ = check_parts_nonempty(
        {"vocals": 0, "bass": 0, "other": 5},
        {"vocals": 0, "bass": 0, "other": 5},
        {"Vocais": {"notes": 0, "chords": 0},
         "Baixo": {"notes": 0, "chords": 0},
         "Outros": {"notes": 2, "chords": 1}})
    assert ok3 is True


# ---------------------------------------------------------------------------
# Teste real (condicional): 15s, vocals 48 / bass 34 / other 96
# ---------------------------------------------------------------------------

def test_real_score_15s():
    from backend.audio.transcriber import TRANSCRIPTIONS_DIR
    fid = "9e996782-e50c-41b8-9bf3-174e8cb99b37"
    counts = {}
    for stem in ("vocals", "bass", "other"):
        p = TRANSCRIPTIONS_DIR / fid / f"{stem}.json"
        if not p.is_file():
            pytest.skip("transcrições reais ausentes")
        counts[stem] = json.loads(p.read_text(encoding="utf-8")).get("notes_count", 0)
    assert counts == {"vocals": 48, "bass": 34, "other": 96}
    if not _notation_ok():
        pytest.skip("music21/.venv-notation indisponível")
    tmp = Path(tempfile.mkdtemp(prefix="real_score_"))
    out_xml = tmp / "score.musicxml"
    out_model = tmp / "score.json"
    # Sol# maior 28%: baixa confiança -> armadura neutra (comportamento da spec)
    py = get_notation_python()
    cmd = [str(Path(py).resolve()), str(WORKER_PATH.resolve()),
           "--file-id", fid,
           "--transcriptions-dir", str((TRANSCRIPTIONS_DIR / fid).resolve()),
           "--output-musicxml", str(out_xml.resolve()),
           "--output-model", str(out_model.resolve()),
           "--tempo", "89", "--time-signature", "4/4",
           "--quantization", "1/16", "--key-mode", "auto",
           "--key", "G#", "--mode", "major", "--key-confidence", "0.28",
           "--beat-offset", "0.0"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(BASE_DIR))
    assert result.returncode == 0, f"worker rc={result.returncode} stderr={result.stderr[-800:]}"
    model = json.loads(out_model.read_text(encoding="utf-8"))
    assert model["key"] is None  # neutra por baixa confiança
    assert any("baixa confiança" in w for w in model["warnings"])
    ET.parse(str(out_xml))
    assert out_xml.stat().st_size > 0
    # Bugfix: cada parte com transcrição deve reabrir com notas/acordes
    for stem, pname in (("vocals", "Vocais"), ("bass", "Baixo"), ("other", "Outros")):
        part = model["parts"][stem]
        assert part["reparsed_notes"] + part["reparsed_chords"] >= 1, pname
    assert model["parts"]["other"]["voices"] >= 1
    shutil.rmtree(tmp, ignore_errors=True)
