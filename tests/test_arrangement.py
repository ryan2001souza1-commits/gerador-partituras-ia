"""
Testes Etapa 7 — beat offset refinado + sopros transpositores + arranjo.

- Lógica pura (offset, instrumentos, arranjador) e API: rápidos.
- Worker .venv-notation via subprocess com cache por fixture.
- Teste real (15s, BPM 89): regenera base com offset normalizado e arranja
  Alto+Trumpet+Trombone; artefatos mantidos para teste manual no MuseScore.
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

from backend.audio.music_analysis import estimate_beat_offset
from backend.arrangement.arrangement_generator import (
    arrangement_config_key,
    validate_arrange_config,
)
from backend.arrangement.arranger import (
    apply_breathing,
    assign_roles,
    build_harmony_line,
    extract_main_melody,
    get_active_harmony,
    select_harmony_note,
    simplify_melody,
)
from backend.arrangement.instrument_definitions import (
    REGISTER_ORDER,
    SUPPORTED_ARRANGE_MODES,
    concert_to_written,
    fit_range,
    get_instrument,
    list_instruments,
    written_to_concert,
)
from backend.notation.score_generator import get_notation_python, is_notation_available

ARRANGE_WORKER = BASE_DIR / "backend" / "workers" / "arrangement_worker.py"


def _notation_ok() -> bool:
    py = get_notation_python()
    return bool(py and Path(py).is_file() and is_notation_available())


def _ev(s, e, p, v=80):
    return {"start": s, "end": e, "duration": round(e - s, 6), "pitch": p,
            "note": "X", "velocity": v, "amplitude": 0.8,
            "confidence": 0.8, "strength": 0.8}


_ARR_CACHE: dict = {}


def _run_arrange(trans_map: dict, instruments: list, mode="automatic",
                 include_originals=True, key_args=None, cache_key=None) -> dict:
    """Executa arrangement_worker (sintético). Retorna xml/model/dump/xml_text."""
    if cache_key and cache_key in _ARR_CACHE:
        return _ARR_CACHE[cache_key]
    if not _notation_ok():
        pytest.skip("music21/.venv-notation indisponível")
    tmp = Path(tempfile.mkdtemp(prefix="arr7_"))
    td = tmp / "trans"
    td.mkdir(parents=True, exist_ok=True)
    for stem in ("vocals", "bass", "other"):
        events = trans_map.get(stem, [])
        with open(td / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump({"file_id": "arr7", "stem": stem, "notes_count": len(events),
                       "events": events}, f)
    out_xml = tmp / "arrangement.musicxml"
    out_model = tmp / "arrangement.json"
    py = get_notation_python()
    ka = key_args or ["--key", "C", "--concert-mode", "major",
                      "--key-confidence", "0.9", "--key-mode", "auto"]
    cmd = [str(Path(py).resolve()), str(ARRANGE_WORKER.resolve()),
           "--file-id", "arr7", "--transcriptions-dir", str(td.resolve()),
           "--output-musicxml", str(out_xml.resolve()),
           "--output-model", str(out_model.resolve()),
           "--tempo", "120", "--time-signature", "4/4", "--quantization", "1/16",
           "--beat-offset", "0.0", "--instruments", ",".join(instruments),
           "--mode", mode, "--base-config-key", "CK"] + ka
    if include_originals:
        cmd.append("--include-original-parts")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(BASE_DIR))
    assert r.returncode == 0, f"arrange rc={r.returncode} stderr={r.stderr[-800:]}"
    model = json.loads(out_model.read_text(encoding="utf-8"))
    xml_text = out_xml.read_text(encoding="utf-8")
    ET.fromstring(xml_text)
    dump = _dump_parts(out_xml)
    res = {"tmp": tmp, "model": model, "dump": dump, "xml_text": xml_text, "xml": out_xml}
    if cache_key:
        _ARR_CACHE[cache_key] = res
    return res


def _dump_parts(musicxml_path: Path) -> dict:
    py = get_notation_python()
    script = (
        "import json\n"
        "from music21 import converter, note, chord\n"
        f"p = converter.parse(r'''{musicxml_path}''')\n"
        "out = []\n"
        "for pt in p.parts:\n"
        "    els = []\n"
        "    for el in pt.flatten().notesAndRests:\n"
        "        if isinstance(el, chord.Chord):\n"
        "            els.append({'t': 'C', 'ql': el.quarterLength,\n"
        "                      'p': sorted(px.midi for px in el.pitches)})\n"
        "        elif isinstance(el, note.Note):\n"
        "            els.append({'t': 'N', 'ql': el.quarterLength, 'p': [el.pitch.midi]})\n"
        "        else:\n"
        "            els.append({'t': 'R', 'ql': el.quarterLength, 'p': []})\n"
        "    insts = list(pt.getElementsByClass('Instrument'))\n"
        "    tr = insts[0].transposition.semitones if insts and insts[0].transposition is not None else 0\n"
        "    out.append({'name': pt.partName, 'els': els, 'transpose': tr})\n"
        "print(json.dumps(out))\n"
    )
    r = subprocess.run([str(Path(py).resolve()), "-c", script],
                       capture_output=True, text=True, timeout=120, cwd=str(BASE_DIR))
    assert r.returncode == 0, f"dump rc={r.returncode} stderr={r.stderr[-500:]}"
    return {"parts": json.loads(r.stdout.strip().splitlines()[-1])}


def _part(dump, name):
    return [p for p in dump["parts"] if p["name"] == name][0]


# ---------------------------------------------------------------------------
# Offset
# ---------------------------------------------------------------------------

def test_offset_normalizes_late_first_beat():
    period = 60.0 / 89
    off = estimate_beat_offset(4.08671201814059, 89)
    assert 0 <= off < period
    assert off == pytest.approx(0.041768, abs=1e-3)


def test_offset_small_unchanged():
    assert estimate_beat_offset(0.1, 120) == pytest.approx(0.1)
    assert estimate_beat_offset(0.0, 120) == 0.0


def test_offset_fallbacks():
    assert estimate_beat_offset(None, 120) == 0.0
    assert estimate_beat_offset(1.5, None) == 0.0
    assert estimate_beat_offset(-2.0, 120) == 0.0
    assert estimate_beat_offset(0.0, 0) == 0.0
    assert estimate_beat_offset(None, 120, beat_times=[0.3, 0.8]) == pytest.approx(0.3)
    assert estimate_beat_offset(None, 120, beat_times=[]) == 0.0


# ---------------------------------------------------------------------------
# Instrumentos
# ---------------------------------------------------------------------------

def test_instrument_definitions():
    defs = list_instruments()
    assert [d.id for d in defs] == ["alto_sax", "tenor_sax", "trumpet", "trombone", "clarinet"]
    assert {d.id: d.transposition_semitones for d in defs} == {
        "alto_sax": 9, "tenor_sax": 14, "trumpet": 2, "trombone": 0, "clarinet": 2}
    for d in defs:
        assert d.written_low < d.written_high
        assert d.written_low <= d.preferred_low <= d.preferred_high <= d.written_high
        assert d.clef in ("treble", "bass")
        assert d.family == "Sopros / Metais"
    assert get_instrument("nope") is None


def test_concert_written_roundtrip_and_fit():
    alto = get_instrument("alto_sax")
    assert concert_to_written(60, alto) == 69
    assert written_to_concert(69, alto) == 60
    trumpet = get_instrument("trumpet")
    p, adj = fit_range(90, trumpet)  # C#7 agudo -> oitavas abaixo
    assert p is not None and adj >= 2
    assert trumpet.written_low - trumpet.transposition_semitones <= p <= \
        trumpet.written_high - trumpet.transposition_semitones
    assert fit_range(10, trumpet) == (58, 4)  # resgate por oitavas
    # Faixa estreita artificial: impossível -> None (pausa + warning).
    from backend.arrangement.instrument_definitions import InstrumentDefinition
    tiny = InstrumentDefinition(id="t", name="T", short_name="T", family="F",
                                concert_key="C", transposition_semitones=0,
                                written_low=61, written_high=65,
                                preferred_low=61, preferred_high=65,
                                clef="treble", music21_instrument="Trumpet")
    assert fit_range(0, tiny)[0] is None
    assert get_instrument("trombone").clef == "bass"


def test_validate_arrange_config():
    cfg = validate_arrange_config(["alto_sax", "trumpet"], "automatic", True)
    assert cfg["instruments"] == ["alto_sax", "trumpet"]
    assert validate_arrange_config(["trumpet", "trumpet"])["instruments"] == ["trumpet"]
    with pytest.raises(ValueError):
        validate_arrange_config([])
    with pytest.raises(ValueError):
        validate_arrange_config(["sax_baritono"])
    with pytest.raises(ValueError):
        validate_arrange_config(["trumpet"], mode="free")
    k1 = arrangement_config_key(["trumpet", "alto_sax"], "automatic", True)
    assert arrangement_config_key(["alto_sax", "trumpet"], "automatic", True) == k1
    assert arrangement_config_key(["trumpet"], "automatic", True) != k1


# ---------------------------------------------------------------------------
# Arranjador puro
# ---------------------------------------------------------------------------

def test_extract_main_melody():
    notes = [{"start": 0.0, "end": 1.2, "pitch": 60, "velocity": 80},
             {"start": 1.0, "end": 2.0, "pitch": 62, "velocity": 80}]
    mel, st = extract_main_melody(notes)
    assert st["source"] == "vocals"
    assert mel[0]["end"] == 1.0
    mel2, st2 = extract_main_melody([], [{"start": 0.0, "end": 1.0,
                                          "pitches": [60, 64], "velocity": 80}])
    assert st2["source"] == "other_upper" and mel2[0]["pitch"] == 64


def test_simplify_and_breathing():
    mel = [{"start": 0.0, "end": 1.0, "pitch": 60, "velocity": 80},
           {"start": 1.0, "end": 1.2, "pitch": 62, "velocity": 80},
           {"start": 1.2, "end": 2.0, "pitch": 64, "velocity": 80}]
    simp, st = simplify_melody(mel)
    assert st["dropped"] == 1 and simp[-1]["end"] == 2.0 and len(simp) == 2
    long_line = [{"start": float(2 * i), "end": float(2 * i + 2), "pitch": 60 + (i % 5),
                  "velocity": 80} for i in range(10)]  # 10 mínimas, 20 beats
    br, bst = apply_breathing(long_line)
    assert bst["breath_adjustments"] >= 1
    assert any(float(n["end"]) - float(n["start"]) < 2.0 for n in br)


def test_harmony_selection():
    trumpet = get_instrument("trumpet")
    other = [{"start": 0.0, "end": 4.0, "pitches": [60, 64, 67], "velocity": 80}]
    assert get_active_harmony(other, 1.0) == [0, 4, 7]
    assert get_active_harmony(other, 5.0) == []
    line, st = build_harmony_line(
        [{"start": 0.0, "end": 1.0, "pitch": 72, "velocity": 80}], other, trumpet)
    assert len(line) == 1
    assert line[0]["pitch"] < 72 - 1  # abaixo da melodia, sem uníssono
    assert line[0]["pitch"] % 12 in (0, 4, 7)
    assert st["from_chord_tones"] == 1
    # Sem harmonia -> fallback 8ª/5ª, nunca 2ª menor arbitrária
    line2, st2 = build_harmony_line(
        [{"start": 0.0, "end": 1.0, "pitch": 72, "velocity": 80}], [], trumpet)
    assert line2 and line2[0]["pitch"] in (60, 65)
    assert st2["from_fallback"] == 1


def test_assign_roles():
    assert assign_roles(["alto_sax"], "automatic") == [("alto_sax", "melody")]
    assert [r for _, r in assign_roles(["trumpet", "trombone"], "automatic")] == ["melody", "harmony"]
    assert [r for _, r in assign_roles(["trumpet", "alto_sax", "tenor_sax"], "automatic")] == \
        ["melody", "harmony", "harmony"]
    four = assign_roles(["trumpet", "alto_sax", "tenor_sax", "trombone"], "automatic")
    assert [i for i, _ in four] == ["trumpet", "alto_sax", "tenor_sax", "trombone"]
    assert four[-1][1] == "low" and four[0][1] == "melody"
    assert all(r == "melody" for _, r in assign_roles(["trumpet", "trombone"], "melody"))
    assert set(SUPPORTED_ARRANGE_MODES) == {"automatic", "melody", "harmony"}


# ---------------------------------------------------------------------------
# Worker: transposição / melodia / formações
# ---------------------------------------------------------------------------

def _c_major_fixture():
    vocals = [_ev(0.0, 0.5, 60), _ev(0.5, 1.0, 62), _ev(1.0, 1.5, 64), _ev(1.5, 2.0, 65)]
    other = [_ev(0.0, 2.0, 60), _ev(0.0, 2.0, 64), _ev(0.0, 2.0, 67)]
    return {"vocals": vocals, "bass": [], "other": other}


def test_transposition_all_instruments():
    r = _run_arrange(_c_major_fixture(),
                     ["alto_sax", "tenor_sax", "trumpet", "trombone", "clarinet"],
                     mode="melody", cache_key="all5")
    assert len(r["dump"]["parts"]) == 3 + 5
    expected_written_first = {"Sax Alto em Eb": 69, "Sax Tenor em Bb": 62,
                              "Trompete em Bb": 62, "Trombone": 48, "Clarinete em Bb": 62}
    for pname, w in expected_written_first.items():
        pt = _part(r["dump"], pname)
        first = [e for e in pt["els"] if e["t"] in ("N", "C")][0]
        assert first["p"][0] == w, pname
    assert r["dump"] and _part(r["dump"], "Sax Alto em Eb")["transpose"] == -9
    assert _part(r["dump"], "Sax Tenor em Bb")["transpose"] == -14
    assert _part(r["dump"], "Trompete em Bb")["transpose"] == -2
    assert _part(r["dump"], "Clarinete em Bb")["transpose"] == -2
    assert _part(r["dump"], "Trombone")["transpose"] == 0
    assert "<transpose>" in r["xml_text"] and "<voice>0</voice>" not in r["xml_text"]
    # Concert pitch interno preservado (written - T), com contorno melódico.
    # Trombone desce a frase uma oitava inteira (48..53) em vez de quebrar.
    for pname, t, want in [("Sax Alto em Eb", 9, [60, 62, 64, 65]),
                           ("Trompete em Bb", 2, [60, 62, 64, 65]),
                           ("Trombone", 0, [48, 50, 52, 53])]:
        pt = _part(r["dump"], pname)
        got = [e["p"][0] - t for e in pt["els"] if e["t"] in ("N", "C")]
        assert got == want, pname


def test_melody_alto_sax():
    r = _run_arrange(_c_major_fixture(), ["alto_sax"], cache_key="mel1")
    assert r["model"]["instruments"][0]["role"] == "melody"
    pt = _part(r["dump"], "Sax Alto em Eb")
    got = [e["p"][0] - 9 for e in pt["els"] if e["t"] in ("N", "C")]
    assert got == [60, 62, 64, 65]


def test_two_instruments_melody_harmony():
    r = _run_arrange(_c_major_fixture(), ["trumpet", "tenor_sax"], cache_key="duo")
    roles = {i["id"]: i["role"] for i in r["model"]["instruments"]}
    assert roles == {"trumpet": "melody", "tenor_sax": "harmony"}
    mel = [e["p"][0] for e in _part(r["dump"], "Trompete em Bb")["els"] if e["t"] in ("N", "C")]
    ten = [e["p"][0] - 14 for e in _part(r["dump"], "Sax Tenor em Bb")["els"] if e["t"] in ("N", "C")]
    assert len(mel) == len(ten) == 4
    for m, t in zip(mel, ten):
        assert t < m - 1  # sem uníssono/cruzamento
        assert t % 12 in (0, 2, 4, 5, 7, 9, 11)  # diatônico em C, sem cromatismo absurdo


def test_four_instruments_no_crossing_no_empty():
    vocals = [_ev(i * 0.5, i * 0.5 + 0.5, 60 + [0, 4, 7, 12, 7, 4][i]) for i in range(6)]
    other = [_ev(0.0, 3.0, 48), _ev(0.0, 3.0, 55), _ev(0.0, 3.0, 60), _ev(0.0, 3.0, 64)]
    r = _run_arrange({"vocals": vocals, "bass": [], "other": other},
                     ["trumpet", "alto_sax", "tenor_sax", "trombone"],
                     include_originals=False, cache_key="quartet")
    assert len(r["dump"]["parts"]) == 4
    avgs = {}
    for pname, t in [("Trompete em Bb", 2), ("Sax Alto em Eb", 9),
                     ("Sax Tenor em Bb", 14), ("Trombone", 0)]:
        pt = _part(r["dump"], pname)
        sounding = [e["p"][0] - t for e in pt["els"] if e["t"] in ("N", "C")]
        assert sounding, pname  # nenhuma parte vazia
        avgs[pname] = sum(sounding) / len(sounding)
        d = get_instrument({"Trompete em Bb": "trumpet", "Sax Alto em Eb": "alto_sax",
                            "Sax Tenor em Bb": "tenor_sax", "Trombone": "trombone"}[pname])
        for e in pt["els"]:
            for wp in e["p"]:
                assert d.written_low <= wp <= d.written_high
    assert avgs["Trompete em Bb"] >= avgs["Trombone"]  # sem inversão grosseira
    assert isinstance(r["model"]["voice_crossings"], int)  # métrica registrada


def test_range_adjustment_octave_shift():
    r = _run_arrange({"vocals": [_ev(0.0, 0.5, 96), _ev(0.5, 1.0, 95)],
                      "bass": [], "other": []}, ["alto_sax"], cache_key="rangehi")
    st = r["model"]["instruments"][0]
    assert st["range_adjustments"] > 0
    assert any("oitava" in w for w in r["model"]["warnings"])
    pt = _part(r["dump"], "Sax Alto em Eb")
    d = get_instrument("alto_sax")
    for e in pt["els"]:
        for wp in e["p"]:
            assert d.written_low <= wp <= d.written_high


def test_low_confidence_key_uses_detected_harmony():
    r = _run_arrange(_c_major_fixture(), ["trumpet", "tenor_sax"],
                     key_args=["--key", "G#", "--concert-mode", "major",
                               "--key-confidence", "0.28", "--key-mode", "auto"],
                     cache_key="lowkey")
    assert r["model"]["concert_key"] is None
    assert any("baixa confiança" in w for w in r["model"]["warnings"])
    roles = {i["id"]: i["role"] for i in r["model"]["instruments"]}
    assert roles == {"trumpet": "melody", "tenor_sax": "harmony"}
    mel = [e["p"][0] for e in _part(r["dump"], "Trompete em Bb")["els"] if e["t"] in ("N", "C")]
    ten = [e["p"][0] - 14 for e in _part(r["dump"], "Sax Tenor em Bb")["els"] if e["t"] in ("N", "C")]
    assert len(mel) == len(ten) == 4
    for m, t in zip(mel, ten):
        assert t < m - 1
        assert t % 12 in (0, 2, 4, 5, 7, 9, 11)  # diatônico, sem cromatismo absurdo


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def test_arrangement_info():
    from app import app
    client = TestClient(app)
    r = client.get("/api/arrangement/info")
    assert r.status_code == 200
    data = r.json()
    assert len(data["instruments"]) == 5
    assert data["max_instruments"] == 5
    assert set(data["supported_modes"]) == {"automatic", "melody", "harmony"}


def test_arrange_validation():
    from app import app
    client = TestClient(app)
    assert client.post("/api/arrange/nope", json={"instruments": ["trumpet"]}).status_code == 400
    fid = str(uuid.uuid4())
    assert client.post(f"/api/arrange/{fid}", json={"instruments": []}).status_code == 400
    r = client.post(f"/api/arrange/{fid}", json={"instruments": ["tuba"]})
    assert r.status_code == 400
    r = client.post(f"/api/arrange/{fid}", json={"instruments": ["trumpet"], "mode": "x"})
    assert r.status_code == 400
    assert client.get("/api/arrangement/nope").status_code == 400
    assert client.get("/api/arrange/status/nope").status_code == 400


def test_arrange_requires_base_score():
    from app import app
    client = TestClient(app)
    fid = str(uuid.uuid4())
    r = client.post(f"/api/arrange/{fid}", json={"instruments": ["trumpet"]})
    assert r.status_code == 409
    assert "partitura base" in r.json()["detail"]


def _stage_base_and_trans(fid: str):
    from backend.notation.score_generator import SCORE_MODELS_DIR
    from backend.audio.transcriber import TRANSCRIPTIONS_DIR
    sm = SCORE_MODELS_DIR / fid
    sm.mkdir(parents=True, exist_ok=True)
    with open(sm / "score.json", "w", encoding="utf-8") as f:
        json.dump({"file_id": fid, "tempo": 120, "time_signature": "4/4",
                   "quantization": "1/16", "key_mode": "none", "key": None,
                   "mode": None, "key_confidence": None, "beat_offset": 0.0,
                   "config_key": "BASECK"}, f)
    td = TRANSCRIPTIONS_DIR / fid
    td.mkdir(parents=True, exist_ok=True)
    for stem in ("vocals", "bass", "other"):
        with open(td / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump({"file_id": fid, "stem": stem, "notes_count": 1,
                       "events": [_ev(0.0, 0.5, 60)]}, f)


def _unstage_all(fid: str):
    from backend.arrangement.arrangement_generator import get_arrangement_paths
    from backend.notation.score_generator import SCORE_MODELS_DIR
    from backend.audio.transcriber import TRANSCRIPTIONS_DIR
    xml_p, model_p = get_arrangement_paths(fid)
    shutil.rmtree(xml_p.parent, ignore_errors=True)
    shutil.rmtree(SCORE_MODELS_DIR / fid, ignore_errors=True)
    shutil.rmtree(TRANSCRIPTIONS_DIR / fid, ignore_errors=True)


def test_arrange_queued_single_and_idempotency_and_download():
    from unittest.mock import AsyncMock, patch
    from backend.arrangement.arrangement_job_manager import clear_arrangement_jobs
    from app import app
    clear_arrangement_jobs()
    client = TestClient(app)
    fid = str(uuid.uuid4())
    _stage_base_and_trans(fid)
    try:
        with patch("app._run_arrange_job", new=AsyncMock()):
            r = client.post(f"/api/arrange/{fid}",
                            json={"instruments": ["trumpet"], "mode": "automatic",
                                  "include_original_parts": True})
            assert r.status_code == 200
            assert r.json()["status"] == "queued"
            r2 = client.post(f"/api/arrange/{fid}", json={"instruments": ["trumpet"]})
            assert r2.status_code == 409
    finally:
        clear_arrangement_jobs()

    # Idempotência: arranjo existente + mesma config/base -> already_completed.
    from backend.arrangement.arrangement_generator import get_arrangement_paths
    xml_p, model_p = get_arrangement_paths(fid)
    xml_p.parent.mkdir(parents=True, exist_ok=True)
    xml_p.write_bytes(b"<score-partwise />")
    with open(model_p, "w", encoding="utf-8") as f:
        json.dump({"file_id": fid, "config_key": arrangement_config_key(
            ["trumpet"], "automatic", True), "base_config_key": "BASECK",
            "instruments": []}, f)
    try:
        r = client.post(f"/api/arrange/{fid}",
                        json={"instruments": ["trumpet"], "mode": "automatic",
                              "include_original_parts": True})
        assert r.status_code == 200
        assert r.json()["already_completed"] is True
        r = client.get(f"/api/arrangement/{fid}")
        assert r.status_code == 200 and r.json()["available"] is True
        r = client.get(f"/api/arrangement/{fid}/musicxml")
        assert r.status_code == 200
        assert "musicxml" in r.headers["content-type"]
        assert "arranjo.musicxml" in r.headers.get("content-disposition", "")
        assert client.get("/api/arrangement/../x/musicxml").status_code in (400, 404, 422)
    finally:
        _unstage_all(fid)
        clear_arrangement_jobs()


# ---------------------------------------------------------------------------
# Teste real: 15s, BPM 89 — Alto + Trompete + Trombone
# ---------------------------------------------------------------------------

def test_real_arrangement_winds():
    from backend.audio.transcriber import TRANSCRIPTIONS_DIR
    from backend.notation.score_generator import (
        SCORE_MODELS_DIR, generate_score_async, get_score_paths)
    from backend.arrangement.arrangement_generator import (
        generate_arrangement_async, get_arrangement_paths)
    fid = "aefde683-36e3-40d2-a296-a2b66f81740f"
    for stem in ("vocals", "bass", "other"):
        p = TRANSCRIPTIONS_DIR / fid / f"{stem}.json"
        if not p.is_file():
            pytest.skip("transcrições reais ausentes")
    if not _notation_ok():
        pytest.skip("music21/.venv-notation indisponível")
    import asyncio
    # 1) Base regenerada com offset normalizado (Etapa 7).
    for p in get_score_paths(fid):
        if p.is_file():
            p.unlink()
    base = asyncio.run(generate_score_async(
        fid, tempo=89, time_signature="4/4", quantization="1/16", key_mode="auto"))
    period = 60.0 / 89
    assert 0 <= base["beat_offset"] < period
    assert base["beat_offset"] == pytest.approx(0.041768, abs=1e-3)
    # 2) Arranjo real (artefatos mantidos p/ MuseScore manual).
    for p in get_arrangement_paths(fid):
        if p.is_file():
            p.unlink()
    model = asyncio.run(generate_arrangement_async(
        fid, ["alto_sax", "trumpet", "trombone"], mode="automatic",
        include_original_parts=True))
    assert model["tempo"] == 89
    assert len(model["instruments"]) == 3
    roles = {i["id"]: i["role"] for i in model["instruments"]}
    assert roles == {"trumpet": "melody", "alto_sax": "harmony", "trombone": "harmony"}
    for inst in model["instruments"]:
        assert inst["notes_count"] > 0, inst["id"]
    assert any("baixa confiança" in w for w in model["warnings"])
    xml_p, _ = get_arrangement_paths(fid)
    assert xml_p.stat().st_size > 0
    ET.parse(str(xml_p))
