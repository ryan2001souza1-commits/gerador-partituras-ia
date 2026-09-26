"""
Testes Etapa 8 — bateria/percussão + dinâmicas + estilos.

- DSP puro (classificação/quantização) e API: rápidos, sem music21.
- Worker .venv-notation via subprocess (cache por fixture).
- Teste real condicional (stem drums existente).
"""
import json
import shutil
import subprocess
import tempfile
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parents[1]
SR = 22050

from backend.drums import drum_utils as DU
from backend.drums.drum_utils import (
    DRUM_CLASSES, DRUM_SR, DRUM_VERSION, GM_MAP,
    classify_hit, detect_onsets, extract_features, transcribe_drums,
)
from backend.arrangement.styles import (
    ARRANGEMENT_STYLES, apply_style_to_drums, get_style, validate_style,
)
from backend.notation.score_generator import get_notation_python, is_notation_available

ARRANGE_WORKER = BASE_DIR / "backend" / "workers" / "arrangement_worker.py"


def _notation_ok() -> bool:
    py = get_notation_python()
    return bool(py and Path(py).is_file() and is_notation_available())


# ---------------------------------------------------------------------------
# Síntese determinística
# ---------------------------------------------------------------------------

def _kick(dur=0.4):
    t = np.arange(int(dur * SR)) / SR
    f = 55 + 40 * np.exp(-t * 30)
    return 0.9 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)


def _snare(dur=0.3):
    t = np.arange(int(dur * SR)) / SR
    rng = np.random.default_rng(7)
    try:
        from scipy import signal
        n = signal.sosfilt(signal.butter(4, 6000, btype="lowpass", fs=SR, output="sos"),
                           rng.standard_normal(len(t)))
    except Exception:
        n = rng.standard_normal(len(t))
    body = (np.sin(2 * np.pi * 190 * t) + 0.5 * np.sin(2 * np.pi * 250 * t)) * np.exp(-t * 25)
    return 0.55 * n * np.exp(-t * 28) + 0.45 * body


def _hat(dur=0.06, seed=3, decay=90.0):
    t = np.arange(int(dur * SR)) / SR
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(len(t))
    try:
        from scipy import signal
        n = signal.sosfilt(signal.butter(4, 7000, btype="highpass", fs=SR, output="sos"), n)
    except Exception:
        pass
    return 0.6 * n * np.exp(-t * decay)


def _crash(dur=1.0):
    t = np.arange(int(dur * SR)) / SR
    rng = np.random.default_rng(11)
    return 0.7 * rng.standard_normal(len(t)) * np.exp(-t * 4)


def _tom(freq=150, dur=0.4):
    t = np.arange(int(dur * SR)) / SR
    rng = np.random.default_rng(21)
    click = rng.standard_normal(len(t)) * np.exp(-t * 300) * 0.15
    return (0.7 * np.sin(2 * np.pi * freq * t) * np.exp(-t * 10)
            + 0.25 * np.sin(2 * np.pi * freq * 2.02 * t) * np.exp(-t * 14) + click)


def _classify_one(y):
    S, freqs, hop = DU._stft_once(y, SR)
    feats = DU.extract_features(y, SR, 0.01, S=S, freqs=freqs, hop=hop)
    return DU.classify_hit(feats, 0.9, y=y, sr=SR, onset_time=0.01)


def _save_wav(y, path):
    import soundfile as sf
    sf.write(str(path), np.asarray(y, dtype=np.float32), SR)


# ---------------------------------------------------------------------------
# API info/validação/pré-condição
# ---------------------------------------------------------------------------

def test_drums_info():
    from app import app
    client = TestClient(app)
    r = client.get("/api/drums/info")
    assert r.status_code == 200
    data = r.json()
    assert data["available"] is True
    assert data["method"] == "spectral-onset"
    assert data["sample_rate"] == 22050
    assert set(["kick", "snare", "closed_hihat"]) <= set(data["classes"])
    assert data["version"] == DRUM_VERSION
    assert "timeout_seconds" in data


def test_drums_invalid_uuid():
    from app import app
    client = TestClient(app)
    assert client.post("/api/drums/nope", json={}).status_code == 400
    assert client.get("/api/drums/nope").status_code == 400
    assert client.get("/api/drums/status/nope").status_code == 400
    assert client.get("/api/drums/../x").status_code in (400, 404, 422)


def test_drums_requires_stems():
    from app import app
    client = TestClient(app)
    fid = str(uuid.uuid4())
    r = client.post(f"/api/drums/{fid}", json={})
    assert r.status_code == 409
    assert "Separe os instrumentos" in r.json()["detail"]


def test_drums_gm_map():
    assert GM_MAP == {"kick": 36, "snare": 38, "closed_hihat": 42, "open_hihat": 46,
                      "crash": 49, "tom_low": 45, "tom_mid": 47, "tom_high": 50}
    assert set(DRUM_CLASSES) >= {"kick", "snare", "closed_hihat", "unknown_percussion"}


# ---------------------------------------------------------------------------
# Classificação sintética
# ---------------------------------------------------------------------------

def test_kick_synth():
    cls, conf, _ = _classify_one(_kick())
    assert cls == "kick"
    assert conf >= 0.5


def test_snare_synth():
    cls, conf, _ = _classify_one(_snare())
    assert cls == "snare"
    assert conf >= 0.4


def test_hihat_synth():
    cls, conf, _ = _classify_one(_hat())
    assert cls in ("closed_hihat",)
    assert conf >= 0.5


def test_open_hihat_synth():
    cls, conf, _ = _classify_one(_hat(dur=0.5, seed=5, decay=4.0))
    assert cls == "open_hihat"


def test_crash_synth():
    cls, conf, _ = _classify_one(_crash())
    assert cls == "crash"


def test_tom_synth():
    assert _classify_one(_tom(110))[0] == "tom_low"
    assert _classify_one(_tom(180))[0] == "tom_mid"
    assert _classify_one(_tom(260))[0] == "tom_high"


def test_simultaneous_kick_hat():
    y = np.zeros(int(1.0 * SR))
    k = _kick(0.4)
    h = _hat()
    off = int(0.1 * SR)
    y[off:off + len(k)] += k
    y[off:off + len(h)] += h
    events, stats = transcribe_drums(y, SR, tempo=120.0, beat_offset=0.0)
    by_beat = {}
    for e in events:
        by_beat.setdefault(e["beat"], set()).add(e["instrument"])
    assert any({"kick", "closed_hihat"} <= v for v in by_beat.values())
    assert stats["simultaneous_hits"] >= 1


def test_dedupe_flam():
    from backend.drums.drum_utils import _dedupe_events
    base = {"strength": 0.9, "instrument": "kick", "confidence": 0.8, "decay_ms": 100.0}
    a = dict(base, time=0.100)
    b = dict(base, time=0.115, strength=0.7)
    out, merged = _dedupe_events([a, b])
    assert len(out) == 1 and merged == 1 and out[0]["strength"] == 0.9
    # End-to-end: flam de kicks curtos -> um único hit de kick.
    y = np.zeros(int(1.0 * SR))
    k1 = _kick(0.09)
    k2 = _kick(0.09) * 0.7
    o1, o2 = int(0.2 * SR), int(0.22 * SR)
    y[o1:o1 + len(k1)] += k1
    y[o2:o2 + len(k2)] += k2
    events, _ = transcribe_drums(y, SR, tempo=120.0, beat_offset=0.0)
    kicks = [e for e in events if e["instrument"] == "kick"]
    assert len(kicks) == 1


def test_grid_bpm120_jitter():
    import random
    rng = random.Random(42)
    y = np.zeros(int(2.6 * SR))
    for b in range(1, 5):
        t = max(0.02, b * 0.5 + rng.uniform(-0.03, 0.03))
        k = _kick(0.3)
        i0 = int(t * SR)
        y[i0:i0 + len(k)] += k
    events, _ = transcribe_drums(y, SR, tempo=120.0, beat_offset=0.0)
    kicks = sorted(e["beat"] for e in events if e["instrument"] == "kick")
    assert kicks == [pytest.approx(1.0), pytest.approx(2.0),
                     pytest.approx(3.0), pytest.approx(4.0)]


def test_68_compound():
    y = np.zeros(int(2.0 * SR))
    for i in range(6):  # 6 colcheias em 6/8 a 120bpm (beat=0.5s por colcheia? usa grid 1/8)
        h = _hat()
        i0 = int(i * 0.25 * SR)
        y[i0:i0 + len(h)] += h
    events, _ = transcribe_drums(y, SR, tempo=120.0, beat_offset=0.0, time_signature="6/8")
    assert events
    assert all(abs(e["beat"] * 2 - round(e["beat"] * 2)) < 1e-6 for e in events)


def test_ghost_natural_vs_detailed():
    y = np.zeros(int(1.5 * SR))
    k = _kick(0.3)
    y[:len(k)] += k
    g = _kick(0.2) * 0.05
    off = int(0.75 * SR)
    y[off:off + len(g)] += g
    ev_n, st_n = transcribe_drums(y, SR, tempo=120.0, beat_offset=0.0, profile="natural")
    ev_d, st_d = transcribe_drums(y, SR, tempo=120.0, beat_offset=0.0, profile="detailed")
    assert st_n["ghost_notes_removed"] >= 0
    assert len(ev_d) >= len(ev_n)  # detailed preserva mais
    assert all(e["confidence"] is not None for e in ev_n)


def test_unknown_does_not_crash():
    rng = np.random.default_rng(99)
    y = (rng.standard_normal(int(0.5 * SR)) * 0.02).astype(float)
    events, stats = transcribe_drums(y, SR, tempo=100.0, beat_offset=0.0)
    assert isinstance(events, list) and isinstance(stats, dict)
    assert stats["raw_onsets"] >= 0


# ---------------------------------------------------------------------------
# Job flow + idempotência + GET
# ---------------------------------------------------------------------------

def _stage_drum_wav(fid, y=None, sr=SR):
    from backend.audio.stem_separator import STEMS_DIR
    d = STEMS_DIR / fid
    d.mkdir(parents=True, exist_ok=True)
    _save_wav(_kick() if y is None else y, d / "drums.wav")
    return d / "drums.wav"


def test_drum_job_queued_and_single():
    from unittest.mock import AsyncMock, patch
    from backend.drums.drum_job_manager import clear_drum_jobs
    from app import app
    clear_drum_jobs()
    client = TestClient(app)
    fid = str(uuid.uuid4())
    _stage_drum_wav(fid)
    try:
        with patch("app._run_drum_job", new=AsyncMock()):
            r = client.post(f"/api/drums/{fid}", json={})
            assert r.status_code == 200
            assert r.json()["status"] == "queued"
            assert "job_id" in r.json()
            r2 = client.post(f"/api/drums/{fid}", json={})
            assert r2.status_code == 409
    finally:
        clear_drum_jobs()
        import shutil
        from backend.audio.stem_separator import STEMS_DIR
        shutil.rmtree(STEMS_DIR / fid, ignore_errors=True)


def test_drum_idempotency_and_get():
    from backend.drums.drum_job_manager import clear_drum_jobs
    from backend.drums.drum_transcriber import (
        drum_config_key, get_drums_json_path)
    from app import app
    clear_drum_jobs()
    client = TestClient(app)
    fid = str(uuid.uuid4())
    _stage_drum_wav(fid)
    try:
        p = get_drums_json_path(fid)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"file_id": fid, "events": [], "stats": {},
                       "config_key": drum_config_key(fid, "natural", "4/4")}, f)
        r = client.post(f"/api/drums/{fid}", json={})
        assert r.status_code == 200
        assert r.json()["already_completed"] is True
        r = client.get(f"/api/drums/{fid}")
        assert r.status_code == 200
        assert r.json()["available"] is True
    finally:
        clear_drum_jobs()
        import shutil
        from backend.audio.stem_separator import STEMS_DIR
        from backend.drums.drum_transcriber import DRUMS_DIR
        shutil.rmtree(STEMS_DIR / fid, ignore_errors=True)
        shutil.rmtree(DRUMS_DIR / fid, ignore_errors=True)


def test_drum_status_flow_real():
    """Transcrição real via API (segundos) + polling do status."""
    import time
    from backend.drums.drum_job_manager import clear_drum_jobs, get_drum_job
    from app import app
    clear_drum_jobs()
    # Context manager: mantém o portal do TestClient vivo para que o
    # background job (asyncio.create_task) rode até completar.
    with TestClient(app) as client:
        fid = str(uuid.uuid4())
        y = np.zeros(int(2.0 * SR))
        for b, gen in [(0.0, _kick), (0.5, _snare), (1.0, _kick), (1.5, _snare)]:
            s = gen()
            i0 = int(b * SR)
            y[i0:i0 + len(s)] += s
        _stage_drum_wav(fid, y)
        try:
            r = client.post(f"/api/drums/{fid}", json={})
            assert r.status_code == 200
            job_id = r.json()["job_id"]
            deadline = time.time() + 60
            last = None
            while time.time() < deadline:
                rs = client.get(f"/api/drums/status/{job_id}")
                assert rs.status_code == 200
                last = rs.json()["status"]
                if last in ("completed", "failed"):
                    break
                time.sleep(0.5)
            assert last == "completed"
            job = get_drum_job(job_id)
            assert job and job.results
        finally:
            clear_drum_jobs()
            import shutil
            from backend.audio.stem_separator import STEMS_DIR
            from backend.drums.drum_transcriber import DRUMS_DIR
            shutil.rmtree(STEMS_DIR / fid, ignore_errors=True)
            shutil.rmtree(DRUMS_DIR / fid, ignore_errors=True)


# ---------------------------------------------------------------------------
# Worker: MusicXML de bateria + dinâmicas + estilos
# ---------------------------------------------------------------------------

_CACHE_A: dict = {}


def _mini_trans(tmp, vocals=None, bass=None, other=None):
    td = tmp / "trans"
    td.mkdir(parents=True, exist_ok=True)

    def ev(s, e, p):
        return {"start": s, "end": e, "duration": e - s, "pitch": p, "note": "X",
                "velocity": 80, "amplitude": 0.8, "confidence": 0.8, "strength": 0.8}
    v = vocals if vocals is not None else [ev(0.0, 0.5, 60), ev(0.5, 1.0, 62),
                                           ev(1.0, 1.5, 64), ev(1.5, 2.0, 65)]
    b = bass if bass is not None else [ev(0.0, 1.0, 40), ev(1.0, 2.0, 43)]
    o = other if other is not None else [ev(0.0, 2.0, 48), ev(0.0, 2.0, 55), ev(0.0, 2.0, 60)]
    for stem, events in (("vocals", v), ("bass", b), ("other", o)):
        with open(td / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump({"file_id": "d8", "stem": stem, "notes_count": len(events),
                       "events": events}, f)
    return td


def _drums_json(tmp, events):
    p = tmp / "drums.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"file_id": "d8", "events": events, "stats": {}}, f)
    return p


def _run_arrange8(trans_map_extra=None, drum_events=None, extra_args=None, cache_key=None):
    if cache_key and cache_key in _CACHE_A:
        return _CACHE_A[cache_key]
    if not _notation_ok():
        pytest.skip("music21/.venv-notation indisponível")
    tmp = Path(tempfile.mkdtemp(prefix="arr8_"))
    td = _mini_trans(tmp)
    cmd = [str(Path(get_notation_python()).resolve()),
           str(ARRANGE_WORKER.resolve()),
           "--file-id", "d8", "--transcriptions-dir", str(td.resolve()),
           "--output-musicxml", str((tmp / "arr.musicxml").resolve()),
           "--output-model", str((tmp / "arr.json").resolve()),
           "--tempo", "120", "--time-signature", "4/4", "--quantization", "1/16",
           "--key-mode", "none", "--beat-offset", "0.0",
           "--instruments", "trumpet", "--mode", "automatic",
           "--base-config-key", "CK", "--cleanup-profile", "detailed",
           "--include-original-parts"]
    if drum_events is not None:
        dp = _drums_json(tmp, drum_events)
        cmd += ["--include-drums", "--drums-json", str(dp.resolve())]
    if extra_args:
        cmd += extra_args
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(BASE_DIR))
    assert r.returncode == 0, f"arr8 rc={r.returncode} stderr={r.stderr[-800:]}"
    model = json.loads((tmp / "arr.json").read_text(encoding="utf-8"))
    xml_text = (tmp / "arr.musicxml").read_text(encoding="utf-8")
    ET.fromstring(xml_text)
    res = {"model": model, "xml_text": xml_text, "xml": tmp / "arr.musicxml"}
    if cache_key:
        _CACHE_A[cache_key] = res
    return res


def _dev(start, end, inst, conf=0.8, strength=0.7):
    return {"time": start, "quantized_time": start, "beat": start / 0.5,
            "instrument": inst, "confidence": conf, "strength": strength,
            "original_time": start, "timing_error_ms": 0.0}


def _dump_parts(xml_path):
    py = get_notation_python()
    script = (
        "import json\n"
        "from music21 import converter, note\n"
        f"p = converter.parse(r'''{xml_path}''')\n"
        "out = []\n"
        "for pt in p.parts:\n"
        "    seq = []\n"
        "    for el in pt.flatten().notesAndRests:\n"
        "        t = type(el).__name__\n"
        "        if t == 'Unpitched':\n"
        "            seq.append(('U', el.displayStep, el.displayOctave))\n"
        "        elif t == 'Rest':\n"
        "            seq.append(('R',))\n"
        "        elif t == 'Chord':\n"
        "            seq.append(('C', sorted(px.midi for px in el.pitches)))\n"
        "        else:\n"
        "            seq.append(('N', el.pitch.midi))\n"
        "    cls = [type(c).__name__ for c in pt.recurse().getElementsByClass('Clef')]\n"
        "    out.append({'name': pt.partName, 'seq': seq, 'clefs': cls})\n"
        "print(json.dumps(out))\n"
    )
    r = subprocess.run([str(Path(py).resolve()), "-c", script],
                       capture_output=True, text=True, timeout=120, cwd=str(BASE_DIR))
    assert r.returncode == 0, f"dump rc={r.returncode} stderr={r.stderr[-500:]}"
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_musicxml_drums_reparse():
    evs = [_dev(0.0, 0.0, "kick"), _dev(0.0, 0.0, "closed_hihat"),
           _dev(0.5, 0.5, "snare"), _dev(0.5, 0.5, "closed_hihat"),
           _dev(1.0, 1.0, "kick")]
    r = _run_arrange8(drum_events=evs, cache_key="drums_basic")
    assert "<voice>0</voice>" not in r["xml_text"]
    assert "<voice>1</voice>" in r["xml_text"] and "<voice>2</voice>" in r["xml_text"]
    parts = _dump_parts(r["xml"])
    names = [p["name"] for p in parts]
    assert "Bateria" in names
    bat = [p for p in parts if p["name"] == "Bateria"][0]
    assert any(c == "PercussionClef" for c in bat["clefs"])
    ups = [e for e in bat["seq"] if e[0] == "U"]
    assert len(ups) >= 5  # nenhuma voz vazia: kick+hat+snare+hat+kick
    assert r["model"]["drums"] and r["model"]["drums"]["part_built"] is True


def test_dynamics_and_articulations_present():
    # Use unique cache key to avoid stale results from previous runs
    import uuid
    evs = [_dev(0.0, 0.0, "kick", strength=0.9), _dev(0.5, 0.5, "snare", strength=0.9)]
    r = _run_arrange8(drum_events=evs, cache_key=f"drums_dyn_{uuid.uuid4().hex[:8]}")
    # music21 exporta <dynamics default-x=...> (com atributos), não <dynamics> puro
    assert "<dynamics" in r["xml_text"]
    assert "<articulations>" in r["xml_text"]
    dyn = r["model"]["dynamics_stats"]
    assert dyn["dynamics_marks"] >= 1
    assert r["model"]["dynamics"] == "automatic"


def test_dynamics_none_skips():
    r = _run_arrange8(drum_events=None, extra_args=["--dynamics", "none"], cache_key=None)
    assert "<dynamics" not in r["xml_text"]
    assert r["model"]["dynamics_stats"]["dynamics_marks"] == 0


def test_styles_all_valid_and_differ():
    outs = {}
    for style in ["automatic", "pop", "rock", "ballad", "brass_band"]:
        evs = [_dev(i * 0.25, i * 0.25, "closed_hihat", strength=0.5) for i in range(8)]
        evs += [_dev(0.0, 0.0, "kick", strength=0.9), _dev(0.5, 0.5, "snare", strength=0.9)]
        r = _run_arrange8(drum_events=evs, extra_args=["--arrangement-style", style],
                          cache_key=f"style_{style}")
        assert r["model"]["arrangement_style"] == style
        outs[style] = r["xml_text"].count("<unpitched")
    assert outs["ballad"] <= outs["rock"]  # balada filtra hats; rock mantém
    assert validate_style("rock") == "rock"
    with pytest.raises(ValueError):
        validate_style("samba")
    assert get_style("automatic")["breath_mult"] == 1.0


def test_winds_preserved_with_drums():
    evs = [_dev(0.0, 0.0, "kick"), _dev(0.5, 0.5, "snare")]
    with_drums = _run_arrange8(drum_events=evs, cache_key="winds_dr")
    without = _run_arrange8(drum_events=None, cache_key="winds_no")
    for pname in ("Trompete em Bb",):
        a = [p for p in _dump_parts(with_drums["xml"]) if p["name"] == pname][0]
        b = [p for p in _dump_parts(without["xml"]) if p["name"] == pname][0]
        na = [e[1] for e in a["seq"] if e[0] in ("N", "C")]
        nb = [e[1] for e in b["seq"] if e[0] in ("N", "C")]
        assert na == nb


# ---------------------------------------------------------------------------
# Caso real (condicional)
# ---------------------------------------------------------------------------

def test_real_drums_aefde683():
    from backend.audio.stem_separator import STEMS_DIR
    from backend.drums.drum_transcriber import DRUMS_DIR
    fid = "aefde683-36e3-40d2-a296-a2b66f81740f"
    wav = STEMS_DIR / fid / "drums.wav"
    if not wav.is_file():
        pytest.skip("stem drums ausente")
    import asyncio
    from backend.drums.drum_transcriber import transcribe_drums_sync
    from backend.notation.score_generator import get_music_context
    ctx = get_music_context(fid)
    tempo = ctx.get("tempo") or 89
    import librosa
    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    from backend.drums.drum_utils import transcribe_drums
    events, stats = transcribe_drums(y, sr, tempo=float(tempo),
                                     beat_offset=float(ctx.get("beat_offset") or 0.0),
                                     time_signature="4/4", profile="natural")
    print(f"\nREAL drums: raw={stats['raw_onsets']} classified={stats['classified_events']} "
          f"k={stats['kick_count']} s={stats['snare_count']} "
          f"chh={stats['closed_hihat_count']} ohh={stats['open_hihat_count']} "
          f"cr={stats['crash_count']} tom={stats['tom_low_count'] + stats['tom_mid_count'] + stats['tom_high_count']} "
          f"disc={stats['discarded_events']} conf={stats['mean_confidence']} "
          f"err={stats['quantization_error_mean_ms']}ms p95={stats['quantization_error_p95_ms']}ms")
    assert stats["kick_count"] + stats["snare_count"] > 0
    assert stats["mean_confidence"] >= 0.3
