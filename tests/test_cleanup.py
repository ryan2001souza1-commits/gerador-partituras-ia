"""
Testes do refinamento musical/auditivo (perfis natural/detailed).

- Puramente determinístico; sem music21 no processo principal.
- Worker .venv-notation via subprocess (1 fixture ruidosa por perfil, cache).
- 153 testes anteriores devem continuar passando (detailed = bit-idêntico).
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

from backend.musical.cleanup import (
    CLEANUP_PROFILES,
    adaptive_quantize,
    apply_breathing_v2,
    block_rhythm,
    clutter_score,
    event_strength,
    get_window_harmony,
    line_stats,
    phrase_boundaries,
    reduce_harmony_rhythm,
    refine_other_accompaniment,
    remove_outliers,
    resolve_mono_natural,
    simplify_chord,
    smooth_melody,
    sustain_merge,
    triage_short_notes,
    trio_voicing,
    validate_cleanup_profile,
)
from backend.notation.score_generator import get_notation_python, is_notation_available

NOTATION_WORKER = BASE_DIR / "backend" / "workers" / "notation_worker.py"
ARRANGE_WORKER = BASE_DIR / "backend" / "workers" / "arrangement_worker.py"


def _notation_ok() -> bool:
    py = get_notation_python()
    return bool(py and Path(py).is_file() and is_notation_available())


def _ev(s, e, p, v=80, amp=0.8):
    return {"start": s, "end": e, "duration": round(e - s, 6), "pitch": p,
            "note": "X", "velocity": v, "amplitude": amp,
            "confidence": amp, "strength": amp}


def _bn(start, end, pitch, velocity=80, strength=0.8):
    return {"start": start, "end": end, "pitch": pitch,
            "velocity": velocity, "strength": strength}


# ---------------------------------------------------------------------------
# Perfis
# ---------------------------------------------------------------------------

def test_profiles_validate():
    assert validate_cleanup_profile("natural") == "natural"
    assert validate_cleanup_profile("detailed") == "detailed"
    assert set(CLEANUP_PROFILES) == {"natural", "detailed"}
    with pytest.raises(ValueError):
        validate_cleanup_profile("ultra")


def test_api_rejects_bad_profile():
    from app import app
    client = TestClient(app)
    fid = str(uuid.uuid4())
    r = client.post(f"/api/score/{fid}", json={"cleanup_profile": "ultra"})
    # 400 (perfil) — ou 409/400 por UUID válido sem transcrição; perfil checado antes
    assert r.status_code in (400, 409)
    if r.status_code == 400:
        assert "cleanup_profile" in r.json()["detail"]
    r = client.post(f"/api/arrange/{fid}", json={"instruments": ["trumpet"],
                                                 "cleanup_profile": "x"})
    assert r.status_code == 400
    assert "cleanup_profile" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Quantização adaptativa
# ---------------------------------------------------------------------------

def test_adaptive_jitter_snaps_to_8ths():
    notes = [_bn(0.20, 1.0, 60), _bn(1.18, 2.0, 62), _bn(2.33, 3.0, 64)]
    out, st = adaptive_quantize(notes, 0.25, "natural")
    assert [n["start"] for n in out] == [pytest.approx(0.0), pytest.approx(1.0),
                                         pytest.approx(2.5)]
    assert st["rhythmic_simplifications"] >= 2
    # Detailed preserva grade fina exata.
    out_d, _ = adaptive_quantize(notes, 0.25, "detailed")
    assert [n["start"] for n in out_d] == [pytest.approx(0.25), pytest.approx(1.25),
                                           pytest.approx(2.25)]


def test_adaptive_keeps_real_16ths():
    # Duas notas distintas no mesmo slot de 1/8 -> evidência -> mantém 1/16.
    notes = [_bn(0.0, 0.25, 60), _bn(0.25, 0.5, 62), _bn(1.0, 2.0, 64)]
    out, st = adaptive_quantize(notes, 0.25, "natural")
    assert out[0]["start"] == pytest.approx(0.0)
    assert out[1]["start"] == pytest.approx(0.25)
    assert st["kept_16ths"] >= 2


def test_adaptive_no_32_in_natural():
    notes = [_bn(0.13, 0.4, 60), _bn(0.5, 1.0, 62)]
    out, _ = adaptive_quantize(notes, 0.125, "natural")
    for n in out:
        assert abs(n["start"] * 2 - round(n["start"] * 2)) < 1e-9
        assert abs(n["end"] * 2 - round(n["end"] * 2)) < 1e-9


# ---------------------------------------------------------------------------
# Notas curtas: MERGE / PRESERVAR / REMOVER
# ---------------------------------------------------------------------------

def test_short_merge_same_pitch():
    notes = [_bn(0.0, 1.0, 60), _bn(1.02, 1.1, 60, strength=0.7), _bn(1.1, 2.0, 60)]
    out, st = triage_short_notes(notes, 0.25, "natural")
    assert st["short_notes_merged"] >= 1
    assert len(out) == 2  # curta absorvida; C4 contínuo 0 -> 2.0
    assert out[0]["end"] == pytest.approx(out[1]["start"])
    assert (out[0]["start"], out[1]["end"]) == pytest.approx((0.0, 2.0))


def test_ghost_note_removed():
    notes = [_bn(0.0, 1.0, 60), _bn(1.0, 1.05, 85, strength=0.1), _bn(1.05, 2.0, 62)]
    out, st = triage_short_notes(notes, 0.25, "natural")
    assert st["short_notes_removed"] == 1
    assert [n["pitch"] for n in out] == [60, 62]


def test_passing_note_preserved():
    notes = [_bn(0.0, 1.0, 60), _bn(1.0, 1.2, 62, strength=0.8), _bn(1.2, 2.0, 64)]
    out, st = triage_short_notes(notes, 0.25, "natural")
    assert [n["pitch"] for n in out] == [60, 62, 64]
    assert st["short_notes_removed"] == 0


def test_triage_detailed_passthrough():
    notes = [_bn(0.0, 1.0, 60), _bn(1.0, 1.05, 85, strength=0.1)]
    out, st = triage_short_notes(notes, 0.25, "detailed")
    assert len(out) == 2 and st["short_notes_removed"] == 0


# ---------------------------------------------------------------------------
# Smoothing de oitava + outliers
# ---------------------------------------------------------------------------

def test_octave_error_corrected():
    notes = [_bn(0, 1, 60), _bn(1, 2, 62), _bn(2, 3, 88), _bn(3, 4, 65)]
    out, st = smooth_melody(notes, "natural")
    assert st["octave_corrections"] == 1
    assert out[2]["pitch"] == 64  # E6 -> E4 (oitava próxima)
    assert [n["pitch"] for n in out] == [60, 62, 64, 65]


def test_legit_progression_untouched():
    assert smooth_melody([_bn(0, 1, 60), _bn(1, 2, 72), _bn(2, 3, 84)],
                         "natural")[1]["octave_corrections"] == 0
    assert smooth_melody([_bn(0, 1, 60), _bn(1, 2, 62), _bn(2, 3, 64)],
                         "natural")[1]["octave_corrections"] == 0
    assert smooth_melody([_bn(0, 1, 60)], "natural")[1]["octave_corrections"] == 0


def test_outlier_removal_conservative():
    loud_high = [_bn(0, 1, 60), _bn(1, 3, 91, strength=0.9), _bn(3, 4, 62)]
    out, st = remove_outliers(loud_high, "natural")
    assert len(out) == 3 and st["outliers_removed"] == 0  # longa+forte fica
    weak = [_bn(0, 1, 60), _bn(1, 1.1, 91, strength=0.1), _bn(1.1, 2, 62)]
    out2, st2 = remove_outliers(weak, "natural")
    assert len(out2) == 2 and st2["outliers_removed"] == 1


# ---------------------------------------------------------------------------
# Monofonia com vencedor
# ---------------------------------------------------------------------------

def test_mono_winner_by_duration_strength():
    prev = _bn(0.0, 1.5, 60, strength=0.9)   # longa e forte
    cur = _bn(1.0, 1.3, 62, strength=0.2)    # curta e fraca sobreposta
    out, st = resolve_mono_natural([prev, cur], 0.25)
    assert st["overlaps_resolved"] == 1
    assert out[0]["end"] == pytest.approx(1.5) or out[0]["pitch"] == 60
    assert all(n["end"] > n["start"] for n in out)


def test_mono_simultaneous_stronger_wins():
    a = _bn(0.0, 1.0, 60, strength=0.2)
    b = _bn(0.0, 1.0, 67, strength=0.8)
    out, st = resolve_mono_natural([a, b], 0.25)
    assert [n["pitch"] for n in out] == [67]
    assert st["notes_dropped"] >= 1


# ---------------------------------------------------------------------------
# Acordes densos
# ---------------------------------------------------------------------------

def test_dense_chord_capped_natural():
    pitches = [36, 48, 60, 72, 84, 52, 64, 55, 67, 71]  # 10 notas
    kept, st = simplify_chord(pitches, "natural")
    assert len(kept) <= 4
    assert st["chords_simplified"] == 1
    assert min(pitches) in kept and max(pitches) in kept  # baixo + topo
    kept_d, _ = simplify_chord(pitches, "detailed")
    assert len(kept_d) == len(set(pitches))  # detailed preserva


def test_octave_stack_simplified():
    kept, st = simplify_chord([48, 60, 72, 64, 67], "natural")
    assert len(kept) <= 4
    assert 48 in kept  # baixo preservado
    assert st["duplicate_octaves_removed"] >= 1


# ---------------------------------------------------------------------------
# Harmonia estável / redução / histerese / blocos / frases / respiro
# ---------------------------------------------------------------------------

def test_window_harmony_ignores_fleeting():
    items = [{"start": 0.0, "end": 4.0, "pitches": [60, 64, 67], "velocity": 80},
             {"start": 1.0, "end": 1.1, "pitches": [61], "velocity": 40}]
    assert get_window_harmony(items, 1.0) == [0, 4, 7]
    assert get_window_harmony(items, 5.0) == []


def test_reduce_and_sustain():
    line = [{"start": 0.0, "end": 0.25, "pitch": 57, "velocity": 80},
            {"start": 0.25, "end": 0.5, "pitch": 60, "velocity": 80},
            {"start": 0.5, "end": 1.5, "pitch": 60, "velocity": 80}]
    out, st = reduce_harmony_rhythm(line)
    assert all(float(n["end"]) - float(n["start"]) >= 0.5 - 1e-9 for n in out)
    assert st["rhythmic_simplifications"] >= 1
    out2, m = sustain_merge([{"start": 0.0, "end": 1.0, "pitch": 55, "velocity": 80},
                             {"start": 1.0, "end": 2.0, "pitch": 55, "velocity": 80}])
    assert len(out2) == 1 and m == 1


def test_block_rhythm_shared():
    a = [{"start": 0.0, "end": 0.3, "pitch": 60, "velocity": 80},
         {"start": 0.6, "end": 1.0, "pitch": 62, "velocity": 80}]
    b = [{"start": 0.1, "end": 0.9, "pitch": 55, "velocity": 80}]
    blocked, _ = block_rhythm({"a": a, "b": b})
    for inst in ("a", "b"):
        for n in blocked[inst]:
            # Todo limite rítmico cai na grade compartilhada de 1/8.
            assert abs(n["start"] * 2 - round(n["start"] * 2)) < 1e-9
            assert abs(n["end"] * 2 - round(n["end"] * 2)) < 1e-9
            assert n["end"] - n["start"] >= 0.5 - 1e-9
    span_a = (blocked["a"][0]["start"], blocked["a"][-1]["end"])
    span_b = (blocked["b"][0]["start"], blocked["b"][-1]["end"])
    assert span_a == span_b == pytest.approx((0.0, 1.0))


def test_phrases_and_breathing_priority():
    line = [{"start": 0.0, "end": 1.0, "pitch": 60, "velocity": 80},
            {"start": 2.0, "end": 3.0, "pitch": 62, "velocity": 80}]
    assert phrase_boundaries(line) == [0, 1]
    # Frase longa com pausa existente: corta antes da pausa, não no meio.
    long_line = [{"start": float(i), "end": float(i + 2), "pitch": 60 + (i % 3),
                  "velocity": 80} for i in range(0, 18, 2)]
    long_line.insert(4, {"start": 8.0, "end": 8.6, "pitch": 64, "velocity": 80})
    out, st = apply_breathing_v2(long_line)
    assert st["breath_adjustments"] >= 1
    # Respiro encostou numa pausa pré-existente (>=0.5 de gap após corte).
    gaps = [float(nxt["start"]) - float(cur["end"])
            for cur, nxt in zip(out, out[1:])]
    assert max(gaps) >= 0.5


def test_clutter_score_orders():
    clean = [{"start": float(i), "end": float(i + 1), "pitch": 60} for i in range(8)]
    noisy = [{"start": i * 0.2, "end": i * 0.2 + 0.1, "pitch": 60 + (30 if i == 5 else i % 3)}
             for i in range(20)]
    c1 = clutter_score(clean, 8.0)["score"]
    c2 = clutter_score(noisy, 4.0)["score"]
    assert c2 > c1
    assert clutter_score(clean, 8.0)["score"] == c1  # determinístico


def test_line_stats_counts():
    line = [{"start": 0.0, "end": 1.0, "pitch": 60, "velocity": 80},
            {"start": 1.0, "end": 2.0, "pitch": 72, "velocity": 80},
            {"start": 2.0, "end": 3.0, "pitch": 72, "velocity": 80}]
    st = line_stats(line)
    assert st == {"notes": 3, "large_leaps": 1, "very_large_leaps": 0,
                  "average_interval": 6.0, "harmonic_changes": 1}


# ---------------------------------------------------------------------------
# Workers: natural x detailed (fixture ruidosa, cache por perfil)
# ---------------------------------------------------------------------------

def _noisy_trans():
    vocals = [_ev(0.0, 0.5, 60), _ev(0.53, 0.58, 60, amp=0.6),
              _ev(0.58, 1.0, 60), _ev(1.0, 1.04, 90, amp=0.1),
              _ev(1.04, 1.5, 62), _ev(1.5, 2.0, 76 + 12),
              _ev(2.0, 2.5, 64), _ev(2.5, 3.0, 65)]
    bass = [_ev(0.0, 1.0, 40), _ev(1.0, 2.0, 43), _ev(2.0, 3.0, 45)]
    other = [_ev(0.0, 0.5, 60), _ev(0.0, 0.5, 60), _ev(0.0, 0.5, 64),
             _ev(0.0, 0.5, 67), _ev(0.0, 0.5, 72), _ev(0.0, 0.5, 48),
             _ev(1.0, 2.0, 65), _ev(2.0, 2.2, 67, amp=0.7)]
    return {"vocals": vocals, "bass": bass, "other": other}


_CACHE_W: dict = {}


def _run_score_worker(trans_map: dict, profile: str) -> dict:
    if profile in _CACHE_W:
        return _CACHE_W[profile]
    if not _notation_ok():
        pytest.skip("music21/.venv-notation indisponível")
    tmp = Path(tempfile.mkdtemp(prefix="cleanup_"))
    td = tmp / "trans"
    td.mkdir(parents=True, exist_ok=True)
    for stem in ("vocals", "bass", "other"):
        events = trans_map.get(stem, [])
        with open(td / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump({"file_id": "cl", "stem": stem, "notes_count": len(events),
                       "events": events}, f)
    out_xml = tmp / "score.musicxml"
    out_model = tmp / "score.json"
    py = get_notation_python()
    cmd = [str(Path(py).resolve()), str(NOTATION_WORKER.resolve()),
           "--file-id", "cl", "--transcriptions-dir", str(td.resolve()),
           "--output-musicxml", str(out_xml.resolve()),
           "--output-model", str(out_model.resolve()),
           "--tempo", "120", "--time-signature", "4/4", "--quantization", "1/16",
           "--key-mode", "none", "--beat-offset", "0.0",
           "--cleanup-profile", profile]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(BASE_DIR))
    assert r.returncode == 0, f"score worker rc={r.returncode} stderr={r.stderr[-800:]}"
    model = json.loads(out_model.read_text(encoding="utf-8"))
    xml_text = out_xml.read_text(encoding="utf-8")
    ET.fromstring(xml_text)
    res = {"model": model, "xml_text": xml_text}
    _CACHE_W[profile] = res
    return res


def test_natural_le_keeps_melody_loses_noise():
    nat = _run_score_worker(_noisy_trans(), "natural")
    det = _run_score_worker(_noisy_trans(), "detailed")
    assert nat["model"]["cleanup_profile"] == "natural"
    assert det["model"]["cleanup_profile"] == "detailed"
    # Melodia preservada (C D E F + oitava corrigida), ruído removido.
    v_nat = nat["model"]["parts"]["vocals"]
    v_det = det["model"]["parts"]["vocals"]
    assert v_nat["reparsed_notes"] <= v_det["reparsed_notes"]
    assert v_nat["reparsed_notes"] >= 4
    assert v_nat["octave_corrections"] >= 1
    assert v_nat["short_notes_merged"] + v_nat["short_notes_removed"] >= 1
    # Acorde denso 6 notas -> teto natural.
    o_nat = nat["model"]["parts"]["other"]
    assert o_nat["chords_simplified"] >= 1
    for key in ("raw_notes", "cleaned_notes", "short_notes_merged", "short_notes_removed",
                "octave_corrections", "outliers_removed", "chords_simplified",
                "duplicate_octaves_removed", "rhythmic_simplifications", "sustains_merged"):
        assert key in v_nat, key


def test_voice_leading_natural_not_worse():
    from backend.arrangement.arranger import arrange
    from backend.arrangement.instrument_definitions import get_instrument
    melody = [{"start": 0.0, "end": 0.5, "pitch": 72, "velocity": 80},
              {"start": 0.5, "end": 1.0, "pitch": 74, "velocity": 80},
              {"start": 1.0, "end": 1.5, "pitch": 76, "velocity": 80},
              {"start": 1.5, "end": 2.0, "pitch": 77, "velocity": 80}]
    other = [{"start": 0.0, "end": 2.0, "pitches": [48, 55, 60, 64], "velocity": 80}]
    defs = [get_instrument("trumpet"), get_instrument("tenor_sax")]
    _, rep_d = arrange(melody, other, defs, mode="automatic", simplify=False, profile="detailed")
    _, rep_n = arrange(melody, other, defs, mode="automatic", simplify=False, profile="natural")
    sd = rep_d["stats"]["tenor_sax"]
    sn = rep_n["stats"]["tenor_sax"]
    assert sn["role"] == sd["role"] == "harmony"
    assert sn["average_interval_after"] <= sd["average_interval_after"] + 0.5
    assert sn["large_leaps_after"] <= sd["large_leaps_after"]


# ---------------------------------------------------------------------------
# Parte 25 (7.2): dez testes do natural mais musical
# ---------------------------------------------------------------------------

def test72_dense_cluster_capped():
    from backend.musical.cleanup import reduce_chord_voicing
    pitches = [36, 48, 50, 52, 55, 60, 61, 64, 67]
    kept, st = reduce_chord_voicing(pitches, "natural")
    assert len(kept) <= 4
    assert 36 in kept and max(pitches) in kept
    assert st["chords_simplified"] == 1
    kept_d, _ = reduce_chord_voicing(pitches, "detailed")
    assert len(kept_d) == len(set(pitches))


def test72_harmony_ignores_passing_blip():
    from backend.arrangement.arranger import build_harmony_line
    from backend.arrangement.instrument_definitions import get_instrument
    melody = [{"start": 0.0, "end": 1.0, "pitch": 72, "velocity": 80},
              {"start": 1.0, "end": 1.2, "pitch": 74, "velocity": 80},
              {"start": 1.2, "end": 2.0, "pitch": 72, "velocity": 80}]
    other = [{"start": 0.0, "end": 2.0, "pitches": [48, 55, 60], "velocity": 80},
             {"start": 1.0, "end": 1.2, "pitches": [49], "velocity": 30}]
    line, _ = build_harmony_line(melody, other, get_instrument("alto_sax"),
                                 profile="natural")
    pcs = {n["pitch"] % 12 for n in line}
    assert pcs <= {0, 3, 7}  # C# passageiro (pc 1) não vira a harmonia


def test72_alto_sustains_fast_melody():
    from backend.arrangement.arranger import arrange
    from backend.arrangement.instrument_definitions import get_instrument
    melody = [{"start": i * 0.25, "end": i * 0.25 + 0.25, "pitch": 72 + (i % 4),
               "velocity": 80} for i in range(8)]  # 8 semicolcheias
    other = [{"start": 0.0, "end": 2.0, "pitches": [48, 55, 60], "velocity": 80}]
    _, rep = arrange(melody, other, [get_instrument("alto_sax"),
                                     get_instrument("trumpet")],
                     mode="automatic", simplify=False, profile="natural")
    alto = rep["stats"]["alto_sax"]
    assert alto["final_notes"] <= 4  # sustenta em vez de trocar 8 vezes
    assert alto["harmonic_changes_after"] <= 3


def test72_trombone_long_notes():
    from backend.arrangement.arranger import arrange
    from backend.arrangement.instrument_definitions import get_instrument
    melody = [{"start": i * 0.25, "end": i * 0.25 + 0.25, "pitch": 60 + (i % 4),
               "velocity": 80} for i in range(8)]
    other = [{"start": 0.0, "end": 2.0, "pitches": [36, 43, 48], "velocity": 80}]
    _, rep = arrange(melody, other, [get_instrument("trombone"),
                                     get_instrument("trumpet")],
                     mode="automatic", simplify=False, profile="natural")
    line_notes = rep["stats"]["trombone"]["final_notes"]
    assert line_notes <= 4
    assert rep["stats"]["trombone"]["harmonic_changes_after"] <= 3


def test72_same_harmony_sustains_gaps():
    from backend.musical.cleanup import merge_adjacent_harmony_notes
    line = [{"start": 0.0, "end": 1.0, "pitch": 55, "velocity": 80},
            {"start": 1.1, "end": 2.0, "pitch": 55, "velocity": 80},
            {"start": 2.5, "end": 3.0, "pitch": 55, "velocity": 80}]
    out, m = merge_adjacent_harmony_notes(line)
    assert len(out) == 2 and m == 1  # gaps .1 funde; .5 não
    assert out[0]["end"] == pytest.approx(2.0)


def test72_large_leap_prefers_step():
    from backend.arrangement.arranger import select_harmony_note_v2
    from backend.arrangement.instrument_definitions import get_instrument
    ten = get_instrument("tenor_sax")
    # De C4, candidatos C5 (salto 12) e C3... com C5 e G3 ativos: prefere perto.
    got, _ = select_harmony_note_v2(72, [0, 7], 48, ten, melody_dur=1.0)
    assert got is not None and abs(got - 48) <= 7


def test72_crossing_fixed():
    from backend.arrangement.arranger import fix_crossings
    from backend.arrangement.instrument_definitions import get_instrument
    lines = {"alto_sax": [{"start": 0.0, "end": 1.0, "pitch": 55, "velocity": 80}],
             "trombone": [{"start": 0.0, "end": 1.0, "pitch": 60, "velocity": 80}]}
    roles = [("alto_sax", "harmony"), ("trombone", "low")]
    defs = {"alto_sax": get_instrument("alto_sax"), "trombone": get_instrument("trombone")}
    fixed_lines, fixed, remaining = fix_crossings(lines, roles, defs)
    assert fixed >= 1 and remaining == 0
    assert fixed_lines["trombone"][0]["pitch"] == 48  # oitava abaixo, no range


def test72_phrase_end_stable():
    from backend.arrangement.arranger import build_harmony_line
    from backend.arrangement.instrument_definitions import get_instrument
    melody = [{"start": 0.0, "end": 1.0, "pitch": 72, "velocity": 80},
              {"start": 1.0, "end": 2.0, "pitch": 74, "velocity": 80}]
    other = [{"start": 0.0, "end": 2.0, "pitches": [48, 55, 60], "velocity": 80}]
    line, _ = build_harmony_line(melody, other, get_instrument("trombone"),
                                 low_role=True, profile="natural")
    assert line and line[-1]["pitch"] % 12 in (0, 3, 7)  # root/3rd/5th, não passagem


def test72_register_center_no_flip():
    from backend.arrangement.arranger import _fit_line
    from backend.arrangement.instrument_definitions import get_instrument
    out, _ = _fit_line([{"start": 0.0, "end": 1.0, "pitch": 64, "velocity": 80},
                        {"start": 1.0, "end": 2.0, "pitch": 40, "velocity": 80}],
                       get_instrument("trombone"))
    assert [n["pitch"] for n in out] == [52, 40]  # empate justeen: centro decide
    med = sorted(n["pitch"] for n in out)[1]
    assert 43 <= med <= 60


def test72_naturalness_orders():
    from backend.musical.cleanup import naturalness_score
    smooth = {"t": [{"start": 0.0, "end": 2.0, "pitch": 60, "velocity": 80},
                    {"start": 2.0, "end": 4.0, "pitch": 60, "velocity": 80}]}
    choppy = {"t": [{"start": i * 0.25, "end": i * 0.25 + 0.2, "pitch": 60 + (i * 7) % 20,
                     "velocity": 80} for i in range(16)]}
    assert naturalness_score(smooth, 4.0)["score"] > naturalness_score(choppy, 4.0)["score"]


def _oit(start, end, pitches, velocity=80):
    return {"start": start, "end": end, "pitches": list(pitches),
            "velocity": velocity, "kind": "chord" if len(pitches) > 1 else "note"}


# ---------------------------------------------------------------------------
# Etapa 7.2.1: acompanhamento limpo de Other (12 testes)
# ---------------------------------------------------------------------------

def test721_trio_cap_max3():
    # A) C2 C3 C4 E4 G4 -> máximo 3, com fundamental/terça/quinta úteis.
    write, st = refine_other_accompaniment(
        [_oit(0.0, 1.0, [36, 48, 60, 64, 67])], "natural")
    assert len(write[0]["pitches"]) <= 3
    assert {p % 12 for p in write[0]["pitches"]} <= {0, 4, 7}
    assert 48 in write[0]["pitches"]  # C3: oitava útil, não os extremos
    assert st["final_polyphony_max"] <= 3


def test721_hold_sustain_repeated():
    # B) C-E-G repetido 4x -> um sustain.
    items = [_oit(float(i), float(i + 1), [60, 64, 67]) for i in range(4)]
    write, st = refine_other_accompaniment(items, "natural")
    assert len(write) == 1
    assert write[0]["end"] == pytest.approx(4.0)
    assert st["harmony_rearticulations_removed"] >= 3


def test721_cluster_weak_removed():
    # C) C C# D G com C#/D fracos e não persistentes -> estrutura limpa.
    items = [_oit(0.0, 1.0, [60], velocity=80),
             _oit(0.0, 2.0, [60, 61, 62, 67], velocity=80),
             _oit(2.0, 3.0, [60, 67], velocity=80)]
    write, st = refine_other_accompaniment(items, "natural")
    mid = [it for it in write if abs(it["start"] - 0.0) < 1e-9][0]
    assert 61 not in mid["pitches"] and 62 not in mid["pitches"]
    assert 60 in mid["pitches"] and 67 in mid["pitches"]
    assert st["cluster_events_removed"] >= 1


def test721_voice_leading_return_home():
    # D) C-E-G -> B-D-G -> C-E-G sem absurdo e voltando ao voicing.
    items = [_oit(0.0, 1.0, [60, 64, 67]),
             _oit(1.0, 2.0, [59, 62, 67]),
             _oit(2.0, 3.0, [60, 64, 67])]
    write, st = refine_other_accompaniment(items, "natural")
    assert write[0]["pitches"] == write[-1]["pitches"]
    assert st["average_voice_movement"] <= 4.0


def test721_root_stable_with_bass():
    # E) Baixo C sustentado + micro C#/D -> raiz C permanece.
    bass = [{"start": 0.0, "end": 4.0, "pitch": 36, "velocity": 90}]
    items = [_oit(0.0, 1.0, [48, 52, 55]),
             _oit(1.0, 1.2, [49, 52], velocity=40),
             _oit(1.2, 2.0, [48, 52, 55])]
    write, _ = refine_other_accompaniment(items, "natural", bass_notes=bass)
    assert all(0 in {p % 12 for p in it["pitches"]} for it in write)
    assert len(write) == 1  # micro-evento absorvido, sem rearticular


def test721_compact_span():
    write, _ = refine_other_accompaniment([_oit(0.0, 1.0, [36, 64, 79])], "natural")
    pl = write[0]["pitches"]
    assert max(pl) - min(pl) <= 16
    assert {p % 12 for p in pl} == {0, 4, 7}


def test721_min_duration_absorb():
    items = [_oit(0.0, 1.0, [60, 64, 67]),
             _oit(1.0, 1.2, [60, 64, 67], velocity=80),
             _oit(1.2, 3.0, [60, 64, 67])]
    write, st = refine_other_accompaniment(items, "natural")
    assert all(float(it["end"]) - float(it["start"]) >= 0.5 - 1e-9 for it in write)
    assert st["harmony_rearticulations_removed"] >= 1


def test721_metrics_keys():
    _, st = refine_other_accompaniment([_oit(0.0, 1.0, [60, 64, 67])], "natural")
    for key in ("raw_polyphony_max", "final_polyphony_max", "raw_chord_count",
                "final_chord_count", "cluster_events_removed",
                "duplicate_pitch_classes_removed", "harmony_rearticulations_removed",
                "average_chord_duration", "average_voice_movement"):
        assert key in st, key


def test721_detailed_legacy_intact():
    items = [_oit(0.0, 1.0, [36, 48, 60, 61, 64, 67])]
    write, st = refine_other_accompaniment(items, "detailed")
    assert write[0]["pitches"] == [36, 48, 60, 61, 64, 67]
    assert all(v == 0 for k, v in st.items() if isinstance(v, int))


def test721_nonempty_guard():
    write, _ = refine_other_accompaniment([_oit(0.0, 0.1, [90], velocity=30)], "natural")
    assert len(write) >= 1  # nunca esvazia parte com conteúdo real


def test721_harmonic_source_complete():
    from backend.workers.notation_worker import process_other
    events = [{"start": 0.0, "end": 0.5, "pitch": 60, "note": "X", "velocity": 80,
               "amplitude": 0.8, "confidence": 0.8, "strength": 0.8},
              {"start": 0.0, "end": 0.5, "pitch": 64, "note": "X", "velocity": 80,
               "amplitude": 0.8, "confidence": 0.8, "strength": 0.8},
              {"start": 0.5, "end": 1.0, "pitch": 67, "note": "X", "velocity": 80,
               "amplitude": 0.8, "confidence": 0.8, "strength": 0.8}]
    items, _, _ = process_other(events, 120.0, 0.0, 0.25, profile="natural")
    raw_pcs = {60 % 12, 64 % 12, 67 % 12}
    src_pcs = {p % 12 for it in items for p in it["pitches"]}
    assert raw_pcs <= src_pcs  # sopros enxergam tudo (fonte intacta)


def test721_register_no_simultaneous_extremes():
    write, _ = refine_other_accompaniment([_oit(0.0, 2.0, [24, 60, 64, 100])], "natural")
    pl = write[0]["pitches"]
    assert max(pl) - min(pl) <= 16
    assert {p % 12 for p in pl} == {0, 4}
