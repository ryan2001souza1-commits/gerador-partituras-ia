"""
Testes automatizados Etapa 3 — BPM, tonalidade, confiança, limpeza e API.
Executar: .\.venv\Scripts\python.exe -m pytest tests -v
Ou: .\.venv\Scripts\python.exe tests/test_music_analysis.py
"""
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

SR = 22050

NOTE_FREQS = {
    "C": 261.63, "C#": 277.18, "D": 293.66, "D#": 311.13, "E": 329.63,
    "F": 349.23, "F#": 369.99, "G": 391.995, "G#": 415.30, "A": 440.00,
    "A#": 466.16, "B": 493.88, "C5": 523.25, "D5": 587.33, "E5": 659.25,
    "F5": 698.46, "G5": 783.99, "A5": 880.00,
}

def generate_click_track(bpm, duration_sec=10, sr=SR):
    total_samples = int(duration_sec * sr)
    y = np.zeros(total_samples, dtype=np.float32)
    interval = 60.0 / bpm
    click_dur = 0.02
    click_samples = int(click_dur * sr)
    click_freq = 1000
    t_click = np.arange(click_samples) / sr
    click_wave = 0.8 * np.sin(2 * np.pi * click_freq * t_click) * np.exp(-t_click*80)
    num_clicks = int(duration_sec / interval)
    for n in range(num_clicks):
        pos = int(n * interval * sr)
        end = pos + click_samples
        if end < total_samples:
            y[pos:end] += click_wave
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.6
    return y

def generate_chord(freqs, duration_sec, sr=SR):
    t = np.arange(int(duration_sec * sr)) / sr
    y = np.zeros_like(t)
    for f in freqs:
        y += 0.4 * np.sin(2*np.pi*f*t) + 0.2 * np.sin(2*np.pi*2*f*t) * np.exp(-t*0.5) + 0.1*np.sin(2*np.pi*3*f*t)
    y = y / (len(freqs)) * 0.5
    fade = int(0.01 * sr)
    y[:fade] *= np.linspace(0,1,fade)
    y[-fade:] *= np.linspace(1,0,fade)
    return y

def generate_progression(progression, chord_dur=2.0, sr=SR):
    ys = [generate_chord(freqs, chord_dur, sr) for freqs in progression]
    y = np.concatenate(ys)
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.6
    return y

def save_wav(y, sr, path):
    sf.write(str(path), y, sr)

# ----------------- TESTS -----------------

def test_import_librosa():
    import librosa
    assert librosa.__version__ == "0.11.0"

def test_py_compile():
    import py_compile
    py_compile.compile("app.py", doraise=True)
    py_compile.compile("backend/audio/probe.py", doraise=True)
    py_compile.compile("backend/audio/music_analysis.py", doraise=True)

def test_health():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def _bpm_check(expected, tol=6):
    y = generate_click_track(expected, duration_sec=12)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        p = Path(tmp.name)
    try:
        save_wav(y, SR, p)
        from backend.audio.music_analysis import analyze_music
        res = analyze_music(p)
        assert res.bpm is not None, f"BPM None for {expected}"
        diff = abs(res.bpm - expected)
        allowed = max(tol, expected*0.08)
        assert diff <= allowed, f"BPM esperado {expected} detectado {res.bpm} diff {diff} > {allowed}"
        assert res.bpm_confidence is not None and 0 <= res.bpm_confidence <= 1
    finally:
        p.unlink(missing_ok=True)

def test_bpm_60():
    _bpm_check(60)

def test_bpm_90():
    _bpm_check(90)

def test_bpm_120():
    _bpm_check(120)

def test_bpm_140():
    _bpm_check(140)

def test_key_c_major():
    prog = [
        [261.63, 329.63, 391.995],
        [349.23, 440.00, 523.25],
        [391.995, 493.88, 587.33],
        [261.63, 329.63, 391.995],
    ]
    y = generate_progression(prog)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        p = Path(tmp.name)
    try:
        save_wav(y, SR, p)
        from backend.audio.music_analysis import analyze_music
        res = analyze_music(p)
        assert res.key == "C" and res.mode == "major", f"got {res.key} {res.mode}"
        assert res.key_confidence is not None and 0 <= res.key_confidence <= 1
    finally:
        p.unlink(missing_ok=True)

def test_key_a_minor():
    prog = [
        [220.00, 261.63, 329.63],
        [293.66, 349.23, 440.00],
        [329.63, 415.30, 493.88],
        [220.00, 261.63, 329.63],
    ]
    y = generate_progression(prog)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        p = Path(tmp.name)
    try:
        save_wav(y, SR, p)
        from backend.audio.music_analysis import analyze_music
        res = analyze_music(p)
        assert res.key == "A" and res.mode == "minor", f"got {res.key} {res.mode}"
    finally:
        p.unlink(missing_ok=True)

def test_key_e_major():
    prog = [
        [329.63, 415.30, 493.88],
        [440.00, 554.37, 659.25],
        [493.88, 622.25, 739.99],
        [329.63, 415.30, 493.88],
    ]
    y = generate_progression(prog)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        p = Path(tmp.name)
    try:
        save_wav(y, SR, p)
        from backend.audio.music_analysis import analyze_music
        res = analyze_music(p)
        assert res.key == "E" and res.mode == "major", f"got {res.key} {res.mode}"
    finally:
        p.unlink(missing_ok=True)

def test_short_file():
    y = generate_click_track(120, duration_sec=1.5)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        p = Path(tmp.name)
    try:
        save_wav(y, SR, p)
        from backend.audio.music_analysis import analyze_music
        res = analyze_music(p, duration_probe=1.5)
        assert res.warning and "curto" in res.warning
    finally:
        p.unlink(missing_ok=True)

def test_silence():
    y = np.zeros(int(5*SR), dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        p = Path(tmp.name)
    try:
        save_wav(y, SR, p)
        from backend.audio.music_analysis import analyze_music
        res = analyze_music(p)
        assert res.bpm is None or (res.bpm_confidence is not None and res.bpm_confidence < 0.4)
        assert res.key is None or (res.key_confidence is not None and res.key_confidence < 0.4)
    finally:
        p.unlink(missing_ok=True)

def test_invalid_audio():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, mode='w') as tmp:
        tmp.write("não é áudio")
        p = Path(tmp.name)
    try:
        from fastapi.testclient import TestClient
        from app import app
        client = TestClient(app)
        with open(p, "rb") as f:
            r = client.post("/api/upload", files={"file": ("fake.wav", f, "audio/wav")})
        assert r.status_code == 400
    finally:
        p.unlink(missing_ok=True)

def test_ffmpeg_error():
    from backend.audio.music_analysis import analyze_music
    p = Path("nao_existe_12345.wav")
    res = analyze_music(p)
    assert res.error is not None

def test_temp_cleanup():
    y = generate_click_track(120, duration_sec=5)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        p = Path(tmp.name)
    try:
        save_wav(y, SR, p)
        from backend.audio.music_analysis import analyze_music
        import tempfile as tf
        before = set(Path(tf.gettempdir()).glob("*.wav"))
        res = analyze_music(p)
        after = set(Path(tf.gettempdir()).glob("*.wav"))
        leaked = [x for x in after if x not in before and "tmp" in x.name]
        assert not leaked
        assert p.exists()
    finally:
        p.unlink(missing_ok=True)

def test_api_fields():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    y = generate_click_track(120, duration_sec=6)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        p = Path(tmp.name)
    try:
        save_wav(y, SR, p)
        with open(p, "rb") as f:
            r = client.post("/api/upload", files={"file": ("test120.wav", f, "audio/wav")})
        assert r.status_code == 200
        file_id = r.json()["file_id"]
        r2 = client.get(f"/api/analyze/{file_id}")
        assert r2.status_code == 200
        data = r2.json()
        for k in ["duration","format","codec","sample_rate","channels","bitrate","size_bytes","music"]:
            assert k in data, f"missing {k}"
        music = data["music"]
        for k in ["bpm","bpm_rounded","bpm_confidence","key","mode","key_confidence"]:
            assert k in music, f"missing music {k}"
    finally:
        p.unlink(missing_ok=True)

def test_frontend_null_handling():
    js = Path("frontend/app.js").read_text(encoding="utf-8")
    assert "showMusicInfo" in js
    assert "formatKeyPT" in js
    assert "Não foi possível determinar" in js

if __name__ == "__main__":
    test_import_librosa(); print("pass import")
    test_py_compile(); print("pass py_compile")
    test_health(); print("pass health")
    test_bpm_60(); print("pass bpm60")
    test_bpm_90(); print("pass bpm90")
    test_bpm_120(); print("pass bpm120")
    test_bpm_140(); print("pass bpm140")
    test_key_c_major(); print("pass c")
    test_key_a_minor(); print("pass a")
    test_key_e_major(); print("pass e")
    test_short_file(); print("pass short")
    test_silence(); print("pass silence")
    test_invalid_audio(); print("pass invalid")
    test_ffmpeg_error(); print("pass ffmpeg")
    test_temp_cleanup(); print("pass temp")
    test_api_fields(); print("pass api")
    test_frontend_null_handling(); print("pass frontend")
    print("ALL PASS")
