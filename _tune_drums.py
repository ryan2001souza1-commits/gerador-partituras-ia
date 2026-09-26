import sys
import numpy as np

sys.path.insert(0, ".")
from backend.drums import drum_utils as D

SR = 22050


def kick(dur=0.4):
    t = np.arange(int(dur * SR)) / SR
    f = 55 + 40 * np.exp(-t * 30)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return 0.9 * np.sin(ph) * np.exp(-t * 9)


def snare(dur=0.3):
    t = np.arange(int(dur * SR)) / SR
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(len(t))
    try:
        from scipy import signal
        sos = signal.butter(4, 6000, btype="lowpass", fs=SR, output="sos")
        noise = signal.sosfilt(sos, noise)
    except Exception:
        pass
    body = (np.sin(2 * np.pi * 190 * t) + 0.5 * np.sin(2 * np.pi * 250 * t)) * np.exp(-t * 25)
    return (0.55 * noise * np.exp(-t * 28) + 0.45 * body)


def hat(dur=0.06, seed=3, decay=90.0):
    t = np.arange(int(dur * SR)) / SR
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(t))
    try:
        from scipy import signal
        sos = signal.butter(4, 7000, btype="highpass", fs=SR, output="sos")
        noise = signal.sosfilt(sos, noise)
    except Exception:
        pass
    return 0.6 * noise * np.exp(-t * decay)


def openhat():
    return hat(dur=0.5, seed=5, decay=4.0)


def crash(dur=1.0):
    t = np.arange(int(dur * SR)) / SR
    rng = np.random.default_rng(11)
    noise = rng.standard_normal(len(t))
    return 0.7 * noise * np.exp(-t * 4)


def tom(freq=150, dur=0.4):
    t = np.arange(int(dur * SR)) / SR
    rng = np.random.default_rng(21)
    click = rng.standard_normal(len(t)) * np.exp(-t * 300) * 0.15
    return (0.7 * np.sin(2 * np.pi * freq * t) * np.exp(-t * 10)
            + 0.25 * np.sin(2 * np.pi * freq * 2.02 * t) * np.exp(-t * 14)
            + click)


cases = [("kick", kick()), ("snare", snare()), ("closed", hat()),
         ("open", openhat()), ("crash", crash()), ("tom_low", tom(110)),
         ("tom_mid", tom(180)), ("tom_high", tom(260))]
for name, y in cases:
    S, freqs, hop = D._stft_once(y, SR)
    feats = D.extract_features(y, SR, 0.01, S=S, freqs=freqs, hop=hop)
    cls, conf, scores = D.classify_hit(feats, 0.9, y=y, sr=SR, onset_time=0.01)
    top = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
    print(f"{name}: kick={scores.get('kick', 0):.3f} hat={scores.get('closed_hihat', 0):.3f} "
          f"snr={scores.get('snare', 0):.3f} -> {cls} {conf:.2f}")
