import sys
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "tests")
from test_drums import _kick, _hat
from backend.drums import drum_utils as DU

SR = 22050

# dedupe case
y = np.zeros(int(1.0 * SR))
k1, k2 = _kick(0.3), _kick(0.3)
y[0:len(k1)] += k1
off = int(0.015 * SR)
y[off:off + len(k2)] += k2 * 0.7
t, s = DU.detect_onsets(y, SR)
print("dedupe onsets:", t, s)
ev, st = DU.transcribe_drums(y, SR, tempo=120.0, beat_offset=0.0)
print("dedupe events:", [(e["instrument"], e["time"], e["confidence"]) for e in ev], st)

# simultaneous case
y2 = np.zeros(int(1.0 * SR))
k, h = _kick(0.4), _hat()
off2 = int(0.1 * SR)
y2[off2:off2 + len(k)] += k
y2[off2:off2 + len(h)] += h
t2, s2 = DU.detect_onsets(y2, SR)
print("sim onsets:", t2, s2)
S, freqs, hop = DU._stft_once(y2, SR)
feats = DU.extract_features(y2, SR, t2[0] if t2 else 0.01, S=S, freqs=freqs, hop=hop)
print("sim feats:", {k: round(v, 3) for k, v in feats.items() if k in ("low", "lowmid", "mid", "high", "centroid", "decay_ms")})
cls, conf, scores = DU.classify_hit(feats, 0.9, y=y2, sr=SR, onset_time=t2[0] if t2 else 0.01)
print("sim class:", cls, conf, sorted(scores.items(), key=lambda kv: -kv[1])[:4])
