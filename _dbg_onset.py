import sys
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "tests")
from test_drums import _kick, _snare, SR
from backend.drums import drum_utils as DU

SRV = SR


def kick_short(dur=0.15):
    t = np.arange(int(dur * SRV)) / SRV
    f = 55 + 40 * np.exp(-t * 30)
    ph = 2 * np.pi * np.cumsum(f) / SRV
    return 0.9 * np.sin(ph) * np.exp(-t * 14)


y = np.zeros(int(2.6 * SRV))
for b in [0.5, 1.0, 1.5, 2.0]:
    k = kick_short()
    i0 = int(b * SRV)
    y[i0:i0 + len(k)] += k
t, s = DU.detect_onsets(y, SRV)
print("short-kick onsets:", [(round(x, 3), round(z, 3)) for x, z in zip(t, s)])

y2 = np.zeros(int(1.5 * SRV))
k = kick_short()
y2[0:len(k)] += k
g = kick_short() * 0.25
off = int(0.75 * SRV)
y2[off:off + len(g)] += g
t2, s2 = DU.detect_onsets(y2, SRV)
print("ghost onsets:", [(round(x, 3), round(z, 3)) for x, z in zip(t2, s2)])
ev, st = DU.transcribe_drums(y2, SRV, tempo=120.0, beat_offset=0.0, profile="natural")
print("ghost natural:", [(e["instrument"], e["beat"], e["strength"], e["confidence"]) for e in ev], st["ghost_notes_removed"])
evd, std = DU.transcribe_drums(y2, SRV, tempo=120.0, beat_offset=0.0, profile="detailed")
print("ghost detailed:", [(e["instrument"], e["beat"]) for e in evd])
