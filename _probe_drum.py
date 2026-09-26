from music21 import stream, note, percussion, instrument, clef, meter

p = stream.Part()
p.partName = "Bateria"
p.insert(0, instrument.UnpitchedPercussion())
p.insert(0, clef.PercussionClef())
p.insert(0, meter.TimeSignature("4/4"))

kick = note.Unpitched()
kick.displayStep = "F"
kick.displayOctave = 4
kick.quarterLength = 0.5
kick.storedInstrument = instrument.BassDrum()

snare = note.Unpitched()
snare.displayStep = "C"
snare.displayOctave = 5
snare.quarterLength = 0.5
snare.storedInstrument = instrument.SnareDrum()

hat = note.Unpitched()
hat.displayStep = "G"
hat.displayOctave = 5
hat.quarterLength = 0.5
hat.notehead = "x"

def _kick(ql=0.5):
    n = note.Unpitched()
    n.displayStep = "F"
    n.displayOctave = 4
    n.quarterLength = ql
    n.storedInstrument = instrument.BassDrum()
    return n


def _snare(ql=0.5):
    n = note.Unpitched()
    n.displayStep = "C"
    n.displayOctave = 5
    n.quarterLength = ql
    n.storedInstrument = instrument.SnareDrum()
    return n


def _hat(ql=0.5):
    n = note.Unpitched()
    n.displayStep = "G"
    n.displayOctave = 5
    n.quarterLength = ql
    n.notehead = "x"
    return n


v1 = stream.Voice()
v1.id = 1
v1.insert(0, _hat())
v1.insert(0.5, _snare())
v1.insert(1.0, _hat())
r = note.Rest()
r.quarterLength = 1.0
v1.insert(1.5, r)
v2 = stream.Voice()
v2.id = 2
v2.insert(0, _kick())
r2 = note.Rest()
r2.quarterLength = 1.5
v2.insert(0.5, r2)
p.insert(0, v1)
p.insert(0, v2)
p.makeMeasures(inPlace=True)
s = stream.Score()
s.insert(0, p)
s.write("musicxml", fp="_probe_drum.xml")
t = open("_probe_drum.xml", encoding="utf-8").read()
import re
print("unpitched tags:", t.count("<unpitched"))
print("notehead x:", t.count("x</notehead>"))
print("clef PERC:", "PERC" in t)
print("voices:", sorted(set(re.findall(r"<voice>(\d+)</voice>", t))))
print("has rest:", "<rest" in t)
# reparse
from music21 import converter
q = converter.parse("_probe_drum.xml")
parts = list(q.parts)
print("parts:", len(parts), parts[0].partName)
els = list(parts[0].flatten().notesAndRests)
print("els:", [(type(e).__name__) for e in els])
