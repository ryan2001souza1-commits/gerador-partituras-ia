"""
Etapa 7 — definições centrais dos instrumentos de sopro (stdlib apenas).

Todo o raciocínio interno do arranjador é em CONCERT PITCH (som real).
Na exportação, written = concert + transposition_semitones e o MusicXML
recebe o elemento <transpose> via music21 (verificado por reparse).

Ranges conservadores (não extremos profissionais):
- absolute_*: limites físicos prat-aprendiz; fora daqui -> pausa + warning.
- preferred_*: tessitura confortável; o arranjador tenta manter aqui com
  octave shift ±12 quando musicalmente possível (registra range_adjustments).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class InstrumentDefinition:
    id: str                      # ex. "alto_sax" (allowlist da API)
    name: str                    # ex. "Sax Alto em Eb" (exibido + partName)
    short_name: str              # ex. "Sx. A."
    family: str                  # "Sopros / Metais"
    concert_key: str             # "Eb" | "Bb" | "C"
    transposition_semitones: int  # written = concert + T (alto +9, tenor +14, ...)
    written_low: int             # MIDI absoluto escrito (grave)
    written_high: int            # MIDI absoluto escrito (agudo)
    preferred_low: int           # MIDI confortável escrito (grave)
    preferred_high: int          # MIDI confortável escrito (agudo)
    clef: str                    # "treble" | "bass"
    music21_instrument: str      # classe em music21.instrument
    melody_capable: bool = True


INSTRUMENTS: Dict[str, InstrumentDefinition] = {
    "alto_sax": InstrumentDefinition(
        id="alto_sax", name="Sax Alto em Eb", short_name="Sx. A.",
        family="Sopros / Metais", concert_key="Eb", transposition_semitones=9,
        written_low=58, written_high=89, preferred_low=62, preferred_high=86,
        clef="treble", music21_instrument="AltoSaxophone",
    ),
    "tenor_sax": InstrumentDefinition(
        id="tenor_sax", name="Sax Tenor em Bb", short_name="Sx. T.",
        family="Sopros / Metais", concert_key="Bb", transposition_semitones=14,
        written_low=46, written_high=78, preferred_low=50, preferred_high=74,
        clef="treble", music21_instrument="TenorSaxophone",
    ),
    "trumpet": InstrumentDefinition(
        id="trumpet", name="Trompete em Bb", short_name="Tpt.",
        family="Sopros / Metais", concert_key="Bb", transposition_semitones=2,
        written_low=55, written_high=84, preferred_low=57, preferred_high=79,
        clef="treble", music21_instrument="Trumpet",
    ),
    "trombone": InstrumentDefinition(
        id="trombone", name="Trombone", short_name="Tbn.",
        family="Sopros / Metais", concert_key="C", transposition_semitones=0,
        written_low=40, written_high=65, preferred_low=43, preferred_high=60,
        clef="bass", music21_instrument="Trombone",
    ),
    "clarinet": InstrumentDefinition(
        id="clarinet", name="Clarinete em Bb", short_name="Cl.",
        family="Sopros / Metais", concert_key="Bb", transposition_semitones=2,
        written_low=55, written_high=91, preferred_low=57, preferred_high=88,
        clef="treble", music21_instrument="Clarinet",
    ),
}

# Ordem aproximada de registro (agudo -> grave) para distribuir vozes.
REGISTER_ORDER = ["trumpet", "clarinet", "alto_sax", "tenor_sax", "trombone"]

MAX_ARRANGEMENT_INSTRUMENTS = 5

SUPPORTED_ARRANGE_MODES = ["automatic", "melody", "harmony"]


def get_instrument(inst_id: str) -> Optional[InstrumentDefinition]:
    return INSTRUMENTS.get(inst_id)


def list_instruments() -> List[InstrumentDefinition]:
    return [INSTRUMENTS[k] for k in ("alto_sax", "tenor_sax", "trumpet", "trombone", "clarinet")]


def concert_to_written(pitch: int, definition: InstrumentDefinition) -> int:
    """Concert pitch (som real) -> written pitch da parte transposta."""
    return int(pitch) + int(definition.transposition_semitones)


def written_to_concert(pitch: int, definition: InstrumentDefinition) -> int:
    return int(pitch) - int(definition.transposition_semitones)


def concert_low_high(definition: InstrumentDefinition) -> tuple:
    """Ranges convertidos para concert pitch (para checagem do arranjador)."""
    t = definition.transposition_semitones
    return (
        definition.written_low - t, definition.written_high - t,
        definition.preferred_low - t, definition.preferred_high - t,
    )


def fit_range(
    pitch: int, definition: InstrumentDefinition
) -> tuple:
    """Tenta encaixar pitch (concert) na tessitura preferida com ±12.

    Retorna (pitch_ajustado_ou_None, adjustments). None = impossível mesmo
    com oitavas (fora do absoluto) -> chamador gera pausa + warning, nunca
    pitch impossível silencioso.
    """
    abs_low, abs_high, pref_low, pref_high = concert_low_high(definition)
    p = int(pitch)
    adjustments = 0
    # Sobe primeiro se grave demais, desce se agudo demais.
    while p < pref_low and p + 12 <= abs_high:
        p += 12
        adjustments += 1
    while p > pref_high and p - 12 >= abs_low:
        p -= 12
        adjustments += 1
    if p < abs_low or p > abs_high:
        return None, adjustments
    return p, adjustments
