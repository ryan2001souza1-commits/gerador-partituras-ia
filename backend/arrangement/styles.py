"""
Etapa 8 — estilos de arranjo (determinísticos, sem ML/generativo).

Estilos NÃO inventam melodias: apenas filtram bateria, ajustam dinâmica,
respiro e sustentação harmônica. `automatic` é neutro (Etapa 7.2.1 +
bateria + dinâmicas seguras).
"""

from __future__ import annotations

from typing import Any, Dict, List

ARRANGEMENT_STYLES = ["automatic", "pop", "rock", "ballad", "brass_band"]

STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "automatic": {
        "drum_min_strength": 0.0,
        "hat_grid": 0.5,          # hats em colcheias
        "dynamics_bias": 0,
        "breath_mult": 1.0,
        "harm_min_dur_mult": 1.0,
        "accent_downbeats": False,
    },
    "pop": {
        "drum_min_strength": 0.05,
        "hat_grid": 0.5,          # hi-hat 1/8 predominante
        "dynamics_bias": 0,       # dinâmica moderada
        "breath_mult": 1.0,
        "harm_min_dur_mult": 1.0,
        "accent_downbeats": False,
    },
    "rock": {
        "drum_min_strength": 0.0,  # bateria mais presente
        "hat_grid": 0.5,
        "dynamics_bias": 1,       # um nível acima
        "breath_mult": 1.0,
        "harm_min_dur_mult": 1.0,
        "accent_downbeats": True,  # kick/snare acentuados
    },
    "ballad": {
        "drum_min_strength": 0.15,  # menos eventos fracos
        "hat_grid": 1.0,           # sem hi-hat agressivo: só tempos
        "dynamics_bias": -1,       # mais suave
        "breath_mult": 1.5,        # respirações maiores
        "harm_min_dur_mult": 2.0,  # harmonias longas
        "accent_downbeats": False,
    },
    "brass_band": {
        "drum_min_strength": 0.05,
        "hat_grid": 0.5,
        "dynamics_bias": 0,
        "breath_mult": 1.0,
        "harm_min_dur_mult": 1.0,
        "accent_downbeats": True,  # ataques alinhados e definidos
    },
}


def validate_style(value: Any) -> str:
    v = str(value or "").strip().lower()
    if v not in ARRANGEMENT_STYLES:
        raise ValueError(f"Estilo inválido. Permitidos: {', '.join(ARRANGEMENT_STYLES)}.")
    return v


def get_style(name: str) -> Dict[str, Any]:
    return dict(STYLE_PRESETS.get(name, STYLE_PRESETS["automatic"]))


def apply_style_to_drums(events: List[Dict[str, Any]], style: str) -> List[Dict[str, Any]]:
    """Filtra eventos de bateria pelo estilo (determinístico, sem inventar).

    - rock/automatic/pop: mantém tudo acima do mínimo;
    - ballad: hats só nos tempos (grid 1.0) + corta fracos.
    Retorna nova lista (não muta a entrada).
    """
    preset = get_style(style)
    min_s = float(preset["drum_min_strength"])
    grid = float(preset["hat_grid"])
    out = []
    for e in events:
        if float(e.get("strength", 0)) < min_s:
            continue
        if e.get("instrument") in ("closed_hihat", "open_hihat") and grid >= 1.0:
            if abs(float(e.get("beat", 0)) - round(float(e.get("beat", 0)))) > 1e-6:
                continue  # ballad: hats fora do tempo saem
        out.append(dict(e))
    return out
