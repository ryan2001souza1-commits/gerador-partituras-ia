"""
Etapa 8.1 — Infraestrutura de chunks para músicas longas.

Divide áudio em chunks determinísticos com sobreposição para:
- não perder notas sustentadas nas fronteiras;
- não perder ataques;
- permitir processamento em segmentos independentes.

Configuração (via variáveis de ambiente):
- CHUNK_DURATION: duração de cada chunk em segundos (default: 60)
- CHUNK_OVERLAP: sobreposição entre chunks em segundos (default: 3)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional


def _validate_chunk_config(chunk_duration: float, overlap: float) -> None:
    """Valida configuração de chunk — bug fix #6/#7.

    Antes: valores inválidos (0, negativo, NaN, Infinity) produziam
    chunks vazios, negativos ou NaN sem erro. Agora: ValueError claro.
    """
    if not math.isfinite(chunk_duration) or chunk_duration <= 0:
        raise ValueError(
            f"chunk_duration inválido: {chunk_duration!r}. "
            f"Deve ser um número finito > 0.")
    if not math.isfinite(overlap) or overlap < 0:
        raise ValueError(
            f"overlap inválido: {overlap!r}. "
            f"Deve ser um número finito >= 0.")
    if overlap >= chunk_duration:
        raise ValueError(
            f"overlap ({overlap}) deve ser < chunk_duration ({chunk_duration}). "
            f"Configuração com overlap >= duração não é suportada.")


def _safe_env_float(name: str, default: str) -> float:
    """Lê env var como float com fallback seguro — bug fix #7.

    Antes: CHUNK_DURATION='abc' crashava o import do servidor inteiro.
    Agora: loga warning e usa default.
    """
    raw = os.getenv(name, default)
    try:
        v = float(raw)
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"valor não positivo/finuto: {v}")
        return v
    except (TypeError, ValueError):
        import logging
        logging.getLogger("uvicorn.error").warning(
            f"Env var {name}={raw!r} inválida; usando default {default}")
        return float(default)


# Configuração central (não espalhar números mágicos)
# Bug fix #7: env vars validadas com fallback seguro
CHUNK_DURATION = _safe_env_float("CHUNK_DURATION", "60")   # 60 segundos
CHUNK_OVERLAP = _safe_env_float("CHUNK_OVERLAP", "3")      # 3 segundos
# Limite para ativar fast path (áudio curto = 1 chunk)
CHUNK_THRESHOLD = _safe_env_float("CHUNK_THRESHOLD", "90")  # acima disso, usa chunks


@dataclass
class ChunkInfo:
    """Um chunk de áudio para processamento segmentado."""
    index: int
    start_seconds: float
    end_seconds: float
    overlap_before: float  # segundos de sobreposição com o chunk anterior
    overlap_after: float   # segundos de sobreposição com o próximo chunk
    global_offset: float = 0.0  # offset global (para conversão local→global)

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds

    def local_to_global(self, local_time: float) -> float:
        """Converte tempo local do chunk para tempo global do áudio."""
        return self.start_seconds + local_time

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "overlap_before": self.overlap_before,
            "overlap_after": self.overlap_after,
            "global_offset": self.global_offset,
        }


def create_chunks(duration_seconds: float,
                  chunk_duration: Optional[float] = None,
                  overlap: Optional[float] = None) -> List[ChunkInfo]:
    """Divide áudio em chunks determinísticos com sobreposição.

    Fast path: se duration <= chunk_duration, retorna 1 chunk único.
    Long path: divide em chunks com overlap para não perder notas.

    A divisão é determinística: mesma duração → mesma lista de chunks.
    """
    if chunk_duration is None:
        chunk_duration = CHUNK_DURATION
    if overlap is None:
        overlap = CHUNK_OVERLAP

    # BUG FIX #6: valida config — valores inválidos agora falham claramente
    _validate_chunk_config(chunk_duration, overlap)

    if duration_seconds <= 0:
        return []
    # BUG FIX: duração NaN/negativa/infinita → vazio, sem crash
    if not math.isfinite(duration_seconds):
        return []

    # Fast path: áudio curto = 1 chunk, sem divisão
    if duration_seconds <= chunk_duration + overlap:
        return [ChunkInfo(
            index=0,
            start_seconds=0.0,
            end_seconds=duration_seconds,
            overlap_before=0.0,
            overlap_after=0.0,
        )]

    chunks: List[ChunkInfo] = []
    step = chunk_duration - overlap  # avanço efetivo por chunk
    if step <= 0:
        step = max(chunk_duration / 2.0, 1.0)  # fallback seguro

    start = 0.0
    index = 0
    while start < duration_seconds:
        end = min(start + chunk_duration, duration_seconds)
        ob_before = overlap if index > 0 else 0.0
        ob_after = overlap if end < duration_seconds else 0.0
        chunks.append(ChunkInfo(
            index=index,
            start_seconds=start,
            end_seconds=end,
            overlap_before=ob_before,
            overlap_after=ob_after,
        ))
        index += 1
        start += step
        # Evita chunk minúsculo no final: se o que sobra cabe no overlap
        # (ou é exatamente o overlap = zero conteúdo único), para.
        # Bug fix: era `<` e criava chunk de 0s únicos quando
        # duration - start == overlap exatamente (ex: d=117, step=57).
        if duration_seconds - start <= overlap:
            break

    return chunks


def is_long_audio(duration_seconds: float,
                  threshold: Optional[float] = None) -> bool:
    """Verifica se o áudio deve usar o pipeline de chunks."""
    if threshold is None:
        threshold = CHUNK_THRESHOLD
    return duration_seconds > threshold


# ---------------------------------------------------------------------------
# Stitching de notas (pós-processamento dos chunks)
# ---------------------------------------------------------------------------

def _notes_overlap(a: dict, b: dict, min_overlap: float = 0.05) -> bool:
    """Verifica se duas notas do mesmo pitch se sobrepõem temporalmente."""
    if a.get("pitch") != b.get("pitch"):
        return False
    a_start, a_end = float(a["start"]), float(a["end"])
    b_start, b_end = float(b["start"]), float(b["end"])
    overlap = min(a_end, b_end) - max(a_start, b_start)
    return overlap >= min_overlap


def _note_score(n: dict) -> float:
    """Score de qualidade da nota (confidence + amplitude)."""
    try:
        conf = float(n.get("confidence", 0.0))
        amp = float(n.get("amplitude", 0.0))
        return conf * 0.6 + amp * 0.4
    except (TypeError, ValueError):
        return 0.0


def stitch_notes(chunk_results: List[List[dict]],
                 chunks: List[ChunkInfo]) -> tuple:
    """Funde resultados de chunks: converte para tempo global + dedupe.

    Args:
        chunk_results: lista de listas de notas (tempos LOCAIS por chunk).
        chunks: lista de ChunkInfo correspondente.

    Returns:
        (stitched_notes, stats) — notas em tempo global, sem duplicatas de
        overlap, com notas atravessando fronteiras unidas.
    """
    stats = {
        "raw_notes": 0,
        "notes_after_stitch": 0,
        "overlap_duplicates_removed": 0,
        "cross_chunk_notes_merged": 0,
        "boundary_notes_preserved": 0,
    }

    if not chunk_results or not chunks:
        return [], stats

    # 1. Converte tempos locais para globais
    all_notes = []
    for result_list, chunk in zip(chunk_results, chunks):
        stats["raw_notes"] += len(result_list)
        for n in result_list:
            global_note = dict(n)
            global_note["start"] = chunk.local_to_global(float(n["start"]))
            global_note["end"] = chunk.local_to_global(float(n["end"]))
            global_note["_chunk"] = chunk.index
            all_notes.append(global_note)

    all_notes.sort(key=lambda n: (n["start"], n.get("pitch", 0)))

    # 2. Dedupe na região de overlap: mesma nota detectada em 2 chunks
    #    → mantém a de maior score (confidence/amplitude)
    kept: List[dict] = []
    removed_overlap = 0
    for note in all_notes:
        is_dup = False
        for i, existing in enumerate(kept):
            if _notes_overlap(existing, note):
                # Mesma nota detectada em chunks distintos (overlap)
                if note["_chunk"] != existing["_chunk"]:
                    # Mantém a de maior score
                    if _note_score(note) > _note_score(existing):
                        kept[i] = note
                    removed_overlap += 1
                else:
                    # Duplicata dentro do mesmo chunk (raro)
                    if _note_score(note) > _note_score(existing):
                        kept[i] = note
                    removed_overlap += 1
                is_dup = True
                break
        if not is_dup:
            kept.append(note)
    stats["overlap_duplicates_removed"] = removed_overlap

    # 3. Une notas atravessando fronteiras (continuidade)
    #    Mesmo pitch, gap pequeno entre fim de um e início do outro
    #    Tolerância documentada: 0.30s (notas cortadas na fronteira de chunk
    #    podem ter gap de até ~0.3s antes do próximo chunk detectar)
    merged: List[dict] = []
    cross_merged = 0
    for note in kept:
        if merged:
            last = merged[-1]
            same_pitch = last.get("pitch") == note.get("pitch")
            gap = float(note["start"]) - float(last["end"])
            # Gap pequeno e chunks adjacentes → nota sustentada única
            adjacent = abs(note["_chunk"] - last["_chunk"]) <= 1
            if same_pitch and 0 <= gap <= 0.30 and adjacent:
                # Une: estende a duração da nota anterior
                last["end"] = note["end"]
                if _note_score(note) > _note_score(last):
                    last["confidence"] = note.get("confidence", last.get("confidence"))
                    last["amplitude"] = note.get("amplitude", last.get("amplitude"))
                cross_merged += 1
                continue
        merged.append(note)
    stats["cross_chunk_notes_merged"] = cross_merged
    stats["notes_after_stitch"] = len(merged)

    # Limpa metadados internos
    for n in merged:
        n.pop("_chunk", None)

    return merged, stats