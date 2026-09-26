"""
Testes Etapa 8.1 — chunks para músicas longas + stitching.

Cobre:
- fast path (áudio curto = 1 chunk);
- divisão determinística com overlap;
- stitching: nota atravessando fronteira vira 1 nota;
- ataques diferentes não são fundidos;
- long sustain através de múltiplos chunks.
"""
import pytest

from backend.audio.chunking import (
    CHUNK_DURATION,
    CHUNK_OVERLAP,
    ChunkInfo,
    create_chunks,
    is_long_audio,
    stitch_notes,
)


# ---------------------------------------------------------------------------
# create_chunks — divisão determinística
# ---------------------------------------------------------------------------

def test_fast_path_short_audio():
    """Áudio curto: 1 chunk único, sem divisão (item 108)."""
    chunks = create_chunks(30.0, chunk_duration=60.0)
    assert len(chunks) == 1
    assert chunks[0].start_seconds == 0.0
    assert chunks[0].end_seconds == 30.0
    assert chunks[0].overlap_before == 0.0
    assert chunks[0].overlap_after == 0.0


def test_fast_path_exactly_chunk_duration():
    """Duração exatamente igual ao chunk: ainda 1 chunk."""
    chunks = create_chunks(60.0, chunk_duration=60.0)
    assert len(chunks) == 1
    assert chunks[0].end_seconds == 60.0


def test_long_audio_multiple_chunks():
    """Áudio longo: múltiplos chunks com overlap (item 68)."""
    chunks = create_chunks(180.0, chunk_duration=60.0, overlap=3.0)
    assert len(chunks) >= 3
    # Primeiro chunk começa em 0
    assert chunks[0].start_seconds == 0.0
    # Todos os chunks têm overlap com o anterior (exceto o primeiro)
    for i in range(1, len(chunks)):
        assert chunks[i].overlap_before > 0
        # Início do chunk N sobreponhe o fim do chunk N-1
        assert chunks[i].start_seconds < chunks[i - 1].end_seconds


def test_chunks_deterministic():
    """Mesma duração → mesma divisão (item 66)."""
    a = create_chunks(300.0, chunk_duration=60.0, overlap=3.0)
    b = create_chunks(300.0, chunk_duration=60.0, overlap=3.0)
    assert len(a) == len(b)
    for ca, cb in zip(a, b):
        assert ca.start_seconds == cb.start_seconds
        assert ca.end_seconds == cb.end_seconds


def test_chunks_cover_full_duration():
    """Chunks cobrem a duração completa do áudio."""
    duration = 250.0
    chunks = create_chunks(duration, chunk_duration=60.0, overlap=3.0)
    assert chunks[0].start_seconds == 0.0
    assert chunks[-1].end_seconds == pytest.approx(duration)


def test_zero_duration_returns_empty():
    chunks = create_chunks(0.0)
    assert chunks == []


def test_negative_duration_returns_empty():
    chunks = create_chunks(-5.0)
    assert chunks == []


def test_chunkinfo_local_to_global():
    c = ChunkInfo(index=1, start_seconds=60.0, end_seconds=120.0,
                  overlap_before=3.0, overlap_after=3.0)
    assert c.local_to_global(0.0) == 60.0
    assert c.local_to_global(10.5) == 70.5


def test_is_long_audio():
    assert not is_long_audio(30.0, threshold=90.0)
    assert not is_long_audio(90.0, threshold=90.0)
    assert is_long_audio(120.0, threshold=90.0)


# ---------------------------------------------------------------------------
# Stitching — notas atravessando fronteiras (itens 69-71, 99-102)
# ---------------------------------------------------------------------------

def _mk_chunks_2():
    """Dois chunks de 60s com overlap de 3s: chunk0=0-60, chunk1=57-117."""
    return [
        ChunkInfo(0, 0.0, 60.0, 0.0, 3.0),
        ChunkInfo(1, 57.0, 117.0, 3.0, 0.0),
    ]


def test_stitch_note_crossing_boundary():
    """Item 99: nota C4 começa 0.5s antes da fronteira, termina 1s depois.
    Após stitching: 1 nota C4 contínua, não duas."""
    chunks = _mk_chunks_2()
    # Chunk 0 detecta C4 de 58.2 → 60.0 (local)
    # Chunk 1 detecta C4 de 60.2 local → global 57+3.2=60.2 → 61.0
    chunk0_notes = [{"pitch": 60, "start": 58.2, "end": 60.0,
                     "confidence": 0.9, "amplitude": 0.8}]
    chunk1_notes = [{"pitch": 60, "start": 3.2, "end": 4.0,
                     "confidence": 0.85, "amplitude": 0.7}]
    stitched, stats = stitch_notes([chunk0_notes, chunk1_notes], chunks)
    # Deve resultar em 1 nota C4 contínua
    c4_notes = [n for n in stitched if n["pitch"] == 60]
    assert len(c4_notes) == 1
    assert c4_notes[0]["start"] == pytest.approx(58.2)
    assert c4_notes[0]["end"] == pytest.approx(61.0)
    assert stats["cross_chunk_notes_merged"] >= 1


def test_stitch_attack_preserved_once():
    """Item 100: ataque na fronteira preservado uma única vez."""
    chunks = _mk_chunks_2()
    # Nota começa exatamente na região de overlap (detectada nos 2 chunks)
    chunk0_notes = [{"pitch": 62, "start": 58.0, "end": 62.0,
                     "confidence": 0.9, "amplitude": 0.8}]
    chunk1_notes = [{"pitch": 62, "start": 1.0, "end": 5.0,
                     "confidence": 0.7, "amplitude": 0.6}]
    stitched, stats = stitch_notes([chunk0_notes, chunk1_notes], chunks)
    d4_notes = [n for n in stitched if n["pitch"] == 62]
    assert len(d4_notes) == 1
    assert stats["overlap_duplicates_removed"] >= 1
    # Preserva a versão de maior score (chunk 0: conf 0.9)
    assert d4_notes[0]["confidence"] == pytest.approx(0.9)


def test_stitch_different_notes_not_merged():
    """Item 101: duas notas realmente diferentes (C4 antes, D4 depois)
    não são fundidas."""
    chunks = _mk_chunks_2()
    chunk0_notes = [{"pitch": 60, "start": 55.0, "end": 59.5,
                     "confidence": 0.9, "amplitude": 0.8}]
    chunk1_notes = [{"pitch": 62, "start": 2.5, "end": 6.0,
                     "confidence": 0.9, "amplitude": 0.8}]
    stitched, _ = stitch_notes([chunk0_notes, chunk1_notes], chunks)
    pitches = sorted(n["pitch"] for n in stitched)
    assert pitches == [60, 62]
    assert len(stitched) == 2


def test_stitch_same_pitch_far_apart_not_merged():
    """Mesmo pitch mas distantes temporalmente: notas separadas."""
    chunks = _mk_chunks_2()
    chunk0_notes = [{"pitch": 60, "start": 1.0, "end": 2.0,
                     "confidence": 0.9, "amplitude": 0.8}]
    chunk1_notes = [{"pitch": 60, "start": 50.0, "end": 51.0,
                     "confidence": 0.9, "amplitude": 0.8}]
    stitched, _ = stitch_notes([chunk0_notes, chunk1_notes], chunks)
    # Globais: 1-2 e 107-108 → não são contíguas
    assert len(stitched) == 2


def test_stitch_long_sustain_multiple_chunks():
    """Item 102: nota longa atravessa múltiplos chunks → 1 nota ligada."""
    chunks = [
        ChunkInfo(0, 0.0, 60.0, 0.0, 3.0),
        ChunkInfo(1, 57.0, 117.0, 3.0, 3.0),
        ChunkInfo(2, 114.0, 174.0, 3.0, 0.0),
    ]
    # C4 sustentada de 10s a 130s (global) — detectada em 3 chunks
    chunk0_notes = [{"pitch": 60, "start": 10.0, "end": 60.0,
                     "confidence": 0.9, "amplitude": 0.8}]
    chunk1_notes = [{"pitch": 60, "start": 3.0, "end": 60.0,
                     "confidence": 0.9, "amplitude": 0.8}]
    chunk2_notes = [{"pitch": 60, "start": 3.0, "end": 16.0,
                     "confidence": 0.9, "amplitude": 0.8}]
    stitched, stats = stitch_notes(
        [chunk0_notes, chunk1_notes, chunk2_notes], chunks)
    c4_notes = [n for n in stitched if n["pitch"] == 60]
    # Deve resultar em 1 nota longa (ou no mínimo 2 se o gap exceder
    # a tolerância, mas NÃO 3 retriggers)
    assert len(c4_notes) <= 2
    assert stats["cross_chunk_notes_merged"] >= 1


def test_stitch_empty_chunks():
    stitched, stats = stitch_notes([[], []], _mk_chunks_2())
    assert stitched == []
    assert stats["raw_notes"] == 0


def test_stitch_empty_input():
    stitched, stats = stitch_notes([], [])
    assert stitched == []
    assert stats == {"raw_notes": 0, "notes_after_stitch": 0,
                     "overlap_duplicates_removed": 0,
                     "cross_chunk_notes_merged": 0,
                     "boundary_notes_preserved": 0}


def test_stitch_preserves_best_confidence():
    """Na região de overlap, preserva a versão com maior confidence."""
    chunks = _mk_chunks_2()
    chunk0_notes = [{"pitch": 64, "start": 58.0, "end": 61.0,
                     "confidence": 0.5, "amplitude": 0.4}]
    chunk1_notes = [{"pitch": 64, "start": 1.0, "end": 4.0,
                     "confidence": 0.95, "amplitude": 0.9}]
    stitched, _ = stitch_notes([chunk0_notes, chunk1_notes], chunks)
    e4 = [n for n in stitched if n["pitch"] == 64]
    assert len(e4) == 1
    assert e4[0]["confidence"] == pytest.approx(0.95)


def test_stats_keys_present():
    """Item 97: métricas de precisão presentes no stats."""
    chunks = _mk_chunks_2()
    _, stats = stitch_notes(
        [[{"pitch": 60, "start": 1.0, "end": 2.0, "confidence": 0.9,
           "amplitude": 0.8}]],
        chunks[:1])
    for key in ("raw_notes", "notes_after_stitch",
                "overlap_duplicates_removed", "cross_chunk_notes_merged"):
        assert key in stats


# ---------------------------------------------------------------------------
# Chunk com áudio real (integração leve com numpy)
# ---------------------------------------------------------------------------

def test_chunks_for_5min_audio():
    """Áudio de 5 minutos: número de chunks previsível (item 103)."""
    duration = 300.0  # 5 minutos
    chunks = create_chunks(duration, chunk_duration=60.0, overlap=3.0)
    # step = 57s; 300/57 ≈ 5.3 → ~5-6 chunks
    assert 4 <= len(chunks) <= 7
    assert is_long_audio(duration, threshold=90.0)


def test_chunks_for_20min_audio():
    """Áudio de 20 minutos (limite): número de chunks previsível."""
    duration = 1200.0
    chunks = create_chunks(duration, chunk_duration=60.0, overlap=3.0)
    # step = 57s; 1200/57 ≈ 21
    assert 18 <= len(chunks) <= 25
