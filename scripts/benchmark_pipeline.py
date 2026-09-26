#!/usr/bin/env python
"""
Benchmark do pipeline — Fase 18/19.

Mede cada estágio com time.perf_counter() (nunca time.time()).
Entrada: arquivo local (não versionado).
Saída: stage, seconds, RTF.
"""
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

# Arquivo de teste: 203s MP3
TEST_FILE = BASE / "uploads" / "ea7da280-ca60-4875-8348-ac51143d2e51.mp3"


def bench(label):
    """Context manager que mede e imprime tempo de um estágio."""
    class Ctx:
        def __init__(self):
            self.t0 = None
            self.elapsed = 0.0
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, *exc):
            self.elapsed = time.perf_counter() - self.t0
            print(f"  {label:35s}: {self.elapsed:8.3f}s")
            return False
    return Ctx()


def main():
    if not TEST_FILE.is_file():
        print(f"ERRO: arquivo de teste não encontrado: {TEST_FILE}")
        sys.exit(1)

    print(f"Arquivo: {TEST_FILE.name}")
    print(f"Tamanho: {TEST_FILE.stat().st_size / 1024 / 1024:.1f} MB")
    print()

    # ---------------------------------------------------------------
    # 1. FFprobe (técnica)
    # ---------------------------------------------------------------
    print("=== FFprobe ===")
    with bench("ffprobe"):
        import subprocess
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(TEST_FILE)],
            capture_output=True, text=True, timeout=15)
        duration = float(r.stdout.strip())
    print(f"  Duração: {duration:.1f}s")
    print()

    # ---------------------------------------------------------------
    # 2. Análise musical (BPM + key)
    # ---------------------------------------------------------------
    print("=== Análise musical (analyze_music) ===")
    from backend.audio.music_analysis import analyze_music, get_beat_grid
    with bench("analyze_music (total)"):
        result = analyze_music(TEST_FILE, duration_probe=duration)
    print(f"  BPM: {result.bpm_rounded} (conf={result.bpm_confidence})")
    print(f"  Key: {result.key} {result.mode} (conf={result.key_confidence})")
    print()

    # ---------------------------------------------------------------
    # 3. Beat grid (separado — demonstra redundância)
    # ---------------------------------------------------------------
    print("=== Beat grid (get_beat_grid) ===")
    with bench("get_beat_grid (total, re-decodifica!)"):
        grid = get_beat_grid(TEST_FILE)
    print(f"  first_beat: {grid.get('first_beat_time')}")
    print()

    # ---------------------------------------------------------------
    # 4. Decompor analyze_music internamente
    # ---------------------------------------------------------------
    print("=== Decomposição: onde está o tempo? ===")
    import librosa
    import numpy as np

    # 4a. FFmpeg decode
    with bench("ffmpeg decode (uma vez)"):
        from backend.audio.music_analysis import _decode_to_wav
        wav_path = _decode_to_wav(TEST_FILE)

    try:
        # 4b. librosa.load
        with bench("librosa.load (uma vez)"):
            y, sr = librosa.load(str(wav_path), sr=22050, mono=True)

        # 4c. HPSS (isolado)
        with bench("HPSS (isolado, uma vez)"):
            y_harm, y_perc = librosa.effects.hpss(y)

        # 4d. onset_strength (isolado)
        with bench("onset_strength (isolado)"):
            onset = librosa.onset.onset_strength(y=y_perc, sr=sr)

        # 4e. beat_track (isolado)
        with bench("beat_track (isolado)"):
            tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr)

        # 4f. Key detection (isolado)
        with bench("_estimate_key (isolado)"):
            from backend.audio.music_analysis import _estimate_key
            key, mode, key_conf, best = _estimate_key(y_harm, sr)
    finally:
        wav_path.unlink(missing_ok=True)

    print()
    print(f"=== RESUMO ===")
    print(f"Duração do áudio: {duration:.1f}s")
    print(f"Total análise musical (com redundância): ~{result.bpm and 'medido acima'}")
    print()

    # ---------------------------------------------------------------
    # 5. Basic Pitch (usando stem de 15s existente)
    # ---------------------------------------------------------------
    stem_dir = BASE / "stems" / "aefde683-36e3-40d2-a296-a2b66f81740f"
    if stem_dir.is_dir():
        print("=== Basic Pitch (stem de 15s) ===")
        from backend.audio.transcriber import (
            get_basic_pitch_python, is_basic_pitch_available,
            transcribe_stem_async, BASIC_PITCH_TIMEOUT,
        )
        import asyncio

        if is_basic_pitch_available():
            py = get_basic_pitch_python()
            import subprocess

            # Medir model load: tempo até primeiro output
            for stem_name in ["vocals", "bass", "other"]:
                stem_path = stem_dir / f"{stem_name}.wav"
                if stem_path.is_file():
                    # Remover cache para medir fresh
                    import shutil
                    from backend.audio.transcriber import (
                        MIDI_DIR, TRANSCRIPTIONS_DIR,
                    )
                    midi_p = MIDI_DIR / "aefde683-36e3-40d2-a296-a2b66f81740f" / f"{stem_name}.mid"
                    json_p = TRANSCRIPTIONS_DIR / "aefde683-36e3-40d2-a296-a2b66f81740f" / f"{stem_name}.json"
                    midi_p.unlink(missing_ok=True)
                    json_p.unlink(missing_ok=True)
                    # Remover cache de chunks se existir
                    chunk_dir = TRANSCRIPTIONS_DIR / "aefde683-36e3-40d2-a296-a2b66f81740f" / "chunks" / stem_name
                    shutil.rmtree(chunk_dir, ignore_errors=True)

                    with bench(f"basic_pitch {stem_name} (15s, fresh)"):
                        result_bp = asyncio.run(
                            transcribe_stem_async(stem_path, "aefde683-36e3-40d2-a296-a2b66f81740f",
                                                  stem_name, timeout=BASIC_PITCH_TIMEOUT))
                    print(f"    notes: {result_bp.get('notes_count', '?')}")
        else:
            print("  Basic Pitch não disponível")

    print()
    print("=== BENCHMARK CONCLUÍDO ===")


if __name__ == "__main__":
    main()
