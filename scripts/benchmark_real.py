"""
Etapa 8.3.1 — Benchmark real do pipeline.
"""
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

FID = "e29d16fd-68f5-45ae-b5ec-c7599491a68d"
UPLOAD = BASE / "uploads" / f"{FID}.mp3"

if not UPLOAD.is_file():
    print(f"ERRO: upload nao encontrado {UPLOAD}")
    sys.exit(1)

print(f"Arquivo: {FID} ({UPLOAD.stat().st_size / 1024 / 1024:.1f} MB)")
print()

from backend.pipeline import cache_identity as ci

# 1. AUDIO HASH
print("=== AUDIO HASH ===")
t0 = time.perf_counter()
audio_hash = ci.compute_file_hash(UPLOAD)
hash_time = time.perf_counter() - t0
print(f"  SHA-256: {audio_hash[:16]}... ({hash_time:.3f}s)")
print()

# 2. ANALYSIS COLD
print("=== ANALYSIS COLD ===")
from backend.audio.analysis_cache import clear_analysis_cache
clear_analysis_cache(FID)
ci.get_analysis_metadata_path(FID).unlink(missing_ok=True)

t0 = time.perf_counter()
from backend.audio.music_analysis import analyze_music
result_cold = analyze_music(UPLOAD, file_id=FID)
analysis_cold = time.perf_counter() - t0
ci.save_analysis_metadata(FID, audio_hash)  # Salva metadata
print(f"  Tempo: {analysis_cold:.1f}s")
print(f"  BPM: {result_cold.bpm_rounded}")
print(f"  BPM stability: {result_cold.bpm_stability}")
print(f"  Key: {result_cold.key} {result_cold.mode}")
print(f"  Key confidence: {result_cold.key_confidence:.4f}")
print()

# 3. ANALYSIS WARM
print("=== ANALYSIS WARM (cache) ===")
t0 = time.perf_counter()
result_warm = analyze_music(UPLOAD, file_id=FID)
analysis_warm = time.perf_counter() - t0
print(f"  Tempo: {analysis_warm:.3f}s")
print(f"  BPM: {result_warm.bpm_rounded}")
print(f"  BPM stability: {result_warm.bpm_stability}")
print(f"  Key: {result_warm.key} {result_warm.mode}")
print(f"  Key confidence: {result_warm.key_confidence:.4f}")
print()

# Verifica identidade
same_bpm = abs(result_cold.bpm - result_warm.bpm) < 0.01
same_key = result_cold.key == result_warm.key
same_mode = result_cold.mode == result_warm.mode
same_conf = abs((result_cold.key_confidence or 0) - (result_warm.key_confidence or 0)) < 0.001
print(f"  BPM identical: {'[OK]' if same_bpm else '[FAIL]'}")
print(f"  Key identical: {'[OK]' if same_key and same_mode else '[FAIL]'}")
print(f"  Confidence identical: {'[OK]' if same_conf else '[FAIL]'}")
print(f"  SPEEDUP: {analysis_cold / max(analysis_warm, 0.001):.0f}x")
print()

# 4. STEM HASHES
print("=== STEM HASHES ===")
t0 = time.perf_counter()
stem_hashes = ci.get_stem_hashes(FID)
hash_all_time = time.perf_counter() - t0
if stem_hashes:
    for stem, h in stem_hashes.items():
        print(f"  {stem}: {h[:16]}...")
    print(f"  Hashing 4 stems: {hash_all_time:.3f}s")
    ci.save_demucs_metadata(FID, audio_hash)
    print("  Demucs metadata: SAVED")

    from backend.audio.transcriber import BASIC_PITCH_DEFAULTS, STEM_FREQ_RANGES
    trans_config = {
        "defaults": dict(BASIC_PITCH_DEFAULTS),
        "stem_freq_ranges": {k: dict(v) if v else None for k, v in STEM_FREQ_RANGES.items()},
    }
    trans_config_hash = ci.compute_config_hash(trans_config)
    for stem in ["vocals", "bass", "other"]:
        ci.save_transcription_metadata(FID, stem, stem_hashes[stem], trans_config_hash)
    print("  Transcription metadata: SAVED")

    from backend.drums.drum_utils import DRUM_VERSION
    drums_config = {"drum_version": DRUM_VERSION, "cleanup_profile": "natural",
                    "time_signature": "4/4"}
    drums_config_hash = ci.compute_config_hash(drums_config)
    ci.save_drums_metadata(FID, stem_hashes.get("drums", ""), drums_config_hash)
    print("  Drums metadata: SAVED")
print()

# 5. CACHE VALIDATION
print("=== CACHE VALIDATION ===")
t0 = time.perf_counter()
a_valid = ci.is_analysis_cache_valid(FID, audio_hash)
d_valid = ci.is_demucs_cache_valid(FID, audio_hash)
t_valid = ci.are_all_transcriptions_valid(FID, trans_config_hash)
dr_valid = ci.is_drums_cache_valid(FID, stem_hashes.get("drums", ""), drums_config_hash)
validation_time = time.perf_counter() - t0
print(f"  Analysis: {a_valid}")
print(f"  Demucs: {d_valid}")
print(f"  Transcription: {t_valid}")
print(f"  Drums: {dr_valid}")
print(f"  Total validation time: {validation_time:.3f}s")
print()

# 6. INVALIDATION
print("=== INVALIDATION BY AUDIO HASH CHANGE ===")
fake = "fake_hash_diff"
print(f"  Analysis with fake hash: {ci.is_analysis_cache_valid(FID, fake)}")
print(f"  Demucs with fake hash: {ci.is_demucs_cache_valid(FID, fake)}")
print()

# 7. SUMMARY
print("=" * 60)
print("BENCHMARK SUMMARY")
print("=" * 60)
print(f"{'Stage':<20} {'Cold':>10} {'Warm':>10} {'Speedup':>10}")
print("-" * 60)
print(f"{'Analysis':<20} {analysis_cold:>8.1f}s {analysis_warm:>8.3f}s {analysis_cold/max(analysis_warm,0.001):>9.0f}x")
print(f"{'Stem hashing':<20} {hash_all_time:>8.3f}s {'--':>10} {'--':>10}")
print(f"{'Cache validation':<20} {'--':>10} {validation_time:>8.3f}s {'--':>10}")
print("-" * 60)
print()
print("NOT MEASURED (requires full execution):")
print("  Demucs cold: ~10min")
print("  Basic Pitch cold: ~1min per stem")
print("  Score cold: ~10s")
print("  Arrangement cold: ~10s")
print()
print("COLD values (measured):")
print(f"  BPM: {result_cold.bpm_rounded}")
print(f"  BPM stability: {result_cold.bpm_stability}")
print(f"  Key: {result_cold.key} {result_cold.mode}")
print(f"  Key confidence: {result_cold.key_confidence:.4f}")
print(f"  Key window agreement: {result_cold.key_window_agreement}")
print(f"  Key analysis used: {result_cold.key_analysis_used}")
print(f"  Beat grid mean error: {result_cold.beat_grid_mean_error_ms}ms")
print(f"  Beat grid p95 error: {result_cold.beat_grid_p95_error_ms}ms")
print()
print("WARM values (cache hit) — MUST BE IDENTICAL:")
print(f"  BPM: {result_warm.bpm_rounded} [{'OK' if same_bpm else 'DIFFERENT!'}]")
print(f"  Key: {result_warm.key} {result_warm.mode} [{'OK' if same_key else 'DIFFERENT!'}]")
print(f"  Key confidence: {result_warm.key_confidence:.4f} [{'OK' if same_conf else 'DIFFERENT!'}]")
