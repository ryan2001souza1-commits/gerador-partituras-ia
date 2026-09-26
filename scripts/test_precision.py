"""Testa precisão melhorada com arquivo real de 203s."""
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from backend.audio.music_analysis import analyze_music

TEST = BASE / "uploads" / "ea7da280-ca60-4875-8348-ac51143d2e51.mp3"
if not TEST.is_file():
    print("ERRO: arquivo de teste nao encontrado")
    sys.exit(1)

print(f"Arquivo: {TEST.name} (203s)")
print()

t0 = time.perf_counter()
result = analyze_music(TEST)
elapsed = time.perf_counter() - t0

print(f"Tempo: {elapsed:.1f}s")
print()
print("=== RESULTADO GLOBAL ===")
print(f"BPM: {result.bpm_rounded} (raw: {result.bpm:.2f})")
print(f"BPM confidence: {result.bpm_confidence:.4f}")
print(f"Key: {result.key} {result.mode}")
print(f"Key confidence: {result.key_confidence:.4f}")
print(f"First beat: {result.first_beat_time:.4f}s")
print()
print("=== METRICAS BPM AVANCADAS ===")
print(f"Local median: {result.bpm_local_median}")
print(f"Local MAD: {result.bpm_local_mad}")
print(f"Window agreement: {result.bpm_window_agreement}")
print(f"Stability: {result.bpm_stability}")
print(f"Windows valid: {result.bpm_windows_valid}")
print(f"Half/Double method: {result.bpm_half_double_method}")
print(f"Beat grid mean error: {result.beat_grid_mean_error_ms}ms")
print(f"Beat grid p95 error: {result.beat_grid_p95_error_ms}ms")
print()
print("=== METRICAS KEY AVANCADAS ===")
print(f"Window agreement: {result.key_window_agreement}")
print(f"Score margin: {result.key_score_margin}")
print(f"Windows valid: {result.key_windows_valid}")
print(f"Method agreement: {result.key_method_agreement}")
print(f"Analysis used: {result.key_analysis_used}")
print()
print("=== ANTES vs DEPOIS ===")
print(f"BPM:        89 -> {result.bpm_rounded}")
print(f"BPM conf:   0.9386 -> {result.bpm_confidence:.4f}")
print(f"Key:        A# minor -> {result.key} {result.mode}")
print(f"Key conf:   0.3626 -> {result.key_confidence:.4f}")
print(f"Stability:  (nao existia) -> {result.bpm_stability}")

# Tambem imprime API dict completo
print()
print("=== API DICT ===")
import json
print(json.dumps(result.to_api_dict(), indent=2, ensure_ascii=False))
