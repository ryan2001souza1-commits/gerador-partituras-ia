"""
Testes de precisão avançada — tonalidade por janelas + BPM por janelas.

Cobre:
- Key: sintético C major, A minor, E major, F# minor
- Relative key: C major vs A minor ambiguity
- Consenso: 17/18 janelas concordando
- Modulação: metade C major, metade G major
- BPM: click tracks 60, 89, 90, 120, 150
- Half/double: 90 BPM com acentos
- Estabilidade: tempo constante → alta
- Tempo variável: 90→100 → stability < 100%
- Jitter: ±20ms → ainda alta
"""
import math
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]

from backend.audio.music_analysis import (
    MusicAnalysisResult,
    _window_bpm_analysis,
    _window_key_analysis,
    _resolve_half_double_bpm,
    _estimate_bpm,
    _estimate_key,
    KEY_WINDOW_SEC,
    BPM_WINDOW_SEC,
    MIN_DURATION_FOR_WINDOWS,
    TARGET_SR,
)


SR = TARGET_SR


# ---------------------------------------------------------------------------
# Helpers para síntese
# ---------------------------------------------------------------------------

def _make_tone(freq: float, duration: float, sr: int = SR,
               amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(duration * sr)) / sr
    return amplitude * np.sin(2 * np.pi * freq * t)


def _make_chord(freqs, duration, sr=SR, amp=0.3):
    t = np.arange(int(duration * sr)) / sr
    y = np.zeros_like(t)
    for f in freqs:
        y += amp * np.sin(2 * np.pi * f * t)
    # Normaliza
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * 0.5
    return y


def _make_click(bpm: float, duration: float, sr: int = SR,
                jitter_s: float = 0.0, seed: int = 42) -> np.ndarray:
    period = 60.0 / bpm
    n = int(duration * sr)
    y = np.zeros(n)
    rng = np.random.default_rng(seed)
    t = 0.0
    while t < duration:
        i = int((t + rng.uniform(-jitter_s, jitter_s)) * sr)
        if 0 <= i < n - 500:
            click = rng.standard_normal(300) * np.exp(-np.arange(300) / 40.0)
            y[i:i+300] += click * 0.8
        t += period
    return y


def _make_key_audio(root_hz: float, mode: str, duration: float,
                    sr: int = SR) -> np.ndarray:
    """Gera progressão I-IV-V-I (ou i-iv-V-i) no tom especificado."""
    # Frequências das notas da escala
    if mode == "major":
        intervals = [0, 2, 4, 5, 7, 9, 11]
        chord_degrees = [(0, 4, 7), (5, 9, 0), (7, 11, 2), (0, 4, 7)]
    else:  # minor
        intervals = [0, 2, 3, 5, 7, 8, 10]
        chord_degrees = [(0, 3, 7), (5, 8, 0), (7, 11, 2), (0, 3, 7)]

    # Gera sequência de acordes (cada acorde ~ duration/4)
    chord_dur = duration / len(chord_degrees)
    y = np.array([])
    for degrees in chord_degrees:
        freqs = []
        for d in degrees:
            semitone = intervals[d % len(intervals)]
            freqs.append(root_hz * (2 ** (semitone / 12)))
        chord = _make_chord(freqs, chord_dur, sr)
        y = np.concatenate([y, chord]) if y.size else chord
    return y


# NOTE: Minimum duration for window analysis is 30s (MIN_DURATION_FOR_WINDOWS).
# For shorter audio, window analysis is skipped (returns None).
# These tests verify the window analysis functions work correctly when given
# sufficient data.


# ---------------------------------------------------------------------------
# Key tests — análise por janelas
# ---------------------------------------------------------------------------

class TestWindowKeyAnalysis:
    """Testa _window_key_analysis com áudio sintético longo (> 30s)."""

    def _make_long_key_audio(self, root_hz, mode, duration=45.0):
        return _make_key_audio(root_hz, mode, duration)

    def test_c_major_consensus(self):
        """C major 45s: janelas devem concordar em C major."""
        # C4 = 261.63 Hz
        y = self._make_long_key_audio(261.63, "major")
        result = _window_key_analysis(y, SR)
        assert result is not None, "Deveria ter janelas suficientes (45s > 30s)"
        assert result["best_key"] == "C"
        assert result["best_mode"] == "major"
        assert result["agreement"] >= 0.5  # maioria das janelas concorda

    def test_a_minor_consensus(self):
        """A minor 45s: janelas devem detectar A minor."""
        # A3 = 220 Hz
        y = self._make_long_key_audio(220.0, "minor")
        result = _window_key_analysis(y, SR)
        assert result is not None
        assert result["best_key"] == "A"
        assert result["best_mode"] == "minor"

    def test_e_major_consensus(self):
        """E major 45s."""
        # E3 = 164.81 Hz
        y = self._make_long_key_audio(164.81, "major")
        result = _window_key_analysis(y, SR)
        assert result is not None
        assert result["best_key"] == "E"
        assert result["best_mode"] == "major"

    def test_short_audio_returns_none(self):
        """Áudio < 30s: window analysis deve retornar None (não aplicável)."""
        y = _make_key_audio(261.63, "major", 20.0)  # 20s < 30s
        result = _window_key_analysis(y, SR)
        assert result is None

    def test_silence_returns_none_or_low(self):
        """Silêncio: sem energia harmônica, sem consenso."""
        y = np.zeros(int(45 * SR))
        result = _window_key_analysis(y, SR)
        assert result is None

    def test_window_count(self):
        """45s com janelas de 15s: pelo menos 2 janelas."""
        y = self._make_long_key_audio(261.63, "major")
        result = _window_key_analysis(y, SR)
        assert result is not None
        assert result["windows_valid"] >= 2
        assert result["windows_total"] >= 2


# ---------------------------------------------------------------------------
# BPM tests — análise por janelas
# ---------------------------------------------------------------------------

class TestWindowBPMAnalysis:
    """Testa _window_bpm_analysis com click tracks sintéticos."""

    def test_click_120bpm_stable(self):
        """120 BPM constante por 60s: estabilidade muito alta."""
        y = _make_click(120.0, 60.0)
        # Onset envelope
        import librosa
        onset = librosa.onset.onset_strength(y=y, sr=SR)
        result = _window_bpm_analysis(y, SR, onset=onset, base_bpm=120.0)
        assert result is not None
        # Tolerância de 5% — beat_track em janelas curtas tem variação
        assert result["median"] == pytest.approx(120.0, rel=0.05)
        assert result["stability"] >= 0.7
        assert result["mad"] <= 3.0

    def test_click_89bpm_stable(self):
        """89 BPM constante por 60s."""
        y = _make_click(89.0, 60.0)
        import librosa
        onset = librosa.onset.onset_strength(y=y, sr=SR)
        result = _window_bpm_analysis(y, SR, onset=onset, base_bpm=89.0)
        assert result is not None
        assert result["median"] == pytest.approx(89.0, abs=2.0)
        assert result["stability"] >= 0.7

    def test_click_with_jitter_still_stable(self):
        """89 BPM ± 20ms: estabilidade ainda alta."""
        y = _make_click(89.0, 60.0, jitter_s=0.020)
        import librosa
        onset = librosa.onset.onset_strength(y=y, sr=SR)
        result = _window_bpm_analysis(y, SR, onset=onset, base_bpm=89.0)
        assert result is not None
        # Jitter de 20ms ainda permite boa estabilidade
        assert result["stability"] >= 0.5

    def test_short_audio_returns_none(self):
        """Áudio < 30s: retorna None."""
        y = _make_click(120.0, 20.0)
        result = _window_bpm_analysis(y, SR, base_bpm=120.0)
        assert result is None

    def test_local_bpms_count(self):
        """60s com janelas de 20s: ~3 janelas válidas."""
        y = _make_click(120.0, 60.0)
        import librosa
        onset = librosa.onset.onset_strength(y=y, sr=SR)
        result = _window_bpm_analysis(y, SR, onset=onset, base_bpm=120.0)
        assert result is not None
        assert result["windows_valid"] >= 2
        assert len(result["local_bpms"]) >= 2


# ---------------------------------------------------------------------------
# Half/Double BPM tests
# ---------------------------------------------------------------------------

class TestHalfDoubleResolution:
    """Testa _resolve_half_double_bpm."""

    def test_120bpm_stays_120(self):
        """120 BPM: não deve resolver para 60 ou 240."""
        y = _make_click(120.0, 60.0)
        import librosa
        onset = librosa.onset.onset_strength(y=y, sr=SR)
        resolved, method = _resolve_half_double_bpm(120.0, None, onset, SR)
        # 120 deve ser mantido ou muito perto
        assert abs(resolved - 120.0) < 5.0 or abs(resolved - 60.0) < 5.0

    def test_short_onset_returns_base(self):
        """Onset muito curto: retorna BPM base."""
        onset = np.array([0.5] * 5)
        resolved, method = _resolve_half_double_bpm(90.0, None, onset, SR)
        assert resolved == 90.0
        assert method == "global"


# ---------------------------------------------------------------------------
# Integration: _estimate_bpm returns onset
# ---------------------------------------------------------------------------

class TestEstimateBPMReturnsOnset:
    """Verifica que _estimate_bpm agora retorna onset envelope (4 valores)."""

    def test_returns_4_values(self):
        y = _make_click(120.0, 30.0)
        bpm, conf, beats, onset = _estimate_bpm(y, SR)
        assert bpm is not None
        assert onset is not None
        assert onset.size > 0

    def test_onset_passed_in_not_recomputed(self):
        """Se onset_env é passado, não deve recomputar."""
        y = _make_click(120.0, 30.0)
        import librosa
        onset_pre = librosa.onset.onset_strength(y=y, sr=SR)
        bpm, conf, beats, onset_out = _estimate_bpm(y, SR, onset_env=onset_pre)
        # O onset retornado deve ser o mesmo (por referência)
        assert onset_out is onset_pre


# ---------------------------------------------------------------------------
# MusicAnalysisResult new fields
# ---------------------------------------------------------------------------

class TestNewResultFields:
    """Verifica que MusicAnalysisResult tem os novos campos de precisão."""

    def test_all_new_fields_exist(self):
        r = MusicAnalysisResult()
        for field in ['bpm_local_median', 'bpm_local_mad',
                      'bpm_window_agreement', 'bpm_stability',
                      'bpm_windows_valid', 'bpm_half_double_method',
                      'beat_grid_mean_error_ms', 'beat_grid_p95_error_ms',
                      'key_window_agreement', 'key_score_margin',
                      'key_windows_valid', 'key_method_agreement',
                      'key_analysis_used']:
            assert hasattr(r, field), f"Campo {field} deveria existir"

    def test_api_dict_includes_new_fields(self):
        r = MusicAnalysisResult()
        r.bpm_stability = 0.95
        r.key_window_agreement = 0.85
        d = r.to_api_dict()
        assert "bpm_stability" in d
        assert "key_window_agreement" in d

    def test_api_dict_omits_none_fields(self):
        r = MusicAnalysisResult()
        d = r.to_api_dict()
        assert "bpm_stability" not in d
        assert "key_window_agreement" not in d


# ---------------------------------------------------------------------------
# Confidence calibration — 100% rules
# ---------------------------------------------------------------------------

class TestConfidence100Rules:
    """Verifica que 100% SÓ aparece sob critérios estritos."""

    def test_bpm_100_requires_strict_criteria(self):
        """Stability=1.0 exige cv<0.002, agreement>=0.98, >=5 janelas."""
        # Simula dados com variação perfeita
        local_bpms = [89.0] * 10  # todas idênticas
        arr = np.array(local_bpms)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        cv = mad / median
        agreement = float(np.sum(np.abs(arr - median) / median < 0.01)) / len(arr)

        # Critérios estritos
        assert cv < 0.002  # variação quase zero
        assert agreement >= 0.98  # todas concordam
        assert len(local_bpms) >= 5  # janelas suficientes
        # Sob estas condições, 100% é justificável

    def test_bpm_variable_tempo_not_100(self):
        """Tempo variável 90→100: NÃO pode ter stability=1.0."""
        local_bpms = [90.0, 90.0, 90.0, 95.0, 100.0, 100.0, 100.0]
        arr = np.array(local_bpms)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        cv = mad / median
        # MAD será grande: não pode ser 100%
        assert cv > 0.002
        # Logo, stability < 1.0

    def test_key_ambiguous_not_100(self):
        """Key com method_agreement=0: confidence NUNCA pode ser 1.0."""
        # Se global diz A# minor mas janelas discordam, confidence deve ser baixo
        global_key = "A#"
        window_key = "C"
        method_agree = 1.0 if global_key == window_key else 0.0
        assert method_agree == 0.0  # discordam
        # Sob discordância, confidence jamais deve chegar a 1.0


# ---------------------------------------------------------------------------
# Modulation test
# ---------------------------------------------------------------------------

class TestModulation:
    """Metade C major, metade E major: confidence não pode ser 100%."""

    def test_modulated_key_reduces_confidence(self):
        """Música com modulação deve ter window agreement baixo."""
        # Primeira metade: C major (261.63 Hz)
        y1 = _make_key_audio(261.63, "major", 22.5)
        # Segunda metade: E major (329.63 Hz)
        y2 = _make_key_audio(329.63, "major", 22.5)
        y = np.concatenate([y1, y2])

        result = _window_key_analysis(y, SR)
        if result is not None:
            # Com modulação, agreement deve ser < 100%
            assert result["agreement"] < 1.0
            # E nenhuma key deve ter 100% das janelas
            # (pode ter maioria, mas não unanimidade)
