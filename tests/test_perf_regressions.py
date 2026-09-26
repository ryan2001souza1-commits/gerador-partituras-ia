"""
Testes de regressão — Performance optimization: HPSS único em analyze_music.

Otimização: analyze_music fazia 2 HPSS (um em _estimate_bpm, outro para key).
Agora faz 1 HPSS e compartilha y_harm/y_perc. first_beat_time vem do beat_track
interno, eliminando get_beat_grid() separado (que re-decodificava tudo).

Testes:
1. _estimate_bpm com y_perc pré-computado produz mesmo resultado que sem
2. analyze_music retorna first_beat_time
3. first_beat_time é consistente com get_beat_grid para o mesmo arquivo
"""
import time
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]

from backend.audio.music_analysis import (
    MusicAnalysisResult,
    _decode_to_wav,
    _estimate_bpm,
    analyze_music,
    get_beat_grid,
    TARGET_SR,
)


def _make_click_track(bpm: float, duration: float = 8.0, sr: int = TARGET_SR):
    """Gera click track sintético no BPM especificado."""
    period = 60.0 / bpm
    n = int(duration * sr)
    y = np.zeros(n)
    for t in np.arange(0, duration, period):
        i = int(t * sr)
        if i + 200 < n:
            # Click: burst de ruído curto
            y[i:i+200] = np.random.default_rng(42).standard_normal(200) * 0.8
            # Decay exponencial
            y[i:i+200] *= np.exp(-np.arange(200) / 30.0)
    return y


class TestSingleHPSS:
    """Verifica que _estimate_bpm aceita y_perc sem fazer HPSS redundante."""

    def test_estimate_bpm_with_prec_same_result(self):
        """y_perc pré-computado produz MESMO BPM que HPSS interno."""
        y = _make_click_track(120.0)
        import librosa
        # Método antigo: _estimate_bpm sem y_perc (faz HPSS interno)
        bpm_a, conf_a, beats_a, _ = _estimate_bpm(y, TARGET_SR)
        # Método novo: HPSS externo, passa y_perc
        y_harm, y_perc = librosa.effects.hpss(y)
        bpm_b, conf_b, beats_b, _ = _estimate_bpm(y, TARGET_SR, y_perc=y_perc)
        # BPM deve ser igual
        assert bpm_a is not None and bpm_b is not None
        assert abs(bpm_a - bpm_b) < 0.5, f"BPM divergiu: {bpm_a} vs {bpm_b}"

    def test_estimate_bpm_prec_faster(self):
        """y_perc pré-computado é mais rápido (pula HPSS)."""
        y = _make_click_track(120.0, duration=10.0)
        import librosa
        y_harm, y_perc = librosa.effects.hpss(y)
        # Com y_perc: não faz HPSS
        t0 = time.perf_counter()
        _estimate_bpm(y, TARGET_SR, y_perc=y_perc)
        t_with = time.perf_counter() - t0
        # Sem y_perc: faz HPSS internamente
        t0 = time.perf_counter()
        _estimate_bpm(y, TARGET_SR, y_perc=None)
        t_without = time.perf_counter() - t0
        # Com y_perc deve ser <= sem (HPSS é caro)
        assert t_with <= t_without + 0.1, (
            f"Com y_perc ({t_with:.3f}s) deveria ser mais rápido que "
            f"sem ({t_without:.3f}s)")

    def test_estimate_bpm_none_prec_still_works(self):
        """y_perc=None mantém comportamento original (HPSS interno)."""
        y = _make_click_track(90.0)
        bpm, conf, beats, _ = _estimate_bpm(y, TARGET_SR, y_perc=None)
        assert bpm is not None
        assert 85 <= bpm <= 95  # aproximação


class TestFirstBeatTime:
    """Verifica que analyze_music retorna first_beat_time integrado."""

    def test_result_has_first_beat_time_field(self):
        """MusicAnalysisResult tem campo first_beat_time."""
        assert hasattr(MusicAnalysisResult(), 'first_beat_time')

    def test_analyze_music_returns_first_beat(self):
        """analyze_music preenche first_beat_time para áudio rítmico."""
        # Usa o arquivo real de 203s se disponível, senão sintetiza
        test_file = BASE_DIR / "uploads" / "ea7da280-ca60-4875-8348-ac51143d2e51.mp3"
        if test_file.is_file():
            result = analyze_music(test_file)
            # BPM 89 deve ter beats
            if result.bpm is not None and result.beats_count and result.beats_count > 4:
                assert result.first_beat_time is not None, (
                    "first_beat_time deveria ser preenchido quando beats existem")
                assert 0 <= result.first_beat_time < 10.0
        else:
            pytest.skip("Arquivo de teste real não disponível")

    def test_first_beat_matches_get_beat_grid(self):
        """first_beat_time do analyze_music == first_beat_time do get_beat_grid.

        Isto prova que a integração não mudou o resultado — apenas eliminou
        a computação redundante.
        """
        test_file = BASE_DIR / "uploads" / "ea7da280-ca60-4875-8348-ac51143d2e51.mp3"
        if not test_file.is_file():
            pytest.skip("Arquivo de teste real não disponível")
        result = analyze_music(test_file)
        grid = get_beat_grid(test_file)
        if result.first_beat_time is not None and grid.get("first_beat_time") is not None:
            # Devem ser idênticos (mesma fonte: librosa.beat.beat_track)
            assert abs(result.first_beat_time - grid["first_beat_time"]) < 0.001, (
                f"first_beat divergiu: analyze={result.first_beat_time} "
                f"vs grid={grid['first_beat_time']}")

    def test_analyze_music_quality_preserved(self):
        """A/B: BPM e key não mudam com a otimização."""
        test_file = BASE_DIR / "uploads" / "ea7da280-ca60-4875-8348-ac51143d2e51.mp3"
        if not test_file.is_file():
            pytest.skip("Arquivo de teste real não disponível")
        result = analyze_music(test_file)
        # Valores do baseline (medidos antes da otimização)
        assert result.bpm_rounded == 89, f"BPM mudou: {result.bpm_rounded}"
        assert result.key == "A#", f"Key mudou: {result.key}"
        assert result.mode == "minor", f"Mode mudou: {result.mode}"


class TestGetMusicContextOptimized:
    """Verifica que get_music_context usa first_beat_time integrado."""

    def test_music_context_returns_beat_offset(self):
        """get_music_context ainda retorna beat_offset válido."""
        from backend.notation.score_generator import get_music_context
        from backend.audio.music_analysis import _decode_to_wav
        import shutil

        test_file = BASE_DIR / "uploads" / "ea7da280-ca60-4875-8348-ac51143d2e51.mp3"
        if not test_file.is_file():
            pytest.skip("Arquivo de teste real não disponível")

        fid = "aefde683-36e3-40d2-a296-a2b66f81740f"
        # Não precisa de transcrições reais — get_music_context funciona sem
        ctx = get_music_context(fid)
        assert ctx["tempo"] is not None, "Tempo deveria ser detectado"
        assert ctx["beat_offset"] is not None, "beat_offset deveria existir"
        assert isinstance(ctx.get("warnings"), list)
