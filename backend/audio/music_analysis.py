"""
Análise musical básica — BPM, tonalidade, modo e confiança.

Fluxo:
  arquivo original -> FFmpeg -> WAV temporário mono 22050 Hz PCM s16le -> librosa

Responsabilidades:
  - decodificar via FFmpeg de forma robusta (sem uso de shell, nome seguro, limpeza)
  - estimar BPM via HPSS percussivo + onset_strength + beat_track
  - estimar tonalidade via HPSS harmônico + chroma_cqt + comparação com perfis Krumhansl-Kessler
  - normalizar resultados e calcular confianças heurísticas (documentadas)

Mono 22050 Hz:
  - Reduz RAM e CPU (~2x vs 44100 estéreo) sem perda significativa para BPM/tonalidade.
  - Suficiente para cobrir faixa musical relevante (Nyquist 11025 Hz > harmônicos fundamentais
    da maioria dos instrumentos). Cromagramas trabalham tipicamente até C8 (~4186 Hz).
  - Consistente com padrões de MIR (librosa default sr=22050).

Segurança:
  - Nenhum uso de shell
  - Arquivo temporário com nome aleatório (tempfile) independente do nome original
  - Remoção garantida via try/finally
  - FFmpeg localizado via PATH + locais WinGet, argumentos como lista, timeout controlado
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Constantes e configurações
# ---------------------------------------------------------------------------

TARGET_SR = 22050  # Hz — justificada acima
TARGET_CHANNELS = 1
FFMPEG_TIMEOUT = 30  # segundos para decodificação
MIN_DURATION_FOR_MUSIC = 3.0  # segundos — abaixo disso análise é inconclusiva
ANALYSIS_TIMEOUT = 40  # segundos implícito (librosa não tem timeout, mas usamos para documentação)
KEYS_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Perfis Krumhansl-Schmuckler / Krumhansl-Kessler normalizados (origem: estudos perceptuais)
# Major e minor são pesos de cada classe de altura quando a tônica é C / C menor.
KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# ---------------------------------------------------------------------------
# Dataclasses de resultado
# ---------------------------------------------------------------------------

@dataclass
class MusicAnalysisResult:
    bpm: Optional[float] = None
    bpm_rounded: Optional[int] = None
    bpm_confidence: Optional[float] = None  # 0.0-1.0 heurística
    key: Optional[str] = None  # ex: "C", "C#", "E"
    mode: Optional[str] = None  # "major" | "minor"
    key_confidence: Optional[float] = None  # 0.0-1.0 heurística
    warning: Optional[str] = None
    error: Optional[str] = None
    duration_analyzed: Optional[float] = None
    # interno: mantém bruto para debug (não exposto na API final necessariamente)
    beats_count: Optional[int] = None
    # PERFORMANCE (otimização): primeiro beat em segundos, extraído do beat_track
    # interno. Elimina chamada separada a get_beat_grid() que re-decodificava
    # o arquivo e re-executava HPSS (economia: ~30s para áudio de 3min).
    first_beat_time: Optional[float] = None
    # PRECISÃO AVANÇADA — métricas de janelas/consenso
    # BPM
    bpm_local_median: Optional[float] = None
    bpm_local_mad: Optional[float] = None
    bpm_window_agreement: Optional[float] = None
    bpm_stability: Optional[float] = None
    bpm_windows_valid: Optional[int] = None
    bpm_half_double_method: Optional[str] = None
    beat_grid_mean_error_ms: Optional[float] = None
    beat_grid_p95_error_ms: Optional[float] = None
    # Key
    key_window_agreement: Optional[float] = None
    key_score_margin: Optional[float] = None
    key_windows_valid: Optional[int] = None
    key_method_agreement: Optional[float] = None
    key_analysis_used: Optional[str] = None  # "global" | "windows" | "ensemble"

    def to_api_dict(self) -> Dict[str, Any]:
        """Converte para dict compatível com spec da Etapa 3."""
        # bpm pode ser None se inconclusivo
        d: Dict[str, Any] = {}
        d["bpm"] = round(float(self.bpm), 2) if self.bpm is not None else None
        d["bpm_rounded"] = int(self.bpm_rounded) if self.bpm_rounded is not None else None
        d["bpm_confidence"] = round(float(self.bpm_confidence), 3) if self.bpm_confidence is not None else None
        d["key"] = self.key
        d["mode"] = self.mode
        d["key_confidence"] = round(float(self.key_confidence), 3) if self.key_confidence is not None else None
        if self.warning:
            d["warning"] = self.warning
        if self.error:
            d["error"] = self.error
        # PRECISÃO AVANÇADA — métricas de janelas/consenso (transparência real)
        if self.bpm_stability is not None:
            d["bpm_stability"] = round(self.bpm_stability, 3)
        if self.bpm_local_median is not None:
            d["bpm_local_median"] = self.bpm_local_median
        if self.bpm_local_mad is not None:
            d["bpm_local_mad"] = self.bpm_local_mad
        if self.bpm_window_agreement is not None:
            d["bpm_window_agreement"] = round(self.bpm_window_agreement, 3)
        if self.bpm_windows_valid is not None:
            d["bpm_windows_valid"] = self.bpm_windows_valid
        if self.bpm_half_double_method is not None:
            d["bpm_half_double_method"] = self.bpm_half_double_method
        if self.beat_grid_mean_error_ms is not None:
            d["beat_grid_mean_error_ms"] = self.beat_grid_mean_error_ms
        if self.beat_grid_p95_error_ms is not None:
            d["beat_grid_p95_error_ms"] = self.beat_grid_p95_error_ms
        if self.key_window_agreement is not None:
            d["key_window_agreement"] = round(self.key_window_agreement, 3)
        if self.key_score_margin is not None:
            d["key_score_margin"] = round(self.key_score_margin, 4)
        if self.key_windows_valid is not None:
            d["key_windows_valid"] = self.key_windows_valid
        if self.key_method_agreement is not None:
            d["key_method_agreement"] = self.key_method_agreement
        if self.key_analysis_used is not None:
            d["key_analysis_used"] = self.key_analysis_used
        return d


# ---------------------------------------------------------------------------
# Localização de FFmpeg (robusta, sem uso de shell)
# ---------------------------------------------------------------------------

def _find_ffmpeg() -> Optional[str]:
    """
    Localiza ffmpeg de forma robusta:
    1. shutil.which
    2. sibling de ffprobe encontrado (WinGet)
    3. locais comuns Windows
    """
    # 1. PATH
    found = shutil.which("ffmpeg")
    if found:
        return found
    found_exe = shutil.which("ffmpeg.exe")
    if found_exe:
        return found_exe

    # 2. Tenta derivar do ffprobe (mesma pasta bin)
    try:
        ffprobe = shutil.which("ffprobe") or shutil.which("ffprobe.exe")
        if ffprobe:
            ffmpeg_sibling = Path(ffprobe).parent / "ffmpeg.exe"
            if ffmpeg_sibling.is_file():
                return str(ffmpeg_sibling)
            ffmpeg_sibling2 = Path(ffprobe).parent / "ffmpeg"
            if ffmpeg_sibling2.is_file():
                return str(ffmpeg_sibling2)
        # WinGet Gyan.FFmpeg
        winget_base = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
        if winget_base.is_dir():
            for pkg in winget_base.glob("Gyan.FFmpeg*"):
                candidates = list(pkg.rglob("ffmpeg.exe"))
                if candidates:
                    candidates.sort(key=lambda p: len(str(p)))
                    return str(candidates[0])
    except Exception as e:
        logger.debug(f"Erro ao procurar ffmpeg em WinGet: {e}")

    common_paths = [
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\ffmpeg\bin\ffmpeg"),
        Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\tools\ffmpeg\bin\ffmpeg.exe"),
    ]
    for p in common_paths:
        if p.is_file():
            return str(p)
    return None


def get_ffmpeg_version() -> Optional[str]:
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return None
    try:
        result = subprocess.run(
            [ffmpeg, "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            first_line = (result.stdout or "").splitlines()[0] if result.stdout else ""
            return first_line.strip() or None
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Decodificação via FFmpeg para WAV PCM mono 22050
# ---------------------------------------------------------------------------

class MusicAnalysisError(RuntimeError):
    pass


class FFmpegNotFoundError(RuntimeError):
    pass


class FFmpegTimeoutError(RuntimeError):
    pass


def _decode_to_wav(input_path: Path, timeout: int = FFMPEG_TIMEOUT) -> Path:
    """
    Decodifica input_path -> WAV temporário PCM s16le mono 22050 Hz.
    Retorna Path do WAV temporário (caller deve remover via try/finally).
    Lança FFmpegNotFoundError, FFmpegTimeoutError, MusicAnalysisError
    """
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise FFmpegNotFoundError("FFmpeg não encontrado no sistema.")

    if not input_path.is_file():
        raise MusicAnalysisError("Arquivo de entrada não encontrado para decodificação.")

    # Cria arquivo temporário seguro (não depende do nome original)
    # delete=False para que ffmpeg possa escrever; será removido pelo caller
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    # Garante que arquivo será criado do zero (ffmpeg -y sobrescreve)
    # Remove se existia resquício (improvável)
    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass

    # Reconstrói path temporário após unlink (NamedTemporaryFile já deu nome único)
    # Precisamos recriar via mesmo nome; acima unlink removeu, então vamos usar o mesmo nome
    # Na prática, NamedTemporaryFile com delete=False já criou o arquivo vazio; removemos e ffmpeg recria.
    # Para evitar race, usamos o mesmo Path
    # Obs: em Windows, arquivo aberto não pode ser sobrescrito, por isso fechamos antes.

    cmd = [
        ffmpeg,
        "-y",
        "-v", "error",
        "-i", str(input_path),
        "-vn",
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SR),
        "-acodec", "pcm_s16le",
        str(tmp_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        # Limpeza
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        logger.error(f"ffmpeg timeout ao decodificar {input_path.name}: {e}")
        raise FFmpegTimeoutError("Tempo excedido ao decodificar áudio.")

    if result.returncode != 0:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        stderr = (result.stderr or "").strip()[:500]
        logger.warning(f"ffmpeg returncode={result.returncode} ao decodificar {input_path.name} stderr={stderr}")
        raise MusicAnalysisError(f"Falha ao decodificar áudio: {stderr or 'erro desconhecido'}")

    if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise MusicAnalysisError("FFmpeg não gerou WAV válido.")

    return tmp_path


# ---------------------------------------------------------------------------
# PRECISÃO AVANÇADA — Análise por janelas (Etapa de precisão)
# ---------------------------------------------------------------------------

# Window analysis: divide a música em segmentos e analisa cada um
# separadamente, depois consolida por consenso ponderado.
KEY_WINDOW_SEC = 15.0     # janela para tonalidade
BPM_WINDOW_SEC = 20.0     # janela para BPM
MIN_WINDOWS_FOR_CONSENSUS = 3   # mínimo de janelas válidas para usar consenso
MIN_DURATION_FOR_WINDOWS = 30.0  # só usa janelas se áudio >= 30s


def _window_key_analysis(y_harm: np.ndarray, sr: int,
                         chroma: Optional[np.ndarray] = None
                         ) -> Optional[Dict[str, Any]]:
    """Análise de tonalidade por janelas com consenso ponderado por energia.

    Returns:
        Dict com keys: windows_total, windows_valid, agreement, best_key,
        best_mode, margin, energy_weight_mean, per_window (list de dicts)
        ou None se insuficiente.
    """
    try:
        import librosa
    except Exception:
        return None

    duration = float(len(y_harm)) / sr
    if duration < MIN_DURATION_FOR_WINDOWS:
        return None  # Áudio curto demais para análise por janelas

    # Reusa chroma se já foi calculado pelo caller
    if chroma is None:
        try:
            chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
        except Exception:
            return None
    if chroma is None or chroma.shape[1] < 2:
        return None

    # Computa frames por janela
    chroma_frame_rate = sr / 512.0  # hop_length=512 padrão do librosa
    frames_per_window = int(KEY_WINDOW_SEC * chroma_frame_rate)
    total_frames = chroma.shape[1]
    n_windows = max(1, total_frames // frames_per_window)

    if n_windows < MIN_WINDOWS_FOR_CONSENSUS:
        return None

    # RMS por janela para peso de energia harmônica
    samples_per_window = int(KEY_WINDOW_SEC * sr)
    per_window: List[Dict[str, Any]] = []

    for w in range(n_windows):
        c_start = w * frames_per_window
        c_end = min(c_start + frames_per_window, total_frames)
        if c_end - c_start < 10:  # janela muito pequena
            continue

        # Energia harmônica da janela (peso)
        s_start = w * samples_per_window
        s_end = min(s_start + samples_per_window, len(y_harm))
        if s_end <= s_start:
            continue
        seg = y_harm[s_start:s_end]
        rms = float(np.sqrt(np.mean(seg ** 2))) if seg.size > 0 else 0.0
        if rms < 1e-4:
            continue  # silêncio: pula janela

        # Chroma médio da janela
        chroma_mean = np.mean(chroma[:, c_start:c_end], axis=1)
        if np.allclose(chroma_mean, 0, atol=1e-6):
            continue
        std = float(np.std(chroma_mean))
        if std < 1e-3:
            continue  # ambíguo demais

        # Correlaciona com 24 perfis
        best_c = -2.0
        second_c = -2.0
        best_k = None
        best_m = None
        for shift in range(12):
            pm = np.roll(KRUMHANSL_MAJOR, shift)
            cm = _pearson_corr(chroma_mean, pm)
            if cm > best_c:
                second_c = best_c
                best_c = cm
                best_k = KEYS_SHARP[shift]
                best_m = "major"
            elif cm > second_c:
                second_c = cm
            pn = np.roll(KRUMHANSL_MINOR, shift)
            cn = _pearson_corr(chroma_mean, pn)
            if cn > best_c:
                second_c = best_c
                best_c = cn
                best_k = KEYS_SHARP[shift]
                best_m = "minor"
            elif cn > second_c:
                second_c = cn

        if best_k is None:
            continue

        margin = best_c - second_c if second_c > -2 else 0.0
        per_window.append({
            "key": best_k,
            "mode": best_m,
            "score": best_c,
            "margin": margin,
            "energy": rms,
        })

    if len(per_window) < MIN_WINDOWS_FOR_CONSENSUS:
        return None

    # Consenso: conta votos ponderados por energia
    votes: Dict[Tuple[str, str], float] = {}
    for pw in per_window:
        k = (pw["key"], pw["mode"])
        votes[k] = votes.get(k, 0.0) + pw["energy"] * (1.0 + pw["margin"])

    # Best por votos ponderados
    best_vote = max(votes.values())
    best_key_mode = max(votes, key=votes.get)
    total_votes = sum(votes.values())
    agreement = best_vote / total_votes if total_votes > 0 else 0.0

    # Conta janelas que concordam com o vencedor
    agreeing = sum(1 for pw in per_window
                   if (pw["key"], pw["mode"]) == best_key_mode)
    agreement_count = agreeing / len(per_window)

    # Margem média das janelas vencedoras
    margins_winner = [pw["margin"] for pw in per_window
                     if (pw["key"], pw["mode"]) == best_key_mode]
    mean_margin = float(np.mean(margins_winner)) if margins_winner else 0.0

    return {
        "best_key": best_key_mode[0],
        "best_mode": best_key_mode[1],
        "agreement": agreement_count,
        "agreement_weighted": agreement,
        "margin": mean_margin,
        "windows_total": n_windows,
        "windows_valid": len(per_window),
        "energy_mean": float(np.mean([p["energy"] for p in per_window])),
        "per_window": per_window,
    }


def _window_bpm_analysis(y: np.ndarray, sr: int,
                         onset: Optional[np.ndarray] = None,
                         base_bpm: Optional[float] = None
                         ) -> Optional[Dict[str, Any]]:
    """Análise de BPM por janelas com mediana robusta e estabilidade real.

    Returns:
        Dict com keys: local_bpms, median, mad, std, agreement, stability,
        windows_total, windows_valid, half_double_resolved
        ou None se insuficiente.
    """
    try:
        import librosa
    except Exception:
        return None

    duration = float(len(y)) / sr
    if duration < MIN_DURATION_FOR_WINDOWS or base_bpm is None:
        return None

    # Reusa onset envelope se disponível
    if onset is None:
        try:
            import librosa
            y_perc_tmp = librosa.effects.hpss(y)[1]
            onset = librosa.onset.onset_strength(y=y_perc_tmp, sr=sr)
        except Exception:
            return None
    if onset is None or onset.size < 10:
        return None

    # Janelas no domínio do onset envelope
    hop = 512
    onset_frame_rate = sr / hop
    frames_per_window = int(BPM_WINDOW_SEC * onset_frame_rate)
    total_frames = onset.size
    n_windows = max(1, total_frames // frames_per_window)

    if n_windows < MIN_WINDOWS_FOR_CONSENSUS:
        return None

    local_bpms: List[float] = []

    for w in range(n_windows):
        start = w * frames_per_window
        end = min(start + frames_per_window, total_frames)
        if end - start < 20:
            continue

        # Energia da janela
        seg_energy = float(np.sqrt(np.mean(onset[start:end] ** 2)))
        if seg_energy < 1e-4:
            continue  # silêncio

        # Beat tracking local
        try:
            local_onset = onset[start:end]
            tempo_local, _ = librosa.beat.beat_track(
                onset_envelope=local_onset, sr=sr)
            if isinstance(tempo_local, np.ndarray):
                if tempo_local.size > 0:
                    tempo_local = float(tempo_local[0])
                else:
                    continue
            else:
                tempo_local = float(tempo_local)
        except Exception:
            continue

        if not np.isfinite(tempo_local) or tempo_local <= 0:
            continue

        # Normaliza half/double para comparar com base_bpm
        # Se local BPM está muito longe do base, tenta /2 ou *2
        ratio = tempo_local / base_bpm
        if ratio > 1.8:
            tempo_local /= 2.0
        elif ratio < 0.55:
            tempo_local *= 2.0

        # Só aceita se estiver razoavelmente perto do base
        if abs(tempo_local - base_bpm) / base_bpm > 0.25:
            continue  # outlier

        local_bpms.append(tempo_local)

    if len(local_bpms) < MIN_WINDOWS_FOR_CONSENSUS:
        return None

    arr = np.array(local_bpms)
    median_bpm = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median_bpm)))
    std = float(np.std(arr))

    # Coeficiente de variação robusto (baseado em MAD)
    cv = mad / median_bpm if median_bpm > 0 else float('inf')

    # Agreement: fração de janelas dentro de 1% do valor mediano
    within_1pct = float(np.sum(np.abs(arr - median_bpm) / median_bpm < 0.01)) / len(arr)

    # Stability: baseada em MAD e agreement
    # conf = clip(1 - cv * K, 0, 1) * agreement_boost
    # K=20: MAD de 0.5% do BPM → cv=0.005 → 1-0.1=0.9
    # K=50: mais sensível
    stability = 1.0 - min(1.0, cv * 50.0)
    stability *= (0.5 + 0.5 * within_1pct)  # boost por agreement
    stability = float(np.clip(stability, 0.0, 1.0))

    # 100% somente se critérios estritos
    # cv < 0.002 (MAD < 0.2% do BPM), agreement >= 0.98, >= 5 janelas
    if cv < 0.002 and within_1pct >= 0.98 and len(local_bpms) >= 5:
        stability = 1.0

    return {
        "local_bpms": [round(b, 2) for b in local_bpms],
        "median": round(median_bpm, 2),
        "mad": round(mad, 3),
        "std": round(std, 3),
        "cv": round(cv, 5),
        "agreement": round(within_1pct, 3),
        "stability": stability,
        "windows_total": n_windows,
        "windows_valid": len(local_bpms),
    }


def _resolve_half_double_bpm(base_bpm: float, local_analysis: Optional[Dict],
                             onset: np.ndarray, sr: int) -> Tuple[float, str]:
    """Resolve ambiguidade half/double tempo usando periodicidade de acentos.

    Compara BPM vs BPM/2 vs BPM*2 contra padrão de acentos no onset envelope.
    Retorna (bpm_resolvido, metodo_escolhido).
    """
    if local_analysis is None or onset is None or onset.size < 20:
        return base_bpm, "global"

    try:
        import librosa
        candidates = [base_bpm, base_bpm / 2.0, base_bpm * 2.0]
        candidates = [c for c in candidates if 30 <= c <= 300]
        if len(candidates) <= 1:
            return base_bpm, "global"

        # Para cada candidato, mede regularidade do beat grid teórico
        best_score = -1.0
        best_bpm = base_bpm
        for bpm_c in candidates:
            beat_period = 60.0 / bpm_c
            # Converte para frames do onset envelope
            beat_frames = beat_period * sr / 512.0
            if beat_frames < 2 or beat_frames > onset.size:
                continue
            # Mede energia em posições de beat
            n_beats = int(onset.size / beat_frames)
            if n_beats < 4:
                continue
            beat_positions = np.arange(n_beats) * beat_frames
            beat_energies = []
            for pos in beat_positions:
                idx = int(pos)
                window = onset[max(0, idx-2):idx+3]
                if window.size > 0:
                    beat_energies.append(float(np.max(window)))
            if not beat_energies:
                continue
            # Score: média de energia nos beats vs fora dos beats
            beat_mean = float(np.mean(beat_energies))
            # Amostra posições fora dos beats
            non_beat_energies = []
            for i in range(0, onset.size, max(1, int(beat_frames))):
                if all(abs(i - pos) > 3 for pos in beat_positions):
                    non_beat_energies.append(float(onset[i]))
            non_beat_mean = float(np.mean(non_beat_energies)) if non_beat_energies else 0.0
            # Score alto se beats têm mais energia que não-beats
            score = beat_mean / (beat_mean + non_beat_mean + 1e-9)
            if score > best_score:
                best_score = score
                best_bpm = bpm_c

        method = "global"
        if abs(best_bpm - base_bpm) > 0.5:
            if best_bpm < base_bpm:
                method = "half_resolved"
            else:
                method = "double_resolved"
        return best_bpm, method
    except Exception:
        return base_bpm, "global"


# ---------------------------------------------------------------------------
# Estimativa de BPM
# ---------------------------------------------------------------------------

def _estimate_bpm(y: np.ndarray, sr: int,
                   y_perc: Optional[np.ndarray] = None,
                   onset_env: Optional[np.ndarray] = None
                   ) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estima BPM usando:
      - HPSS para extrair componente percussiva (quando benéfico)
      - onset_strength + beat_track

    PERFORMANCE: aceita y_perc e onset_env pré-computados para evitar
    HPSS e onset_strength redundantes. Se não fornecidos, computa internamente
    (comportamento original preservado para chamadores externos).

    Retorna (bpm, bpm_confidence, beats_frames, onset_envelope)
    Nota: onset_envelope é retornado para reuso pela análise de janelas.

    bpm_confidence — heurística documentada:
      - baseada na regularidade dos intervalos entre beats (coeficiente de variação)
      - conf = clip(1 - coeff*1.2, 0, 1) com ajuste por número de beats
      - beats <4 => confiança baixa (0.2)
      - beats <8 => penaliza 40%
      - se não há beats ou bpm inconclusivo => 0.0
      - NÃO é certeza científica; é estimativa heurística de estabilidade rítmica
      - valor bruto de bpm é preservado (sem forçar normalização half/double);
        half/double-time não é corrigido automaticamente nesta etapa para evitar
        falsos positivos. O caller pode decidir exibir bpm_raw.

    Nunca inventa BPM quando inconclusivo: retorna None.
    """
    try:
        import librosa
    except Exception as e:
        logger.error(f"librosa não disponível para BPM: {e}")
        return None, None, None, None

    if y is None or y.size == 0:
        return None, 0.0, None, None

    # Verifica energia/silêncio
    try:
        rms = float(np.sqrt(np.mean(np.square(y))))
        if rms < 1e-4:  # silêncio
            logger.info("Áudio com energia muito baixa (silêncio) — BPM inconclusivo")
            return None, 0.0, None, None
    except Exception:
        pass

    # Tenta HPSS para percussivo (benéfico para BPM)
    # PERFORMANCE: se y_perc já foi computado pelo caller (analyze_music),
    # reutiliza — evita HPSS redundante (~15s para áudio de 3min)
    if y_perc is not None and np.size(y_perc) > 0:
        pass  # Usa o pré-computado
    else:
        y_perc = y
        try:
            # HPSS pode falhar em sinais muito curtos; fallback para y
            y_harm, y_perc_tmp = librosa.effects.hpss(y)
            # Se percussivo tem energia razoável, usa; senão mantém original
            if y_perc_tmp is not None and np.size(y_perc_tmp) > 0:
                # Verifica se não é tudo zero
                if float(np.mean(np.abs(y_perc_tmp))) > 1e-6:
                    y_perc = y_perc_tmp
        except Exception as e:
            logger.debug(f"HPSS percussivo falhou, usando sinal original para BPM: {e}")
            y_perc = y

    # Calcula onset strength — PERFORMANCE: reusa onset_env se fornecido
    onset = None
    if onset_env is not None and np.size(onset_env) > 0:
        onset = onset_env
    else:
        try:
            onset = librosa.onset.onset_strength(y=y_perc, sr=sr)
        except Exception as e:
            logger.debug(f"onset_strength falhou: {e}")
            onset = None

    # Beat tracking
    tempo = None
    beats = None
    try:
        if onset is not None:
            tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr)
        else:
            tempo, beats = librosa.beat.beat_track(y=y_perc, sr=sr)
    except Exception as e:
        logger.warning(f"beat_track falhou: {e}")
        return None, 0.0, None, None

    # Normaliza tempo (librosa 0.11 retorna array)
    try:
        if isinstance(tempo, np.ndarray):
            if tempo.size == 0:
                tempo_val = None
            elif tempo.size == 1:
                tempo_val = float(tempo[0])
            else:
                # Pode retornar múltiplos candidatos; pega o primeiro (mais provável)
                # Alternativa seria mediana, mas mantemos primeiro para preservar bruto
                tempo_val = float(tempo[0])
        else:
            tempo_val = float(tempo) if tempo is not None else None
    except Exception:
        tempo_val = None

    if tempo_val is None or not np.isfinite(tempo_val) or tempo_val <= 0:
        logger.info(f"BPM inconclusivo (tempo_val={tempo_val})")
        return None, 0.0, beats

    # Sanitiza: BPM fora de faixa musical plausível pode ser mantido, mas loga
    # Não forçamos half/double-time automaticamente (documentado).
    # Se desejado no futuro, preservar raw e expor alternatico.
    bpm = float(tempo_val)

    # Confiança baseada em regularidade dos beats
    bpm_confidence: Optional[float] = None
    try:
        if beats is None or len(beats) < 4:
            # Poucos beats => instável
            bpm_confidence = 0.2 if bpm is not None else 0.0
        else:
            # Converte frames de beat para tempo em segundos
            beat_times = librosa.frames_to_time(beats, sr=sr)
            intervals = np.diff(beat_times)
            # Filtra intervalos inválidos
            intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
            if intervals.size < 2:
                bpm_confidence = 0.3
            else:
                median = float(np.median(intervals))
                std = float(np.std(intervals))
                if median <= 1e-6:
                    bpm_confidence = 0.0
                else:
                    coeff = std / median  # coeficiente de variação
                    # Heurística: confiança inversamente proporcional à variação
                    # conf = 1 - coeff*1.2, limitado [0,1]
                    conf = 1.0 - coeff * 1.2
                    conf = float(np.clip(conf, 0.0, 1.0))
                    # Penaliza poucos beats (<8)
                    if len(beats) < 8:
                        conf *= 0.6
                    # Se onset médio é muito baixo, reduz confiança (sinal pouco percussivo)
                    # Não temos onset por beat, mas podemos estimar via std
                    bpm_confidence = float(np.clip(conf, 0.0, 1.0))
            # Ajuste final: se BPM extremo (<40 ou >220) confiança reduz 20% (pode ser half/double artefato)
            if bpm < 40 or bpm > 220:
                if bpm_confidence is not None:
                    bpm_confidence = float(np.clip(bpm_confidence * 0.8, 0.0, 1.0))
    except Exception as e:
        logger.debug(f"Falha ao calcular bpm_confidence: {e}")
        bpm_confidence = 0.5 if bpm is not None else 0.0

    # Clamp final
    if bpm_confidence is not None:
        bpm_confidence = float(np.clip(bpm_confidence, 0.0, 1.0))

    return bpm, bpm_confidence, beats, onset


# ---------------------------------------------------------------------------
# Estimativa de tonalidade (Krumhansl)
# ---------------------------------------------------------------------------

def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Correlação de Pearson segura (retorna -1..1, 0 se inválido)."""
    try:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        # Remove média
        a_mean = a - np.mean(a)
        b_mean = b - np.mean(b)
        denom = float(np.linalg.norm(a_mean) * np.linalg.norm(b_mean))
        if denom < 1e-8:
            return 0.0
        corr = float(np.dot(a_mean, b_mean) / denom)
        # Clip por segurança numérica
        corr = float(np.clip(corr, -1.0, 1.0))
        if not np.isfinite(corr):
            return 0.0
        return corr
    except Exception:
        return 0.0


def _estimate_key(y_harm: np.ndarray, sr: int) -> Tuple[Optional[str], Optional[str], Optional[float], Optional[float]]:
    """
    Estima tonalidade comparando cromagrama médio com perfis Krumhansl transpostos.

    Passos:
      1. chroma_cqt (preferencial, melhor para música) ou chroma_stft fallback
      2. média temporal das 12 classes
      3. para cada 12 tônicas x 2 modos (24 perfis), rotaciona perfil e correlaciona (Pearson)
      4. escolhe melhor correlação

    Retorna (key, mode, key_confidence, best_corr)
    key_confidence heurística:
      - compara melhor perfil vs segundo melhor e vs média
      - diff_conf = clip((best - second_best)*2.5, 0,1)
      - abs_conf = (best+1)/2
      - confidence = 0.7*diff_conf + 0.3*abs_conf, clamp 0-1
      - se best <=0 => confiança baixa (0.05-0.15)
      - Essa é medida relativa/heurística, não certeza científica. Para músicas ambíguas
        (modulação, atonal, pouca harmonia, ruído) a confiança será baixa e frontend
        exibirá aviso "Resultado com baixa confiança".

    Não escolhe apenas nota mais frequente.
    """
    try:
        import librosa
    except Exception as e:
        logger.error(f"librosa não disponível para tonalidade: {e}")
        return None, None, None, None

    if y_harm is None or y_harm.size == 0:
        return None, None, 0.0, None

    # Verifica energia harmônica
    try:
        rms = float(np.sqrt(np.mean(np.square(y_harm))))
        if rms < 1e-4:
            logger.info("Componente harmônico muito baixo — tonalidade inconclusiva")
            return None, None, 0.0, None
    except Exception:
        pass

    # Gera cromagrama
    chroma = None
    try:
        # chroma_cqt é preferível para material musical (resolução log-freq)
        chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    except Exception as e:
        logger.debug(f"chroma_cqt falhou, tentando chroma_stft: {e}")
        try:
            chroma = librosa.feature.chroma_stft(y=y_harm, sr=sr)
        except Exception as e2:
            logger.warning(f"Falha ao gerar cromagrama: {e2}")
            return None, None, 0.0, None

    if chroma is None or chroma.size == 0:
        return None, None, 0.0, None

    # Média temporal (12 classes)
    try:
        chroma_mean = np.mean(chroma, axis=1)
        # Normalização simples: divide pelo max ou norma para estabilidade numérica
        # Mas Pearson já normaliza via média, então mantemos raw mean
        # Verifica se cromagrama é plano/silêncio
        if np.allclose(chroma_mean, 0, atol=1e-6):
            return None, None, 0.0, None
        # Se desvio muito baixo (distribuição uniforme => ambíguo)
        std = float(np.std(chroma_mean))
        if std < 1e-3:
            # Muito ambíguo
            return None, None, 0.15, None
    except Exception as e:
        logger.warning(f"Falha ao agregar cromagrama: {e}")
        return None, None, 0.0, None

    # Compara com 24 perfis
    correlations: List[Tuple[float, str, str]] = []  # (corr, key, mode)
    best_corr = -2.0
    best_key = None
    best_mode = None

    for shift in range(12):
        # Major rotacionado
        profile_major_rot = np.roll(KRUMHANSL_MAJOR, shift)
        corr_major = _pearson_corr(chroma_mean, profile_major_rot)
        correlations.append((corr_major, KEYS_SHARP[shift], "major"))
        if corr_major > best_corr:
            best_corr = corr_major
            best_key = KEYS_SHARP[shift]
            best_mode = "major"
        # Minor rotacionado
        profile_minor_rot = np.roll(KRUMHANSL_MINOR, shift)
        corr_minor = _pearson_corr(chroma_mean, profile_minor_rot)
        correlations.append((corr_minor, KEYS_SHARP[shift], "minor"))
        if corr_minor > best_corr:
            best_corr = corr_minor
            best_key = KEYS_SHARP[shift]
            best_mode = "minor"

    if best_key is None or best_mode is None or best_corr is None:
        return None, None, 0.0, None

    # Ordena por correlação decrescente
    correlations.sort(key=lambda x: x[0], reverse=True)
    second_best = correlations[1][0] if len(correlations) > 1 else -1.0

    # Calcula confiança heurística documentada
    try:
        if best_corr <= 0:
            # Correlação negativa ou zero => baixa confiança absoluta
            # Ainda retorna melhor palpite mas com confiança mínima
            key_confidence = 0.08
        else:
            diff = float(best_corr - second_best)  # [0,2]
            diff_conf = float(np.clip(diff * 2.5, 0.0, 1.0))  # diff 0.4 => 1.0
            abs_conf = float((best_corr + 1.0) / 2.0)  # mapeia -1..1 -> 0..1, mas best>0 => 0.5..1
            # Média ponderada: dif é mais informativo que abs
            key_confidence = 0.7 * diff_conf + 0.3 * abs_conf
            # Penaliza se std do cromagrama baixo (distribuição pouco destacada)
            # já verificado acima, mas se best_corr <0.3, reduz mais
            if best_corr < 0.3:
                key_confidence *= 0.6
            key_confidence = float(np.clip(key_confidence, 0.0, 1.0))
    except Exception as e:
        logger.debug(f"Falha ao calcular key_confidence: {e}")
        key_confidence = float(np.clip((best_corr + 1) / 2 * 0.5, 0.0, 1.0)) if best_corr is not None else 0.0

    return best_key, best_mode, key_confidence, float(best_corr)


# ---------------------------------------------------------------------------
# Função pública principal
# ---------------------------------------------------------------------------

def analyze_music(
    input_path: Path,
    duration_probe: Optional[float] = None,
    file_id: Optional[str] = None,
) -> MusicAnalysisResult:
    """
    Analisa arquivo de áudio original e retorna MusicAnalysisResult.

    ETAPA 8.3 — CACHE PERSISTENTE:
    Se file_id for fornecido, verifica cache antes de computar.
    Cache hit: retorna imediatamente (não roda librosa/HPSS/chroma/beat_track).
    Cache miss: computa normalmente e salva resultado no cache.

    - Decodifica via FFmpeg para WAV mono 22050
    - Carrega com librosa
    - Estima BPM + confiança (global + janelas)
    - Estima tonalidade + confiança (global + janelas)

    Garante limpeza de WAV temporário inclusive em exceção (try/finally).
    Nunca lança exceção para o caller principal (app.py) — retorna Result com error/warning
    quando tecnicamente delimitado. Exceções internas são logadas.

    Duração mínima: se < MIN_DURATION_FOR_MUSIC, retorna warning e confiança baixa,
    mas tenta analisar mesmo assim (não quebra).
    """
    result = MusicAnalysisResult()

    # ------------------------------------------------------------------
    # ETAPA 8.3 — CACHE PERSISTENTE: verifica antes de computar
    # ------------------------------------------------------------------
    if file_id:
        try:
            from backend.audio.analysis_cache import (
                get_cached_analysis, save_analysis_cache, result_to_cache_dict,
            )
            from backend.audio.long_audio import compute_audio_hash

            audio_hash = compute_audio_hash(Path(input_path))
            if audio_hash:
                cached = get_cached_analysis(file_id, audio_hash)
                if cached is not None:
                    # CACHE HIT: reconstrói MusicAnalysisResult do dict
                    logger.info(f"Analysis CACHE HIT file_id={file_id}")
                    result.bpm = cached.get("bpm")
                    result.bpm_rounded = cached.get("bpm_rounded")
                    result.bpm_confidence = cached.get("bpm_confidence")
                    result.key = cached.get("key")
                    result.mode = cached.get("mode")
                    result.key_confidence = cached.get("key_confidence")
                    result.duration_analyzed = cached.get("duration")
                    result.first_beat_time = cached.get("first_beat_time")
                    result.beats_count = cached.get("beats_count")
                    result.bpm_stability = cached.get("bpm_stability")
                    result.bpm_local_median = cached.get("bpm_local_median")
                    result.bpm_local_mad = cached.get("bpm_local_mad")
                    result.bpm_window_agreement = cached.get("bpm_window_agreement")
                    result.bpm_windows_valid = cached.get("bpm_windows_valid")
                    result.bpm_half_double_method = cached.get("bpm_half_double_method")
                    result.beat_grid_mean_error_ms = cached.get("beat_grid_mean_error_ms")
                    result.beat_grid_p95_error_ms = cached.get("beat_grid_p95_error_ms")
                    result.key_window_agreement = cached.get("key_window_agreement")
                    result.key_score_margin = cached.get("key_score_margin")
                    result.key_windows_valid = cached.get("key_windows_valid")
                    result.key_method_agreement = cached.get("key_method_agreement")
                    result.key_analysis_used = cached.get("key_analysis_used")
                    if cached.get("warning"):
                        result.warning = cached["warning"]
                    return result
        except Exception as e:
            logger.debug(f"Analysis cache check falhou (não crítico): {e}")

    # Valida duração mínima via probe se disponível
    if duration_probe is not None and duration_probe < MIN_DURATION_FOR_MUSIC:
        result.warning = f"Áudio muito curto ({duration_probe:.1f}s). Análise musical pode ser imprecisa. Mínimo recomendado: {MIN_DURATION_FOR_MUSIC:.0f}s."

    wav_path: Optional[Path] = None
    try:
        # 1. Decodificação
        try:
            wav_path = _decode_to_wav(input_path)
        except FFmpegNotFoundError as e:
            logger.error(f"analyze_music ffmpeg ausente: {e}")
            result.error = "FFmpeg não disponível para análise musical."
            result.warning = result.error
            return result
        except FFmpegTimeoutError as e:
            logger.error(f"analyze_music ffmpeg timeout: {e}")
            result.error = "Tempo excedido ao decodificar áudio para análise musical."
            result.warning = result.error
            return result
        except MusicAnalysisError as e:
            logger.warning(f"analyze_music falha decodificação: {e}")
            result.error = "Não foi possível decodificar o áudio para análise musical."
            result.warning = result.error
            return result
        except Exception as e:
            logger.error(f"analyze_music erro inesperado decodificação: {e}")
            result.error = "Erro ao decodificar áudio."
            result.warning = result.error
            return result

        # 2. Carrega áudio com librosa (já decodificado como mono 22050, então load é rápido)
        try:
            import librosa
            # librosa.load respeita sr do arquivo se sr=None, mas forçamos TARGET_SR para garantir
            # usa mono=True embora wav já seja mono
            y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
        except Exception as e:
            logger.error(f"Falha ao carregar WAV com librosa: {e}")
            result.error = "Não foi possível carregar o áudio para análise."
            result.warning = result.error
            return result

        if y is None or y.size == 0:
            result.error = "Áudio vazio ou ilegível após decodificação."
            result.warning = result.error
            return result

        # Duração analisada
        try:
            duration_analyzed = float(len(y) / float(sr)) if sr else None
            result.duration_analyzed = duration_analyzed
            if duration_analyzed is not None and duration_analyzed < MIN_DURATION_FOR_MUSIC:
                # Se probe não avisou, avisa agora
                if not result.warning:
                    result.warning = f"Áudio muito curto ({duration_analyzed:.1f}s). Resultados com baixa confiança."
                # Não aborta, continua mas confiança será baixa
        except Exception:
            pass

        # Verifica se áudio muito longo: documentamos que analisamos completo em 22050 mono
        # Performance: para arquivos >10 min (~13M samples) ainda é ok em RAM (~50MB)
        # Não cortamos arbitrariamente; se necessário, futuro poderá limitar a 180s com aviso.

        # 3+4. BPM e Tonalidade — PERFORMANCE: UMA chamada HPSS para ambos.
        # Antes: _estimate_bpm fazia HPSS #1, depois analyze_music fazia HPSS #2
        #        para key, e get_beat_grid (separado) fazia HPSS #3.
        #        Total: 3 HPSS = ~45s para áudio de 3min. Agora: 1 HPSS = ~15s.
        # Os componentes y_harm/y_perc são idênticos aos que HPSS produzia
        # separadamente — mesma função, mesmos parâmetros, qualidade igual.
        y_harm = y
        y_perc_ss = y  # fallback: sinal original se HPSS falhar
        hpss_ok = False
        try:
            y_harm_tmp, y_perc_tmp = librosa.effects.hpss(y)
            if y_harm_tmp is not None and np.size(y_harm_tmp) > 0:
                if float(np.mean(np.abs(y_harm_tmp))) > 1e-6:
                    y_harm = y_harm_tmp
                hpss_ok = True
            if y_perc_tmp is not None and np.size(y_perc_tmp) > 0:
                if float(np.mean(np.abs(y_perc_tmp))) > 1e-6:
                    y_perc_ss = y_perc_tmp
        except Exception as e:
            logger.debug(f"HPSS falhou, usando sinal original: {e}")
            y_harm = y
            y_perc_ss = y

        # ------------------------------------------------------------------
        # PRECISÃO AVANÇADA — Onset envelope computado UMA VEZ para todo o pipeline.
        # Reusado por: _estimate_bpm, _window_bpm_analysis, _resolve_half_double.
        # PERFORMANCE: antes era computado 2x (dentro de _estimate_bpm + aqui).
        # ------------------------------------------------------------------
        onset_env_shared: Optional[np.ndarray] = None
        if duration_analyzed is not None and duration_analyzed >= MIN_DURATION_FOR_WINDOWS:
            try:
                onset_env_shared = librosa.onset.onset_strength(y=y_perc_ss, sr=sr)
            except Exception:
                onset_env_shared = None

        # 3. BPM — usa y_perc e onset_env do HPSS compartilhado (não recalcula)
        bpm, bpm_conf, beats, onset_returned = _estimate_bpm(
            y, sr, y_perc=y_perc_ss if hpss_ok else None,
            onset_env=onset_env_shared)
        # Se onset_env não foi computado acima (áudio curto), usa o retornado
        if onset_env_shared is None and onset_returned is not None:
            onset_env_shared = onset_returned
        result.bpm = bpm
        result.bpm_rounded = int(round(bpm)) if bpm is not None else None
        result.bpm_confidence = bpm_conf
        if beats is not None:
            try:
                result.beats_count = int(len(beats))
            except Exception:
                result.beats_count = None
            # PERFORMANCE: extrai first_beat_time do beat_track já computado.
            # Elimina get_beat_grid() separado (que re-decodificava + HPSS #3).
            try:
                if len(beats) > 0:
                    result.first_beat_time = float(
                        librosa.frames_to_time(int(beats[0]), sr=sr))
            except Exception:
                result.first_beat_time = None

        # ------------------------------------------------------------------
        # PRECISÃO AVANÇADA — BPM por janelas (para áudio >= 30s)
        # Onset envelope JÁ computado acima — reusado, não recalculado.
        # ------------------------------------------------------------------
        if bpm is not None and duration_analyzed is not None and duration_analyzed >= MIN_DURATION_FOR_WINDOWS:
            try:
                onset_env = onset_env_shared  # reusa o compartilhado

                # Análise por janelas: BPM local em cada segmento
                bpm_window_data = _window_bpm_analysis(
                    y, sr, onset=onset_env, base_bpm=bpm)

                if bpm_window_data is not None:
                    result.bpm_local_median = bpm_window_data["median"]
                    result.bpm_local_mad = bpm_window_data["mad"]
                    result.bpm_window_agreement = bpm_window_data["agreement"]
                    result.bpm_stability = bpm_window_data["stability"]
                    result.bpm_windows_valid = bpm_window_data["windows_valid"]

                    # Resolve half/double tempo se detectado
                    bpm_resolved, hd_method = _resolve_half_double_bpm(
                        bpm, bpm_window_data, onset_env, sr)
                    if hd_method != "global" and abs(bpm_resolved - bpm) > 1.0:
                        result.bpm_half_double_method = hd_method
                        bpm = bpm_resolved
                        result.bpm = bpm
                        result.bpm_rounded = int(round(bpm))

                    # Se mediana local está muito perto do global, usa mediana (mais precisa)
                    if abs(bpm_window_data["median"] - bpm) / bpm < 0.05:
                        bpm = bpm_window_data["median"]
                        result.bpm = bpm
                        result.bpm_rounded = int(round(bpm))

                    # Beat grid error: mede desvio dos beats reais vs grid teórico
                    if beats is not None and len(beats) >= 4:
                        try:
                            beat_times = librosa.frames_to_time(beats, sr=sr)
                            beat_period = 60.0 / bpm
                            # Grid teórico a partir do primeiro beat
                            if result.first_beat_time is not None:
                                t0 = result.first_beat_time
                                theoretical = np.arange(t0, beat_times[-1] + beat_period, beat_period)
                                # Para cada beat real, distância ao beat teórico mais próximo
                                errors = []
                                for bt in beat_times:
                                    idx = int(round((bt - t0) / beat_period))
                                    if 0 <= idx < len(theoretical):
                                        err = (bt - theoretical[idx]) * 1000.0  # ms
                                        errors.append(abs(err))
                                if errors:
                                    errors_arr = np.array(errors)
                                    result.beat_grid_mean_error_ms = round(float(np.mean(errors_arr)), 1)
                                    result.beat_grid_p95_error_ms = round(
                                        float(np.percentile(errors_arr, 95)), 1)
                        except Exception as e:
                            logger.debug(f"beat_grid_error falhou: {e}")

                    # Atualiza bpm_confidence com estabilidade real das janelas
                    if result.bpm_stability is not None:
                        # Combina confiança global com estabilidade por janelas
                        # Estabilidade real domina quando há dados suficientes
                        if result.bpm_windows_valid and result.bpm_windows_valid >= 3:
                            if result.bpm_stability >= 0.98:
                                # Janelas muito concordantes → alta confiança
                                # 1.0 SOMENTE se: stability=1.0, agreement alto, MAD muito baixo
                                if (result.bpm_stability == 1.0
                                        and result.bpm_window_agreement is not None
                                        and result.bpm_window_agreement >= 0.98
                                        and result.bpm_local_mad is not None
                                        and result.bpm_local_mad < 0.3):
                                    bpm_conf = 1.0
                                else:
                                    bpm_conf = max(bpm_conf or 0, result.bpm_stability * 0.98)
                            else:
                                # Estabilidade menor → usa valor real
                                bpm_conf = min(bpm_conf or 0, result.bpm_stability)
                            result.bpm_confidence = round(float(np.clip(bpm_conf, 0, 1)), 4)
            except Exception as e:
                logger.debug(f"Window BPM analysis falhou (não crítico): {e}")

        # 4. Tonalidade — usa y_harm do HPSS compartilhado (não faz HPSS novamente)
        key, mode, key_conf, best_corr = _estimate_key(y_harm, sr)
        result.key = key
        result.mode = mode
        result.key_confidence = key_conf

        # ------------------------------------------------------------------
        # PRECISÃO AVANÇADA — Key por janelas (para áudio >= 30s)
        # Consenso entre janelas ponderado por energia harmônica.
        # ------------------------------------------------------------------
        if duration_analyzed is not None and duration_analyzed >= MIN_DURATION_FOR_WINDOWS:
            try:
                key_window_data = _window_key_analysis(y_harm, sr)
                if key_window_data is not None and key_window_data["windows_valid"] >= MIN_WINDOWS_FOR_CONSENSUS:
                    result.key_window_agreement = key_window_data["agreement"]
                    result.key_score_margin = round(key_window_data["margin"], 4)
                    result.key_windows_valid = key_window_data["windows_valid"]

                    # Method agreement: janelas concordam com estimativa global?
                    w_key = key_window_data["best_key"]
                    w_mode = key_window_data["best_mode"]
                    method_agree = 1.0 if (w_key == key and w_mode == mode) else 0.0
                    result.key_method_agreement = method_agree

                    if method_agree:
                        # Global e janelas concordam → confidence boost
                        result.key_analysis_used = "ensemble"
                        agreement = key_window_data["agreement"]
                        margin = key_window_data["margin"]

                        # Formula: base_conf (global) + window_consensus boost
                        # window_conf = agreement * quality_of_each_window
                        window_conf = agreement * (0.5 + 0.5 * min(1.0, margin * 8))

                        # 100% somente se TODOS os critérios estritos:
                        # agreement >= 98%, margin >= 0.05, method_agree == 1, energia ok
                        if (agreement >= 0.98 and margin >= 0.05
                                and key_window_data["energy_mean"] > 0.001):
                            key_conf = 1.0
                        else:
                            # Combina global + janelas (weighted)
                            key_conf = min(1.0, (key_conf or 0) * 0.35 + window_conf * 0.65)
                    else:
                        # Global e janelas DISCORDAM → usa janelas se consenso forte
                        result.key_analysis_used = "windows_override"
                        agreement = key_window_data["agreement"]
                        if agreement >= 0.75 and key_window_data["windows_valid"] >= 5:
                            # Janelas têm consenso forte — usa resultado das janelas
                            key = w_key
                            mode = w_mode
                            result.key = key
                            result.mode = mode
                            window_conf = agreement * (0.5 + 0.5 * min(1.0, key_window_data["margin"] * 8))
                            key_conf = min(0.95, window_conf * 0.85)
                        else:
                            # Discordância sem consenso claro → reduz confidence
                            result.key_analysis_used = "ambiguous"
                            key_conf = (key_conf or 0) * 0.4

                    result.key_confidence = round(float(np.clip(key_conf, 0, 1)), 4)
            except Exception as e:
                logger.debug(f"Window key analysis falhou (não crítico): {e}")

        # Ajusta warnings para baixa confiança / ambiguidade
        warnings: List[str] = []
        if result.warning:
            warnings.append(result.warning)

        # Se BPM inconclusivo ou baixa confiança
        if result.bpm is None:
            warnings.append("Não foi possível determinar o BPM deste áudio (sinal pouco rítmico ou muito curto).")
        elif result.bpm_confidence is not None and result.bpm_confidence < 0.4:
            warnings.append("BPM com baixa confiança/estabilidade.")

        if result.key is None or result.key_confidence is None or result.key_confidence < 0.35:
            # Se tonalidade ambígua
            if result.key is not None:
                warnings.append("Resultado de tonalidade com baixa confiança — harmonia ambígua ou modulação.")
            else:
                warnings.append("Não foi possível determinar a tonalidade (pouca harmonia, ruído ou áudio muito curto).")

        # Consolida warnings
        if warnings:
            # Mantém primeiro warning de duração se era o principal, senão junta
            # Evita duplicatas
            uniq = []
            seen = set()
            for w in warnings:
                if w not in seen:
                    uniq.append(w)
                    seen.add(w)
            result.warning = " ".join(uniq) if uniq else None
        else:
            # Se não havia warning e confiança ok, limpa warning inicial de duração (se duração era ok)
            if duration_analyzed is not None and duration_analyzed >= MIN_DURATION_FOR_MUSIC and result.warning and "muito curto" in result.warning:
                result.warning = None

        # Se ambos inconclusivos mas sem erro técnico, não considera erro, apenas warning
        # error permanece None salvo falha técnica acima

        # ------------------------------------------------------------------
        # ETAPA 8.3 — CACHE PERSISTENTE: salva resultado após computar
        # ------------------------------------------------------------------
        if file_id and not result.error:
            try:
                from backend.audio.analysis_cache import (
                    save_analysis_cache, result_to_cache_dict,
                )
                from backend.audio.long_audio import compute_audio_hash

                audio_hash = compute_audio_hash(Path(input_path))
                if audio_hash:
                    cache_dict = result_to_cache_dict(result)
                    saved = save_analysis_cache(file_id, audio_hash, cache_dict)
                    if saved:
                        logger.info(f"Analysis CACHE SAVED file_id={file_id}")
            except Exception as e:
                logger.debug(f"Analysis cache save falhou (não crítico): {e}")

        return result

    except Exception as e:
        logger.error(f"analyze_music erro inesperado: {e}", exc_info=True)
        result.error = "Não foi possível determinar a estrutura musical deste áudio."
        if not result.warning:
            result.warning = result.error
        return result
    finally:
        # Garante limpeza do WAV temporário
        if wav_path is not None:
            try:
                wav_path.unlink(missing_ok=True)
            except Exception as e:
                logger.debug(f"Falha ao remover WAV temporário {wav_path}: {e}")


# ---------------------------------------------------------------------------
# Beat grid interno para Etapa 6 (não exposto ao frontend)
# ---------------------------------------------------------------------------

def get_beat_grid(input_path: Path) -> Dict[str, Any]:
    """Retorna beat grid interno: {"beat_times": [...], "first_beat_time": float|None}.

    Reaproveita o mesmo pipeline da Etapa 3 (FFmpeg -> mono 22050 -> HPSS
    percussivo -> onset -> beat_track). NÃO expõe milhares de valores ao
    frontend; uso interno da Etapa 6 para alinhar t=0 ao primeiro pulso.
    Em falha, retorna {"beat_times": [], "first_beat_time": None} e o
    chamador usa fallback beat_offset=0 com warning.
    """
    empty: Dict[str, Any] = {"beat_times": [], "first_beat_time": None}
    wav_path: Optional[Path] = None
    try:
        try:
            wav_path = _decode_to_wav(input_path)
        except Exception as e:
            logger.debug(f"get_beat_grid decode falhou: {e}")
            return empty
        try:
            import librosa
            y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
        except Exception as e:
            logger.debug(f"get_beat_grid load falhou: {e}")
            return empty
        if y is None or y.size == 0:
            return empty
        try:
            y_harm, y_perc = librosa.effects.hpss(y)
            y_use = y_perc
            if y_use is None or np.size(y_use) == 0 or float(np.mean(np.abs(y_use))) < 1e-6:
                y_use = y
        except Exception:
            y_use = y
        try:
            onset = librosa.onset.onset_strength(y=y_use, sr=sr)
            _, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr)
        except Exception as e:
            logger.debug(f"get_beat_grid beat_track falhou: {e}")
            return empty
        if beats is None or len(beats) == 0:
            return empty
        try:
            beat_times = librosa.frames_to_time(beats, sr=sr)
            beat_list = [float(t) for t in beat_times if float(t) >= 0]
            if not beat_list:
                return empty
            return {"beat_times": beat_list, "first_beat_time": float(beat_list[0])}
        except Exception as e:
            logger.debug(f"get_beat_grid conversão falhou: {e}")
            return empty
    finally:
        if wav_path is not None:
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Refinamento do beat_offset — Etapa 7 (não altera o BPM)
# ---------------------------------------------------------------------------

def estimate_beat_offset(
    first_beat_time: Any,
    bpm: Any,
    beat_times: Optional[List[float]] = None,
    earliest_note_time: Optional[float] = None,
) -> float:
    """Projeta a grade rítmica para trás (back-projection) até próximo de 0.

    Se o beat tracker só encontra pulso confiável tarde (ex. 4.09s), usar
    `first_beat_time` como offset comprime tudo que veio antes no beat 0
    (na Etapa 6 isso gerou um acorde artificial gigante em `other`).
    Como o BPM é conhecido, a fase da grade é periódica:

        beat_offset = first_beat_time mod beat_period,  beat_period = 60/BPM

    com 0 <= offset < beat_period (salvo fallback documentado 0.0).

    Parâmetros:
    - first_beat_time: primeiro beat confiável (s) ou None;
    - bpm: andamento da Etapa 3 (NÃO é alterado aqui);
    - beat_times: lista completa (opcional); se first_beat_time for None,
      usa beat_times[0] como fallback;
    - earliest_note_time: início da primeira nota transcrita (opcional,
      informativo — a fase é determinada pela back-projection; o volume de
      notas anteriores ao offset é medido pelo worker e gera warning se
      exceder o limite documentado).

    Retorna offset em segundos, sempre em [0, beat_period), ou 0.0 em
    fallback (BPM/beat ausente ou inválido).
    """
    try:
        b = float(bpm)
    except (TypeError, ValueError):
        return 0.0
    if not (b > 0) or not bool(np.isfinite(b)):
        return 0.0
    period = 60.0 / b

    fb: Optional[float] = None
    try:
        if first_beat_time is not None:
            fb = float(first_beat_time)
    except (TypeError, ValueError):
        fb = None
    if (fb is None or not (fb > 0)) and beat_times:
        try:
            cands = [float(t) for t in beat_times if float(t) > 0]
            fb = min(cands) if cands else None
        except (TypeError, ValueError):
            fb = None
    if fb is None or not (fb > 0) or not bool(np.isfinite(fb)):
        return 0.0
    # Back-projection: fase equivalente dentro de um período.
    off = fb % period
    # Poeira de ponto flutuante próxima ao período -> 0.
    if period - off < 1e-3:
        off = 0.0
    off = round(max(0.0, min(off, period)), 6)
    if off >= period:
        off = 0.0
    return off

