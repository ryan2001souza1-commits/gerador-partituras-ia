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
# Estimativa de BPM
# ---------------------------------------------------------------------------

def _estimate_bpm(y: np.ndarray, sr: int) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray]]:
    """
    Estima BPM usando:
      - HPSS para extrair componente percussiva (quando benéfico)
      - onset_strength + beat_track

    Retorna (bpm, bpm_confidence, beats_frames)

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
        return None, None, None

    if y is None or y.size == 0:
        return None, 0.0, None

    # Verifica energia/silêncio
    try:
        rms = float(np.sqrt(np.mean(np.square(y))))
        if rms < 1e-4:  # silêncio
            logger.info("Áudio com energia muito baixa (silêncio) — BPM inconclusivo")
            return None, 0.0, None
    except Exception:
        pass

    # Tenta HPSS para percussivo (benéfico para BPM)
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

    # Calcula onset strength
    onset = None
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
        return None, 0.0, None

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

    return bpm, bpm_confidence, beats


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
) -> MusicAnalysisResult:
    """
    Analisa arquivo de áudio original e retorna MusicAnalysisResult.

    - Decodifica via FFmpeg para WAV mono 22050
    - Carrega com librosa
    - Estima BPM + confiança
    - Estima tonalidade + confiança

    Garante limpeza de WAV temporário inclusive em exceção (try/finally).
    Nunca lança exceção para o caller principal (app.py) — retorna Result com error/warning
    quando tecnicamente delimitado. Exceções internas são logadas.

    Duração mínima: se < MIN_DURATION_FOR_MUSIC, retorna warning e confiança baixa,
    mas tenta analisar mesmo assim (não quebra).
    """
    result = MusicAnalysisResult()

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

        # 3. BPM — usa percussivo via HPSS quando útil
        bpm, bpm_conf, beats = _estimate_bpm(y, sr)
        result.bpm = bpm
        result.bpm_rounded = int(round(bpm)) if bpm is not None else None
        result.bpm_confidence = bpm_conf
        if beats is not None:
            try:
                result.beats_count = int(len(beats))
            except Exception:
                result.beats_count = None

        # 4. Tonalidade — usa harmônico via HPSS quando útil
        y_harm = y
        try:
            # Tenta separar harmônico; se falhar, usa y
            y_harm_tmp, _ = librosa.effects.hpss(y)
            if y_harm_tmp is not None and np.size(y_harm_tmp) > 0 and float(np.mean(np.abs(y_harm_tmp))) > 1e-6:
                y_harm = y_harm_tmp
        except Exception as e:
            logger.debug(f"HPSS harmônico falhou, usando sinal original para tonalidade: {e}")
            y_harm = y

        key, mode, key_conf, best_corr = _estimate_key(y_harm, sr)
        result.key = key
        result.mode = mode
        result.key_confidence = key_conf

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

