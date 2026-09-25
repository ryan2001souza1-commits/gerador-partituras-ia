document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("upload-form");
  const fileInput = document.getElementById("file-input");
  const btn = document.getElementById("btn-analisar");
  const statusEl = document.getElementById("status");
  const audioInfo = document.getElementById("audio-info");
  const musicInfo = document.getElementById("music-info");

  const infoDuration = document.getElementById("info-duration");
  const infoFormat = document.getElementById("info-format");
  const infoCodec = document.getElementById("info-codec");
  const infoSample = document.getElementById("info-sample");
  const infoChannels = document.getElementById("info-channels");
  const infoBitrate = document.getElementById("info-bitrate");
  const infoSize = document.getElementById("info-size");

  const infoBpm = document.getElementById("info-bpm");
  const infoKey = document.getElementById("info-key");
  const infoKeyConf = document.getElementById("info-key-conf");
  const infoBpmConf = document.getElementById("info-bpm-conf");
  const musicWarning = document.getElementById("music-warning");

  const ALLOWED_EXTS = [".mp3", ".wav", ".flac", ".m4a", ".ogg"];
  const MAX_SIZE = 100 * 1024 * 1024; // 100 MB

  function showStatus(message, type) {
    statusEl.textContent = message;
    statusEl.className = "status visible " + type;
  }

  function clearStatus() {
    statusEl.textContent = "";
    statusEl.className = "status";
  }

  function hideAudioInfo() {
    audioInfo.classList.add("hidden");
  }

  function hideMusicInfo() {
    if (musicInfo) musicInfo.classList.add("hidden");
    if (musicWarning) {
      musicWarning.textContent = "";
      musicWarning.classList.add("hidden");
    }
  }

  function showAudioInfo(data) {
    infoDuration.textContent = data.duration_formatted || (data.duration != null ? String(data.duration) : "-");
    infoFormat.textContent = data.format ? data.format.toUpperCase() : "-";
    infoCodec.textContent = data.codec ? data.codec.toUpperCase() : "-";
    infoSample.textContent = formatSampleRate(data.sample_rate);
    infoChannels.textContent = formatChannels(data.channels);
    infoBitrate.textContent = formatBitrate(data.bitrate);
    infoSize.textContent = formatSize(data.size_bytes);
    audioInfo.classList.remove("hidden");
  }

  // Mapa de chaves técnicas -> português
  const KEY_PT = {
    "C": "Dó",
    "C#": "Dó#",
    "D": "Ré",
    "D#": "Ré#",
    "E": "Mi",
    "F": "Fá",
    "F#": "Fá#",
    "G": "Sol",
    "G#": "Sol#",
    "A": "Lá",
    "A#": "Lá#",
    "B": "Si",
  };

  function formatKeyPT(key, mode) {
    if (!key || !mode) return "-";
    const pt = KEY_PT[key] || key;
    const modePt = mode === "major" ? "maior" : mode === "minor" ? "menor" : mode;
    return `${pt} ${modePt}`;
  }

  function formatConfidence(val) {
    if (val == null || isNaN(val)) return "-";
    const pct = Math.round(val * 100);
    let level = "baixa";
    let cls = "confidence-low";
    if (val >= 0.75) { level = "alta"; cls = "confidence-high"; }
    else if (val >= 0.45) { level = "média"; cls = "confidence-medium"; }
    return { text: `${pct}% (${level})`, pct, level, cls };
  }

  function showMusicInfo(music, fallbackWarning) {
    if (!musicInfo || !infoBpm || !infoKey) return;
    // Trata caso music seja null ou totalmente inconclusivo
    if (!music) {
      infoBpm.textContent = "-";
      infoKey.textContent = "-";
      infoKeyConf.textContent = "-";
      infoBpmConf.textContent = "-";
      if (musicWarning) {
        musicWarning.textContent = fallbackWarning || "Não foi possível determinar a estrutura musical deste áudio.";
        musicWarning.classList.remove("hidden");
      }
      musicInfo.classList.remove("hidden");
      return;
    }

    const hasBpm = music.bpm != null;
    const hasKey = music.key && music.mode;

    // BPM
    if (hasBpm) {
      const rounded = music.bpm_rounded != null ? music.bpm_rounded : Math.round(music.bpm);
      infoBpm.textContent = String(rounded);
    } else {
      infoBpm.textContent = "-";
    }

    // Tonalidade
    if (hasKey) {
      infoKey.textContent = formatKeyPT(music.key, music.mode);
    } else {
      infoKey.textContent = "-";
    }

    // Confianças com cor por faixa
    const kc = formatConfidence(music.key_confidence);
    const bc = formatConfidence(music.bpm_confidence);

    if (kc.text === "-") {
      infoKeyConf.textContent = "-";
      infoKeyConf.className = "";
    } else {
      infoKeyConf.textContent = kc.text;
      infoKeyConf.className = kc.cls;
    }

    if (bc.text === "-") {
      infoBpmConf.textContent = "-";
      infoBpmConf.className = "";
    } else {
      infoBpmConf.textContent = bc.text;
      infoBpmConf.className = bc.cls;
    }

    // Warning / baixa confiança
    let warningText = music.warning || music.error || fallbackWarning || "";
    // Se confiança baixa, adiciona aviso complementar se não houver
    const lowKey = music.key_confidence != null && music.key_confidence < 0.35;
    const lowBpm = music.bpm_confidence != null && music.bpm_confidence < 0.4;
    if ((lowKey || lowBpm) && !warningText) {
      warningText = "Resultado com baixa confiança — harmonia ou ritmo ambíguo. Revisão musical recomendada.";
    }
    if (!hasBpm && !hasKey && !warningText) {
      warningText = "Não foi possível determinar a estrutura musical deste áudio.";
    }

    if (musicWarning) {
      if (warningText) {
        // Trunca levemente se muito longo mas mantém completo
        musicWarning.textContent = warningText;
        musicWarning.classList.remove("hidden");
      } else {
        musicWarning.textContent = "";
        musicWarning.classList.add("hidden");
      }
    }

    musicInfo.classList.remove("hidden");
  }

  function formatSampleRate(sr) {
    if (sr == null) return "-";
    if (sr >= 1000) {
      const khz = sr / 1000;
      // 44100 -> 44.1, 48000 -> 48
      return (Number.isInteger(khz) ? khz.toFixed(0) : khz.toFixed(1)) + " kHz";
    }
    return sr + " Hz";
  }

  function formatBitrate(br) {
    if (br == null) return "-";
    const kbps = Math.round(br / 1000);
    return kbps + " kbps";
  }

  function formatSize(bytes) {
    if (bytes == null) return "-";
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(2) + " MB";
  }

  function formatChannels(ch) {
    if (ch == null) return "-";
    if (ch === 1) return "Mono";
    if (ch === 2) return "Estéreo";
    return ch + " canais";
  }

  function getExtension(filename) {
    const idx = filename.lastIndexOf(".");
    if (idx === -1) return "";
    return filename.slice(idx).toLowerCase();
  }

  // Reset ao selecionar novo arquivo
  fileInput.addEventListener("change", () => {
    clearStatus();
    hideAudioInfo();
    hideMusicInfo();
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearStatus();
    hideAudioInfo();
    hideMusicInfo();

    const file = fileInput.files[0];

    if (!file) {
      showStatus("Selecione um arquivo antes de analisar.", "error");
      return;
    }

    const ext = getExtension(file.name);
    if (!ALLOWED_EXTS.includes(ext)) {
      showStatus(
        "Formato inválido. Formatos aceitos: .mp3, .wav, .flac, .m4a, .ogg",
        "error"
      );
      return;
    }

    if (file.size === 0) {
      showStatus("O arquivo está vazio.", "error");
      return;
    }

    if (file.size > MAX_SIZE) {
      showStatus(
        "Arquivo excede o tamanho máximo permitido de 100 MB.",
        "error"
      );
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    btn.disabled = true;
    const originalText = btn.textContent;
    btn.textContent = "Enviando...";
    showStatus("Enviando música...", "info");

    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json().catch(() => null);

      if (!response.ok || !data || !data.success) {
        const msg =
          (data && (data.detail || data.message)) ||
          "Erro ao enviar o arquivo. Tente novamente.";
        showStatus(msg, "error");
        return;
      }

      showStatus(data.message || "Arquivo enviado com sucesso.", "success");

      // Inicia análise técnica + musical automaticamente
      const fileId = data.file_id;
      btn.textContent = "Analisando estrutura musical...";
      showStatus("Analisando estrutura musical...", "info");

      try {
        const analyzeResp = await fetch(`/api/analyze/${encodeURIComponent(fileId)}`, {
          method: "GET",
        });
        const analyzeData = await analyzeResp.json().catch(() => null);

        if (analyzeResp.ok && analyzeData && analyzeData.success) {
          showAudioInfo(analyzeData);
          // Mostra análise musical mesmo se parcialmente inconclusiva
          showMusicInfo(analyzeData.music || null, analyzeData.music_warning || null);
          showStatus("Análise concluída.", "success");
        } else {
          const msg =
            (analyzeData && (analyzeData.detail || analyzeData.message)) ||
            "Não foi possível analisar o áudio.";
          showStatus(msg, "error");
        }
      } catch (err) {
        showStatus("Erro ao analisar o áudio. Verifique a conexão.", "error");
      }
    } catch (err) {
      showStatus("Erro de conexão. Verifique se o servidor está ativo.", "error");
    } finally {
      btn.disabled = false;
      btn.textContent = originalText;
    }
  });
});
