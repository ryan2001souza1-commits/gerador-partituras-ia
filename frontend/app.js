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

  const separateSection = document.getElementById("separate-section");
  const btnSeparate = document.getElementById("btn-separate");
  const separateStatus = document.getElementById("separate-status");
  const stemsInfo = document.getElementById("stems-info");
  const stemsWarning = document.getElementById("stems-warning");

  let currentFileId = null;
  let pollingInterval = null;
  let currentJobId = null;

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

  function hideSeparateSection() {
    if (separateSection) separateSection.classList.add("hidden");
    if (separateStatus) {
      separateStatus.textContent = "";
      separateStatus.className = "status";
      separateStatus.style.display = "none";
    }
    if (btnSeparate) {
      btnSeparate.disabled = false;
      btnSeparate.textContent = "Separar instrumentos";
    }
    if (pollingInterval) {
      clearInterval(pollingInterval);
      pollingInterval = null;
    }
    currentJobId = null;
  }

  function showSeparateSection(fileId) {
    currentFileId = fileId;
    if (separateSection) separateSection.classList.remove("hidden");
    // Verifica se stems já existem para exibir imediatamente
    checkExistingStems(fileId);
  }

  function hideStemsInfo() {
    if (stemsInfo) stemsInfo.classList.add("hidden");
    if (stemsWarning) {
      stemsWarning.textContent = "";
      stemsWarning.classList.add("hidden");
    }
    ["vocals","drums","bass","other"].forEach(stem => {
      const el = document.getElementById(`audio-${stem}`);
      if (el) {
        el.pause();
        el.removeAttribute("src");
        el.load();
      }
    });
  }

  function showStems(fileId) {
    if (!stemsInfo) return;
    const stems = ["vocals","drums","bass","other"];
    stems.forEach(stem => {
      const el = document.getElementById(`audio-${stem}`);
      if (el) {
        el.src = `/api/stems/${encodeURIComponent(fileId)}/${stem}?t=${Date.now()}`;
        el.load();
      }
    });
    stemsInfo.classList.remove("hidden");
  }

  function updateSeparateStatus(message, type) {
    if (!separateStatus) return;
    separateStatus.textContent = message;
    separateStatus.className = "status visible " + (type || "info");
    separateStatus.style.display = "block";
  }

  function clearSeparateStatus() {
    if (!separateStatus) return;
    separateStatus.textContent = "";
    separateStatus.className = "status";
    separateStatus.style.display = "none";
  }

  async function checkExistingStems(fileId) {
    try {
      const resp = await fetch(`/api/stems/${encodeURIComponent(fileId)}`);
      const data = await resp.json().catch(() => null);
      if (resp.ok && data && data.available) {
        showStems(fileId);
        updateSeparateStatus("Instrumentos já separados.", "success");
        if (btnSeparate) {
          btnSeparate.textContent = "Instrumentos já separados";
          btnSeparate.disabled = true;
        }
        return true;
      }
    } catch (e) {
      // ignora, apenas não mostra stems
    }
    return false;
  }

  async function pollJob(jobId) {
    try {
      const resp = await fetch(`/api/separate/status/${encodeURIComponent(jobId)}`);
      const data = await resp.json().catch(() => null);
      if (!resp.ok || !data) {
        updateSeparateStatus("Erro ao consultar status da separação.", "error");
        if (btnSeparate) btnSeparate.disabled = false;
        clearInterval(pollingInterval);
        pollingInterval = null;
        return;
      }
      const status = data.status;
      const msg = data.message || status;
      if (status === "queued") {
        updateSeparateStatus(msg || "Preparando separação...", "info");
      } else if (status === "running") {
        // Tenta interpretar progresso honesto
        updateSeparateStatus(msg || "Separando instrumentos... (pode levar vários minutos)", "info");
        if (btnSeparate) btnSeparate.disabled = true;
      } else if (status === "completed") {
        updateSeparateStatus(msg || "Separação concluída.", "success");
        if (btnSeparate) {
          btnSeparate.textContent = "Separação concluída";
          btnSeparate.disabled = true;
        }
        clearInterval(pollingInterval);
        pollingInterval = null;
        if (data.file_id) showStems(data.file_id);
        else if (currentFileId) showStems(currentFileId);
      } else if (status === "failed") {
        let friendly = msg || "Não foi possível separar os instrumentos.";
        if (data.error && data.error.includes("Demucs não está instalado")) {
          friendly = "Demucs não está instalado.";
        } else if (data.error && data.error.includes("Timeout")) {
          friendly = "A separação demorou mais que o esperado.";
        }
        updateSeparateStatus(friendly, "error");
        if (btnSeparate) btnSeparate.disabled = false;
        clearInterval(pollingInterval);
        pollingInterval = null;
      }
    } catch (e) {
      updateSeparateStatus("Erro ao consultar status.", "error");
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
    hideSeparateSection();
    hideStemsInfo();
  });

  // Botão Separar instrumentos
  if (btnSeparate) {
    btnSeparate.addEventListener("click", async () => {
      if (!currentFileId) {
        updateSeparateStatus("Nenhum arquivo analisado.", "error");
        return;
      }
      btnSeparate.disabled = true;
      const originalText = btnSeparate.textContent;
      btnSeparate.textContent = "Iniciando separação...";
      updateSeparateStatus("Preparando separação...", "info");
      try {
        const resp = await fetch(`/api/separate/${encodeURIComponent(currentFileId)}`, { method: "POST" });
        const data = await resp.json().catch(() => null);
        if (resp.status === 409) {
          updateSeparateStatus("Já existe uma separação em andamento.", "error");
          btnSeparate.disabled = false;
          btnSeparate.textContent = originalText;
          return;
        }
        if (resp.status === 503) {
          updateSeparateStatus("Demucs não está instalado.", "error");
          btnSeparate.disabled = false;
          btnSeparate.textContent = originalText;
          return;
        }
        if (!resp.ok || !data || !data.job_id) {
          const msg = (data && (data.detail || data.message)) || "Não foi possível iniciar a separação.";
          updateSeparateStatus(msg, "error");
          btnSeparate.disabled = false;
          btnSeparate.textContent = originalText;
          return;
        }
        if (data.already_completed) {
          updateSeparateStatus("Instrumentos já separados.", "success");
          showStems(currentFileId);
          btnSeparate.textContent = "Instrumentos já separados";
          btnSeparate.disabled = true;
          return;
        }
        currentJobId = data.job_id;
        updateSeparateStatus(data.message || "Separação agendada. Carregando modelo...", "info");
        // Polling a cada 2s
        if (pollingInterval) clearInterval(pollingInterval);
        pollingInterval = setInterval(() => pollJob(currentJobId), 2000);
        // Primeira checagem rápida após 1s
        setTimeout(() => pollJob(currentJobId), 1000);
      } catch (e) {
        updateSeparateStatus("Erro ao iniciar separação.", "error");
        btnSeparate.disabled = false;
        btnSeparate.textContent = originalText;
      }
    });
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearStatus();
    hideAudioInfo();
    hideMusicInfo();
    hideSeparateSection();
    hideStemsInfo();
    currentFileId = null;

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
          showSeparateSection(fileId);
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
