document.addEventListener("DOMContentLoaded", () => {
  // ===== MOBILE NAV TOGGLE (redesign) =====
  const navToggle = document.getElementById("nav-toggle");
  const mainNav = document.getElementById("main-nav");
  if (navToggle && mainNav) {
    navToggle.addEventListener("click", () => {
      mainNav.classList.toggle("open");
    });
    // Fecha menu ao clicar em link
    mainNav.querySelectorAll("a").forEach(a => {
      a.addEventListener("click", () => {
        mainNav.classList.remove("open");
      });
    });
  }

  // ===== CHIP STATE (instrumentos selecionados) =====
  document.querySelectorAll("#arrange-instruments input[type=checkbox]").forEach(cb => {
    const updateChip = () => {
      const chip = cb.closest(".chip");
      if (chip) {
        if (cb.checked) chip.classList.add("checked");
        else chip.classList.remove("checked");
      }
    };
    cb.addEventListener("change", updateChip);
    updateChip(); // estado inicial
  });

  // ===== EMPTY STATE OBSERVER (redesign) =====
  const scoreEmpty = document.getElementById("score-empty-state");
  const scoreInfoEl = document.getElementById("score-info");
  const arrangeInfoEl = document.getElementById("arrange-info");
  if (scoreEmpty && scoreInfoEl && arrangeInfoEl) {
    const updateEmptyState = () => {
      const hasScore = !scoreInfoEl.classList.contains("hidden");
      const hasArrange = !arrangeInfoEl.classList.contains("hidden");
      if (hasScore || hasArrange) {
        scoreEmpty.classList.add("hidden");
      } else {
        scoreEmpty.classList.remove("hidden");
      }
    };
    const mo = new MutationObserver(updateEmptyState);
    mo.observe(scoreInfoEl, { attributes: true, attributeFilter: ["class"] });
    mo.observe(arrangeInfoEl, { attributes: true, attributeFilter: ["class"] });
    updateEmptyState();
  }

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

  const transcribeSection = document.getElementById("transcribe-section");
  const btnTranscribe = document.getElementById("btn-transcribe");
  const transcribeStatus = document.getElementById("transcribe-status");
  const transcriptionInfo = document.getElementById("transcription-info");
  const transcriptionWarning = document.getElementById("transcription-warning");

  let currentFileId = null;
  let pollingInterval = null;
  let currentJobId = null;
  let transcribePollingInterval = null;
  let currentTranscribeJobId = null;

  const ALLOWED_EXTS = [".mp3", ".wav", ".flac", ".m4a", ".ogg"];
  const MAX_SIZE = 100 * 1024 * 1024; // 100 MB

  // BUG FIX #9: cancela todos os pollings ativos ao trocar/enviar arquivo.
  // Antes: polling de job A continuava após enviar arquivo B, causando
  // cross-file contamination (stems de A exibidos enquanto currentFileId = B).
  function cancelAllPolling() {
    if (pollingInterval) { clearInterval(pollingInterval); pollingInterval = null; }
    if (transcribePollingInterval) { clearInterval(transcribePollingInterval); transcribePollingInterval = null; }
    if (drumsPollingInterval) { clearInterval(drumsPollingInterval); drumsPollingInterval = null; }
    if (scorePollingInterval) { clearInterval(scorePollingInterval); scorePollingInterval = null; }
    if (arrangePollingInterval) { clearInterval(arrangePollingInterval); arrangePollingInterval = null; }
    if (typeof pipelinePollingInterval !== 'undefined' && pipelinePollingInterval) { clearInterval(pipelinePollingInterval); pipelinePollingInterval = null; }
    currentJobId = null;
    currentTranscribeJobId = null;
    currentDrumJobId = null;
    currentScoreJobId = null;
    currentArrangeJobId = null;
  }

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
    if (typeof pipelineSection !== 'undefined' && pipelineSection) pipelineSection.classList.add("hidden");
    if (typeof pipelineProgress !== 'undefined' && pipelineProgress) pipelineProgress.classList.add("hidden");
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
    if (pipelineSection) pipelineSection.classList.remove("hidden");
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
    // Após mostrar stems, mostra também as seções de transcrição e bateria
    showTranscribeSection(fileId);
    showDrumsSection(fileId);
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

  // Transcrição helpers
  function hideTranscribeSection() {
    if (transcribeSection) transcribeSection.classList.add("hidden");
    if (transcribeStatus) {
      transcribeStatus.textContent = "";
      transcribeStatus.className = "status";
      transcribeStatus.style.display = "none";
    }
    if (btnTranscribe) {
      btnTranscribe.disabled = false;
      btnTranscribe.textContent = "Transcrever para notas";
    }
    if (transcribePollingInterval) {
      clearInterval(transcribePollingInterval);
      transcribePollingInterval = null;
    }
    currentTranscribeJobId = null;
  }

  function showTranscribeSection(fileId) {
    currentFileId = fileId;
    if (transcribeSection) transcribeSection.classList.remove("hidden");
    checkExistingTranscriptions(fileId);
  }

  // ---- Bateria / percussão (Etapa 8) ----
  const drumsSection = document.getElementById("drums-section");
  const btnDrums = document.getElementById("btn-drums");
  const drumsStatus = document.getElementById("drums-status");
  const drumsSummary = document.getElementById("drums-summary");
  let drumsPollingInterval = null;
  let currentDrumJobId = null;

  function updateDrumsStatus(message, type) {
    if (!drumsStatus) return;
    drumsStatus.textContent = message;
    drumsStatus.className = "status visible " + (type || "info");
    drumsStatus.style.display = "block";
  }

  function hideDrumsSection() {
    if (drumsSection) drumsSection.classList.add("hidden");
    if (drumsSummary) drumsSummary.classList.add("hidden");
    if (drumsStatus) {
      drumsStatus.textContent = "";
      drumsStatus.className = "status";
      drumsStatus.style.display = "none";
    }
    if (btnDrums) {
      btnDrums.disabled = false;
      btnDrums.textContent = "Transcrever bateria";
    }
    if (drumsPollingInterval) {
      clearInterval(drumsPollingInterval);
      drumsPollingInterval = null;
    }
    currentDrumJobId = null;
  }

  function showDrumsSection(fileId) {
    currentFileId = fileId;
    if (drumsSection) drumsSection.classList.remove("hidden");
    checkExistingDrums(fileId);
  }

  function showDrumsSummary(data) {
    const stats = (data && data.stats) || {};
    const set = (id, v) => {
      const el = document.getElementById(id);
      if (el) el.textContent = String(v);
    };
    set("drums-kick", stats.kick_count || 0);
    set("drums-snare", stats.snare_count || 0);
    set("drums-hihat", (stats.closed_hihat_count || 0) + (stats.open_hihat_count || 0));
    set("drums-cymbals", stats.crash_count || 0);
    set("drums-toms", (stats.tom_low_count || 0) + (stats.tom_mid_count || 0) + (stats.tom_high_count || 0));
    if (drumsSummary) drumsSummary.classList.remove("hidden");
    const info = document.getElementById("trans-drums-info");
    if (info && data && data.stats) {
      const n = (data.events || []).length || 0;
      info.textContent = `${n} hits de bateria.`;
    }
    // Habilita "Incluir bateria" no arranjo
    const chk = document.getElementById("arrange-include-drums");
    if (chk) chk.disabled = false;
  }

  async function checkExistingDrums(fileId) {
    try {
      const resp = await fetch(`/api/drums/${encodeURIComponent(fileId)}`);
      const data = await resp.json().catch(() => null);
      if (resp.ok && data && data.available) {
        showDrumsSummary(data);
        updateDrumsStatus("Bateria já transcrita.", "success");
        if (btnDrums) {
          btnDrums.textContent = "Bateria transcrita";
          btnDrums.disabled = true;
        }
        return true;
      }
    } catch (e) {}
    return false;
  }

  async function pollDrumJob(jobId) {
    try {
      const resp = await fetch(`/api/drums/status/${encodeURIComponent(jobId)}`);
      const data = await resp.json().catch(() => null);
      if (!resp.ok || !data) {
        updateDrumsStatus("Erro ao consultar status da bateria.", "error");
        if (btnDrums) btnDrums.disabled = false;
        clearInterval(drumsPollingInterval);
        drumsPollingInterval = null;
        return;
      }
      const status = data.status;
      const msg = data.message || status;
      if (status === "queued" || status === "running") {
        updateDrumsStatus(msg || "Transcrevendo bateria...", "info");
        if (btnDrums) btnDrums.disabled = true;
      } else if (status === "completed") {
        updateDrumsStatus(msg || "Concluído.", "success");
        if (btnDrums) {
          btnDrums.textContent = "Bateria transcrita";
          btnDrums.disabled = true;
        }
        clearInterval(drumsPollingInterval);
        drumsPollingInterval = null;
        const model = data.results || data.drums;
        const fid = data.file_id || currentFileId;
        if (model && model.stats) showDrumsSummary(model);
        else if (fid) checkExistingDrums(fid);
      } else if (status === "failed") {
        updateDrumsStatus(msg || "Não foi possível transcrever a bateria.", "error");
        if (btnDrums) btnDrums.disabled = false;
        clearInterval(drumsPollingInterval);
        drumsPollingInterval = null;
      }
    } catch (e) {
      updateDrumsStatus("Erro ao consultar status da bateria.", "error");
    }
  }

  if (btnDrums) {
    btnDrums.addEventListener("click", async () => {
      if (!currentFileId) {
        updateDrumsStatus("Nenhum arquivo analisado.", "error");
        return;
      }
      btnDrums.disabled = true;
      btnDrums.textContent = "Transcrevendo bateria...";
      updateDrumsStatus("Preparando...", "info");
    try {
      const resp = await fetch(`/api/drums/${encodeURIComponent(currentFileId)}`, { method: "POST" });
      const data = await resp.json().catch(() => null);
      if (resp.status === 409) {
        updateDrumsStatus((data && data.detail) || "Separe os instrumentos antes.", "error");
        btnDrums.disabled = false;
        btnDrums.textContent = "Transcrever bateria";
        return;
      }
      if (!resp.ok || !data || !data.job_id) {
        updateDrumsStatus((data && (data.detail || data.message)) || "Não foi possível iniciar.", "error");
        btnDrums.disabled = false;
        btnDrums.textContent = "Transcrever bateria";
        return;
      }
      if (data.already_completed) {
        updateDrumsStatus("Bateria já transcrita.", "success");
        if (data.drums) showDrumsSummary(data.drums);
        btnDrums.textContent = "Bateria transcrita";
        btnDrums.disabled = true;
        return;
      }
      currentDrumJobId = data.job_id;
      updateDrumsStatus(data.message || "Transcrição agendada.", "info");
      if (drumsPollingInterval) clearInterval(drumsPollingInterval);
      drumsPollingInterval = setInterval(() => pollDrumJob(currentDrumJobId), 2000);
      setTimeout(() => pollDrumJob(currentDrumJobId), 1000);
    } catch (err) {
      updateDrumsStatus("Erro ao transcrever bateria.", "error");
      btnDrums.disabled = false;
      btnDrums.textContent = "Transcrever bateria";
    }
    });
  }

  function hideTranscriptionInfo() {
    if (transcriptionInfo) transcriptionInfo.classList.add("hidden");
    if (transcriptionWarning) {
      transcriptionWarning.textContent = "";
      transcriptionWarning.classList.add("hidden");
    }
    ["vocals","bass","other"].forEach(stem => {
      const infoEl = document.getElementById(`trans-${stem}-info`);
      const midiEl = document.getElementById(`trans-${stem}-midi`);
      const jsonEl = document.getElementById(`trans-${stem}-json`);
      if (infoEl) infoEl.textContent = "-";
      if (midiEl) { midiEl.style.display = "none"; midiEl.removeAttribute("href"); }
      if (jsonEl) { jsonEl.style.display = "none"; jsonEl.removeAttribute("href"); }
    });
  }

  function showTranscriptionInfo(fileId, data) {
    if (!transcriptionInfo) return;
    // data pode vir de /api/transcriptions/{file_id} ou de polling
    const stems = ["vocals","bass","other"];
    stems.forEach(stem => {
      const infoEl = document.getElementById(`trans-${stem}-info`);
      const midiEl = document.getElementById(`trans-${stem}-midi`);
      const jsonEl = document.getElementById(`trans-${stem}-json`);
      // Tenta encontrar info no data
      let stemData = null;
      if (data && data.stems) {
        stemData = data.stems.find(s => s.stem === stem);
      } else if (data && data[stem]) {
        stemData = data[stem];
      }
      const notes = stemData ? stemData.notes_count : null;
      const warning = stemData ? stemData.warning : null;
      if (infoEl) {
        if (notes != null) {
          infoEl.textContent = `${notes} notas` + (warning ? ` (${warning})` : "");
          infoEl.className = notes === 0 ? "trans-info confidence-low" : "trans-info";
        } else {
          infoEl.textContent = "-";
        }
      }
      if (midiEl) {
        if (notes != null) {
          midiEl.href = `/api/midi/${encodeURIComponent(fileId)}/${stem}`;
          midiEl.style.display = "inline-block";
        } else {
          midiEl.style.display = "none";
        }
      }
      if (jsonEl) {
        if (notes != null) {
          jsonEl.href = `/api/transcriptions/${encodeURIComponent(fileId)}/${stem}`;
          jsonEl.textContent = "Ver eventos";
          jsonEl.style.display = "inline-block";
        } else {
          jsonEl.style.display = "none";
        }
      }
    });
    transcriptionInfo.classList.remove("hidden");
  }

  function updateTranscribeStatus(message, type) {
    if (!transcribeStatus) return;
    transcribeStatus.textContent = message;
    transcribeStatus.className = "status visible " + (type || "info");
    transcribeStatus.style.display = "block";
  }

  async function checkExistingTranscriptions(fileId) {
    try {
      const resp = await fetch(`/api/transcriptions/${encodeURIComponent(fileId)}`);
      const data = await resp.json().catch(() => null);
      if (resp.ok && data && data.available) {
        showTranscriptionInfo(fileId, data);
        updateTranscribeStatus("Transcrição já concluída.", "success");
        if (btnTranscribe) {
          btnTranscribe.textContent = "Transcrição já concluída";
          btnTranscribe.disabled = true;
        }
        return true;
      }
    } catch (e) {}
    return false;
  }

  async function pollTranscribeJob(jobId) {
    try {
      const resp = await fetch(`/api/transcribe/status/${encodeURIComponent(jobId)}`);
      const data = await resp.json().catch(() => null);
      if (!resp.ok || !data) {
        updateTranscribeStatus("Erro ao consultar status da transcrição.", "error");
        if (btnTranscribe) btnTranscribe.disabled = false;
        clearInterval(transcribePollingInterval);
        transcribePollingInterval = null;
        return;
      }
      const status = data.status;
      const msg = data.message || status;
      // Etapa 8.2: progresso real (percent, chunks) quando disponível
      let progressMsg = msg;
      if (data.progress && typeof data.progress.progress_percent === "number") {
        const p = data.progress;
        const pct = p.progress_percent;
        let detail = "";
        if (p.total_chunks > 1) {
          detail = ` — trecho ${p.current_chunk} de ${p.total_chunks}`;
        }
        if (p.total_seconds > 0) {
          const done = Math.floor(p.processed_seconds / 60) + ":" +
            String(Math.floor(p.processed_seconds % 60)).padStart(2, "0");
          const total = Math.floor(p.total_seconds / 60) + ":" +
            String(Math.floor(p.total_seconds % 60)).padStart(2, "0");
          detail += ` (${done} / ${total})`;
        }
        progressMsg = `${msg}${detail} — ${pct}%`;
        // Barra de progresso via status element
        if (transcribeStatus) {
          transcribeStatus.textContent = progressMsg;
          const bar = "█".repeat(Math.round(pct / 5)) + "░".repeat(20 - Math.round(pct / 5));
          transcribeStatus.textContent = `${progressMsg}\n${bar}`;
          transcribeStatus.style.whiteSpace = "pre";
        }
      }
      if (status === "queued") {
        updateTranscribeStatus(progressMsg || "Preparando transcrição...", "info");
      } else if (status === "running") {
        updateTranscribeStatus(progressMsg || "Transcrevendo... (pode levar vários minutos)", "info");
        if (btnTranscribe) btnTranscribe.disabled = true;
      } else if (status === "completed") {
        updateTranscribeStatus(msg || "Transcrição concluída.", "success");
        if (btnTranscribe) {
          btnTranscribe.textContent = "Transcrição concluída";
          btnTranscribe.disabled = true;
        }
        clearInterval(transcribePollingInterval);
        transcribePollingInterval = null;
        // Mostra info
        if (data.results) showTranscriptionInfo(data.file_id || currentFileId, data.results);
        else if (data.file_id) {
          // busca info completa
          const infoResp = await fetch(`/api/transcriptions/${encodeURIComponent(data.file_id)}`);
          const infoData = await infoResp.json().catch(() => null);
          if (infoResp.ok && infoData) showTranscriptionInfo(data.file_id, infoData);
        } else if (currentFileId) {
          const infoResp = await fetch(`/api/transcriptions/${encodeURIComponent(currentFileId)}`);
          const infoData = await infoResp.json().catch(() => null);
          if (infoResp.ok && infoData) showTranscriptionInfo(currentFileId, infoData);
        }
      } else if (status === "failed") {
        let friendly = msg || "Não foi possível transcrever.";
        if (data.error && data.error.includes("Basic Pitch não está instalado")) friendly = "Basic Pitch não está instalado.";
        else if (data.error && data.error.includes("Timeout")) friendly = "A transcrição demorou mais que o esperado.";
        else if (data.error && data.error.includes("Separe os instrumentos")) friendly = "Separe os instrumentos antes de transcrever.";
        updateTranscribeStatus(friendly, "error");
        if (btnTranscribe) btnTranscribe.disabled = false;
        clearInterval(transcribePollingInterval);
        transcribePollingInterval = null;
      }
    } catch (e) {
      updateTranscribeStatus("Erro ao consultar status da transcrição.", "error");
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
    // BUG FIX #9: cancela pollings de jobs de arquivo anterior
    cancelAllPolling();
    clearStatus();
    hideAudioInfo();
    hideMusicInfo();
    hideSeparateSection();
    hideStemsInfo();
    hideTranscribeSection();
    hideTranscriptionInfo();
    hideDrumsSection();
    hideScoreSection();
    lastMusic = null;
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

  // Botão Transcrever para notas
  if (btnTranscribe) {
    btnTranscribe.addEventListener("click", async () => {
      if (!currentFileId) {
        updateTranscribeStatus("Nenhum arquivo analisado.", "error");
        return;
      }
      btnTranscribe.disabled = true;
      const originalText = btnTranscribe.textContent;
      btnTranscribe.textContent = "Iniciando transcrição...";
      updateTranscribeStatus("Preparando transcrição...", "info");
      try {
        const resp = await fetch(`/api/transcribe/${encodeURIComponent(currentFileId)}`, { method: "POST" });
        const data = await resp.json().catch(() => null);
        if (resp.status === 409) {
          const msg = (data && data.detail) || "Já existe uma transcrição em andamento.";
          updateTranscribeStatus(msg, "error");
          btnTranscribe.disabled = false;
          btnTranscribe.textContent = originalText;
          return;
        }
        if (resp.status === 503) {
          updateTranscribeStatus("Basic Pitch não está instalado.", "error");
          btnTranscribe.disabled = false;
          btnTranscribe.textContent = originalText;
          return;
        }
        if (!resp.ok || !data || !data.job_id) {
          const msg = (data && (data.detail || data.message)) || "Não foi possível iniciar a transcrição.";
          updateTranscribeStatus(msg, "error");
          btnTranscribe.disabled = false;
          btnTranscribe.textContent = originalText;
          return;
        }
        if (data.already_completed) {
          updateTranscribeStatus("Transcrição já concluída.", "success");
          // Busca info completa
          const infoResp = await fetch(`/api/transcriptions/${encodeURIComponent(currentFileId)}`);
          const infoData = await infoResp.json().catch(() => null);
          if (infoResp.ok && infoData) showTranscriptionInfo(currentFileId, infoData);
          btnTranscribe.textContent = "Transcrição já concluída";
          btnTranscribe.disabled = true;
          return;
        }
        currentTranscribeJobId = data.job_id;
        updateTranscribeStatus(data.message || "Transcrição agendada.", "info");
        if (transcribePollingInterval) clearInterval(transcribePollingInterval);
        transcribePollingInterval = setInterval(() => pollTranscribeJob(currentTranscribeJobId), 2000);
        setTimeout(() => pollTranscribeJob(currentTranscribeJobId), 1000);
      } catch (e) {
        updateTranscribeStatus("Erro ao iniciar transcrição.", "error");
        btnTranscribe.disabled = false;
        btnTranscribe.textContent = originalText;
      }
    });
  }

  // ---- Partitura MusicXML (Etapa 6) ----
  const scoreSection = document.getElementById("score-section");
  const scoreInfo = document.getElementById("score-info");
  const btnScore = document.getElementById("btn-generate-score");
  const scoreStatus = document.getElementById("score-status");
  const scoreTempo = document.getElementById("score-tempo");
  const scoreTimesig = document.getElementById("score-timesig");
  const scoreQuant = document.getElementById("score-quant");
  const scoreKeymode = document.getElementById("score-keymode");
  const scoreCleanup = document.getElementById("score-cleanup");
  const scoreKeyHint = document.getElementById("score-key-hint");
  const scoreWarning = document.getElementById("score-warning");
  let scorePollingInterval = null;
  let currentScoreJobId = null;
  let lastMusic = null;

  function updateScoreStatus(message, type) {
    if (!scoreStatus) return;
    scoreStatus.textContent = message;
    scoreStatus.className = "status visible " + (type || "info");
    scoreStatus.style.display = "block";
  }

  function hideScoreSection() {
    if (scoreSection) scoreSection.classList.add("hidden");
    if (scoreInfo) scoreInfo.classList.add("hidden");
    if (scoreStatus) {
      scoreStatus.textContent = "";
      scoreStatus.className = "status";
      scoreStatus.style.display = "none";
    }
    if (btnScore) {
      btnScore.disabled = false;
      btnScore.textContent = "Gerar partitura MusicXML";
    }
    if (scorePollingInterval) {
      clearInterval(scorePollingInterval);
      scorePollingInterval = null;
    }
    currentScoreJobId = null;
    hideArrangeSection();
  }

  function showScoreSection(fileId, music) {
    currentFileId = fileId;
    if (music) lastMusic = music;
    if (!scoreSection) return;
    // Prefill BPM da Etapa 3 quando válido
    if (scoreTempo && lastMusic && lastMusic.bpm_rounded) {
      scoreTempo.value = String(lastMusic.bpm_rounded);
    } else if (scoreTempo && lastMusic && lastMusic.bpm) {
      scoreTempo.value = String(Math.round(lastMusic.bpm));
    }
    // Aviso de armadura quando confiança baixa
    if (scoreKeyHint) {
      const kc = lastMusic ? lastMusic.key_confidence : null;
      if (kc != null && kc < 0.45) {
        scoreKeyHint.textContent = "Ton. detectada com baixa confiança; recomendamos Sem armadura ou revisão manual.";
        scoreKeyHint.classList.remove("hidden");
      } else {
        scoreKeyHint.textContent = "";
        scoreKeyHint.classList.add("hidden");
      }
    }
    scoreSection.classList.remove("hidden");
    checkExistingScore(fileId);
  }

  function showScoreResult(fileId, model) {
    if (!scoreInfo) return;
    const partsEl = document.getElementById("score-parts");
    const bpmEl = document.getElementById("score-bpm");
    const tsEl = document.getElementById("score-timesig-info");
    const qEl = document.getElementById("score-quant-info");
    const dlEl = document.getElementById("score-download");
    if (partsEl) partsEl.textContent = "Vocais, Baixo, Outros";
    if (bpmEl) bpmEl.textContent = model.tempo != null ? String(model.tempo) : "-";
    if (tsEl) tsEl.textContent = model.time_signature || "-";
    if (qEl) qEl.textContent = model.quantization || "-";
    if (dlEl) {
      dlEl.href = `/api/score/${encodeURIComponent(fileId)}/musicxml`;
      dlEl.style.display = "inline-block";
    }
    if (scoreWarning) {
      const warns = model.warnings || [];
      if (warns.length) {
        scoreWarning.textContent = warns.join(" ");
        scoreWarning.classList.remove("hidden");
      } else {
        scoreWarning.textContent = "";
        scoreWarning.classList.add("hidden");
      }
    }
    scoreInfo.classList.remove("hidden");
    showArrangeSection(fileId);
  }

  async function checkExistingScore(fileId) {
    try {
      const resp = await fetch(`/api/score/${encodeURIComponent(fileId)}`);
      const data = await resp.json().catch(() => null);
      if (resp.ok && data && data.available) {
        showScoreResult(fileId, data);
        updateScoreStatus("Partitura já gerada.", "success");
        if (btnScore) {
          btnScore.textContent = "Regenerar partitura";
          btnScore.disabled = false;
        }
        return true;
      }
    } catch (e) {}
    return false;
  }

  async function pollScoreJob(jobId) {
    try {
      const resp = await fetch(`/api/score/status/${encodeURIComponent(jobId)}`);
      const data = await resp.json().catch(() => null);
      if (!resp.ok || !data) {
        updateScoreStatus("Erro ao consultar status da partitura.", "error");
        if (btnScore) btnScore.disabled = false;
        clearInterval(scorePollingInterval);
        scorePollingInterval = null;
        return;
      }
      const status = data.status;
      const msg = data.message || status;
      if (status === "queued" || status === "running") {
        updateScoreStatus(msg || "Gerando partitura...", "info");
        if (btnScore) btnScore.disabled = true;
      } else if (status === "completed") {
        updateScoreStatus(msg || "Partitura concluída.", "success");
        if (btnScore) {
          btnScore.textContent = "Regenerar partitura";
          btnScore.disabled = false;
        }
        clearInterval(scorePollingInterval);
        scorePollingInterval = null;
        const model = data.results || data.score;
        const fid = data.file_id || currentFileId;
        if (model && fid) showScoreResult(fid, model);
        else if (fid) checkExistingScore(fid);
      } else if (status === "failed") {
        updateScoreStatus(msg || "Não foi possível gerar a partitura.", "error");
        if (btnScore) btnScore.disabled = false;
        clearInterval(scorePollingInterval);
        scorePollingInterval = null;
      }
    } catch (e) {
      updateScoreStatus("Erro ao consultar status da partitura.", "error");
    }
  }

  // ---- Arranjo para sopros (Etapa 7) ----
  const arrangeSection = document.getElementById("arrange-section");
  const arrangeInfo = document.getElementById("arrange-info");
  const btnArrange = document.getElementById("btn-arrange");
  const arrangeStatus = document.getElementById("arrange-status");
  const arrangeMode = document.getElementById("arrange-mode");
  const arrangeCleanup = document.getElementById("arrange-cleanup");
  const arrangeInclude = document.getElementById("arrange-include-originals");
  const arrangeWarning = document.getElementById("arrange-warning");
  let arrangePollingInterval = null;
  let currentArrangeJobId = null;

  function updateArrangeStatus(message, type) {
    if (!arrangeStatus) return;
    arrangeStatus.textContent = message;
    arrangeStatus.className = "status visible " + (type || "info");
    arrangeStatus.style.display = "block";
  }

  function hideArrangeSection() {
    if (arrangeSection) arrangeSection.classList.add("hidden");
    if (arrangeInfo) arrangeInfo.classList.add("hidden");
    if (arrangeStatus) {
      arrangeStatus.textContent = "";
      arrangeStatus.className = "status";
      arrangeStatus.style.display = "none";
    }
    if (btnArrange) {
      btnArrange.disabled = false;
      btnArrange.textContent = "Criar arranjo";
    }
    if (arrangePollingInterval) {
      clearInterval(arrangePollingInterval);
      arrangePollingInterval = null;
    }
    currentArrangeJobId = null;
  }

  function showArrangeSection(fileId) {
    currentFileId = fileId;
    if (!arrangeSection) return;
    arrangeSection.classList.remove("hidden");
    checkExistingArrangement(fileId);
  }

  function showArrangeResult(fileId, model) {
    if (!arrangeInfo) return;
    const grid = document.getElementById("arrange-result-grid");
    const dlEl = document.getElementById("arrange-download");
    if (grid) {
      grid.innerHTML = "";
      (model.instruments || []).forEach((inst) => {
        const div = document.createElement("div");
        div.className = "info-item";
        const label = document.createElement("span");
        label.className = "info-label";
        label.textContent = `${inst.name} — ${inst.role}:`;
        const val = document.createElement("span");
        val.textContent = ` ${inst.notes_count} notas`;
        div.appendChild(label);
        div.appendChild(val);
        grid.appendChild(div);
      });
    }
    if (dlEl) {
      dlEl.href = `/api/arrangement/${encodeURIComponent(fileId)}/musicxml`;
      dlEl.style.display = "inline-block";
    }
    if (arrangeWarning) {
      const warns = model.warnings || [];
      if (warns.length) {
        arrangeWarning.textContent = warns.join(" ");
        arrangeWarning.classList.remove("hidden");
      } else {
        arrangeWarning.textContent = "";
        arrangeWarning.classList.add("hidden");
      }
    }
    arrangeInfo.classList.remove("hidden");
  }

  async function checkExistingArrangement(fileId) {
    try {
      const resp = await fetch(`/api/arrangement/${encodeURIComponent(fileId)}`);
      const data = await resp.json().catch(() => null);
      if (resp.ok && data && data.available) {
        showArrangeResult(fileId, data);
        updateArrangeStatus("Arranjo já gerado.", "success");
        return true;
      }
    } catch (e) {}
    return false;
  }

  async function pollArrangeJob(jobId) {
    try {
      const resp = await fetch(`/api/arrange/status/${encodeURIComponent(jobId)}`);
      const data = await resp.json().catch(() => null);
      if (!resp.ok || !data) {
        updateArrangeStatus("Erro ao consultar status do arranjo.", "error");
        if (btnArrange) btnArrange.disabled = false;
        clearInterval(arrangePollingInterval);
        arrangePollingInterval = null;
        return;
      }
      const status = data.status;
      const msg = data.message || status;
      if (status === "queued" || status === "running") {
        updateArrangeStatus(msg || "Gerando arranjo...", "info");
        if (btnArrange) btnArrange.disabled = true;
      } else if (status === "completed") {
        updateArrangeStatus(msg || "Arranjo concluído.", "success");
        if (btnArrange) {
          btnArrange.textContent = "Recriar arranjo";
          btnArrange.disabled = false;
        }
        clearInterval(arrangePollingInterval);
        arrangePollingInterval = null;
        const model = data.results || data.arrangement;
        const fid = data.file_id || currentFileId;
        if (model && fid) showArrangeResult(fid, model);
        else if (fid) checkExistingArrangement(fid);
      } else if (status === "failed") {
        updateArrangeStatus(msg || "Não foi possível gerar o arranjo.", "error");
        if (btnArrange) btnArrange.disabled = false;
        clearInterval(arrangePollingInterval);
        arrangePollingInterval = null;
      }
    } catch (e) {
      updateArrangeStatus("Erro ao consultar status do arranjo.", "error");
    }
  }

  if (btnArrange) {
    btnArrange.addEventListener("click", async () => {
      if (!currentFileId) {
        updateArrangeStatus("Nenhum arquivo analisado.", "error");
        return;
      }
      const boxes = document.querySelectorAll("#arrange-instruments input[type=checkbox]:checked");
      const selected = Array.from(boxes).map((b) => b.value);
      if (!selected.length) {
        updateArrangeStatus("Selecione ao menos 1 instrumento.", "error");
        return;
      }
      const arrangeStyle = document.getElementById("arrange-style");
      const arrangeDrums = document.getElementById("arrange-include-drums");
      const arrangeDynamics = document.getElementById("arrange-dynamics");
      const payload = {
        instruments: selected,
        mode: arrangeMode ? arrangeMode.value : "automatic",
        include_original_parts: arrangeInclude ? arrangeInclude.checked : true,
        cleanup_profile: arrangeCleanup ? arrangeCleanup.value : "natural",
        arrangement_style: arrangeStyle ? arrangeStyle.value : "automatic",
        include_drums: arrangeDrums ? arrangeDrums.checked : true,
        dynamics: arrangeDynamics ? arrangeDynamics.value : "automatic",
      };
      btnArrange.disabled = true;
      btnArrange.textContent = "Criando arranjo...";
      updateArrangeStatus("Analisando melodia...", "info");
      try {
        const resp = await fetch(`/api/arrange/${encodeURIComponent(currentFileId)}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await resp.json().catch(() => null);
        if (resp.status === 409 || resp.status === 400) {
          updateArrangeStatus((data && data.detail) || "Não foi possível gerar o arranjo.", "error");
          btnArrange.disabled = false;
          btnArrange.textContent = "Criar arranjo";
          return;
        }
        if (resp.status === 503) {
          updateArrangeStatus("music21 não está instalado.", "error");
          btnArrange.disabled = false;
          btnArrange.textContent = "Criar arranjo";
          return;
        }
        if (!resp.ok || !data || !data.job_id) {
          updateArrangeStatus((data && (data.detail || data.message)) || "Não foi possível gerar o arranjo.", "error");
          btnArrange.disabled = false;
          btnArrange.textContent = "Criar arranjo";
          return;
        }
        if (data.already_completed) {
          updateArrangeStatus("Arranjo já gerado.", "success");
          if (data.arrangement) showArrangeResult(currentFileId, data.arrangement);
          btnArrange.textContent = "Recriar arranjo";
          btnArrange.disabled = false;
          return;
        }
        currentArrangeJobId = data.job_id;
        updateArrangeStatus(data.message || "Arranjo agendado.", "info");
        if (arrangePollingInterval) clearInterval(arrangePollingInterval);
        arrangePollingInterval = setInterval(() => pollArrangeJob(currentArrangeJobId), 2000);
        setTimeout(() => pollArrangeJob(currentArrangeJobId), 1000);
      } catch (e) {
        updateArrangeStatus("Erro ao gerar arranjo.", "error");
        btnArrange.disabled = false;
        btnArrange.textContent = "Criar arranjo";
      }
    });
  }

  if (btnScore) {
    btnScore.addEventListener("click", async () => {
      if (!currentFileId) {
        updateScoreStatus("Nenhum arquivo analisado.", "error");
        return;
      }
      const tempoVal = scoreTempo && scoreTempo.value ? Number(scoreTempo.value) : null;
      const payload = {
        tempo: tempoVal,
        time_signature: scoreTimesig ? scoreTimesig.value : "4/4",
        quantization: scoreQuant ? scoreQuant.value : "1/16",
        key_mode: scoreKeymode ? scoreKeymode.value : "auto",
        cleanup_profile: scoreCleanup ? scoreCleanup.value : "natural",
      };
      btnScore.disabled = true;
      btnScore.textContent = "Gerando partitura...";
      updateScoreStatus("Preparando partitura...", "info");
      try {
        const resp = await fetch(`/api/score/${encodeURIComponent(currentFileId)}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await resp.json().catch(() => null);
        if (resp.status === 409 || resp.status === 400) {
          updateScoreStatus((data && data.detail) || "Não foi possível gerar a partitura.", "error");
          btnScore.disabled = false;
          btnScore.textContent = "Gerar partitura MusicXML";
          return;
        }
        if (resp.status === 503) {
          updateScoreStatus("music21 não está instalado.", "error");
          btnScore.disabled = false;
          btnScore.textContent = "Gerar partitura MusicXML";
          return;
        }
        if (!resp.ok || !data || !data.job_id) {
          updateScoreStatus((data && (data.detail || data.message)) || "Não foi possível gerar a partitura.", "error");
          btnScore.disabled = false;
          btnScore.textContent = "Gerar partitura MusicXML";
          return;
        }
        if (data.already_completed) {
          updateScoreStatus("Partitura já gerada.", "success");
          const model = data.score;
          if (model) showScoreResult(currentFileId, model);
          btnScore.textContent = "Regenerar partitura";
          btnScore.disabled = false;
          return;
        }
        currentScoreJobId = data.job_id;
        updateScoreStatus(data.message || "Geração agendada.", "info");
        if (scorePollingInterval) clearInterval(scorePollingInterval);
        scorePollingInterval = setInterval(() => pollScoreJob(currentScoreJobId), 2000);
        setTimeout(() => pollScoreJob(currentScoreJobId), 1000);
      } catch (e) {
        updateScoreStatus("Erro ao gerar partitura.", "error");
        btnScore.disabled = false;
        btnScore.textContent = "Gerar partitura MusicXML";
      }
    });
  }

  // Expõe score section sempre que a transcrição for exibida:
  const _origShowTranscriptionInfo2 = showTranscriptionInfo;
  showTranscriptionInfo = function (fileId, data) {
    _origShowTranscriptionInfo2(fileId, data);
    fetch(`/api/analyze/${encodeURIComponent(fileId)}`)
      .then((r) => r.json().catch(() => null))
      .then((a) => showScoreSection(fileId, a && a.music ? a.music : lastMusic))
      .catch(() => showScoreSection(fileId, lastMusic));
  };

  // Guarda última análise musical para prefill de BPM:
  const _origShowMusicInfo = showMusicInfo;
  showMusicInfo = function (music, fallbackWarning) {
    lastMusic = music || lastMusic;
    _origShowMusicInfo(music, fallbackWarning);
  };

  // Hook: quando transcrição conclui via polling, mostra score section
  const _origPollTranscribe = pollTranscribeJob;
  pollTranscribeJob = async function (jobId) {
    await _origPollTranscribe(jobId);
    try {
      if (!transcribePollingInterval && currentFileId && transcriptionInfo && !transcriptionInfo.classList.contains("hidden")) {
        if (scoreSection && scoreSection.classList.contains("hidden")) {
          fetch(`/api/analyze/${encodeURIComponent(currentFileId)}`)
            .then((r) => r.json().catch(() => null))
            .then((a) => showScoreSection(currentFileId, a && a.music ? a.music : lastMusic))
            .catch(() => showScoreSection(currentFileId, lastMusic));
        }
      }
    } catch (e) {}
  };

  // ---- PIPELINE COMPLETO — Etapa 8.3 ----
  const pipelineSection = document.getElementById("pipeline-section");
  const btnPipeline = document.getElementById("btn-pipeline");
  const pipelineStatus = document.getElementById("pipeline-status");
  const pipelineProgress = document.getElementById("pipeline-progress");
  const pipelineProgressBar = document.getElementById("pipeline-progress-bar");
  const pipelineStagesList = document.getElementById("pipeline-stages-list");
  let pipelinePollingInterval = null;
  let currentPipelineJobId = null;

  function updatePipelineStatus(message, type) {
    if (!pipelineStatus) return;
    pipelineStatus.textContent = message;
    pipelineStatus.className = "status visible " + (type || "info");
    pipelineStatus.style.display = "block";
  }

  function renderPipelineStages(stages, currentLabel) {
    if (!pipelineStagesList) return;
    const icons = { pending: "\u25CB", running: "\u2192", completed: "\u2713", failed: "\u2717", skipped: "\u21BB" };
    let html = "";
    for (const [name, s] of Object.entries(stages || {})) {
      const icon = icons[s.status] || "\u25CB";
      const cachedTag = s.cached ? " (reutilizada)" : "";
      const progressTag = s.status === "running" && s.progress > 0 ? ` \u2014 ${s.progress}%` : "";
      const color = s.status === "completed" ? "var(--success)" :
                   s.status === "running" ? "var(--gold)" :
                   s.status === "failed" ? "var(--danger)" : "var(--text-muted)";
      html += `<div style="color:${color}">${icon} ${s.label}${cachedTag}${progressTag}</div>`;
    }
    if (currentLabel) {
      html += `<div style="color:var(--text-secondary);margin-top:4px;font-size:12px">${currentLabel}</div>`;
    }
    pipelineStagesList.innerHTML = html;
  }

  async function pollPipelineJob(jobId) {
    try {
      const resp = await fetch(`/api/pipeline/status/${encodeURIComponent(jobId)}`);
      const data = await resp.json().catch(() => null);
      if (!resp.ok || !data) {
        updatePipelineStatus("Erro ao consultar pipeline.", "error");
        if (btnPipeline) btnPipeline.disabled = false;
        clearInterval(pipelinePollingInterval);
        pipelinePollingInterval = null;
        return;
      }
      const status = data.status;
      const pct = data.progress_percent || 0;

      if (pipelineProgressBar) pipelineProgressBar.style.width = `${pct}%`;
      renderPipelineStages(data.stages, data.current_stage_label);

      if (status === "queued") {
        updatePipelineStatus("Pipeline agendado...", "info");
      } else if (status === "running") {
        updatePipelineStatus(data.current_stage_label || "Processando...", "info");
        if (btnPipeline) btnPipeline.disabled = true;
      } else if (status === "completed") {
        updatePipelineStatus("Pipeline concluído!", "success");
        if (btnPipeline) { btnPipeline.disabled = false; btnPipeline.textContent = "Processar Novamente"; }
        clearInterval(pipelinePollingInterval);
        pipelinePollingInterval = null;
        // Refresh all visible sections
        if (currentFileId) {
          checkExistingStems(currentFileId);
          checkExistingTranscriptions(currentFileId);
          checkExistingDrums(currentFileId);
          checkExistingScore(currentFileId);
          checkExistingArrangement(currentFileId);
        }
      } else if (status === "failed") {
        updatePipelineStatus(data.error || "Pipeline falhou.", "error");
        if (btnPipeline) { btnPipeline.disabled = false; btnPipeline.textContent = "Continuar Processamento"; }
        clearInterval(pipelinePollingInterval);
        pipelinePollingInterval = null;
      }
    } catch (e) {
      updatePipelineStatus("Erro de conexão com pipeline.", "error");
    }
  }

  if (btnPipeline) {
    btnPipeline.addEventListener("click", async () => {
      if (!currentFileId) {
        updatePipelineStatus("Nenhum arquivo analisado.", "error");
        return;
      }
      // Coleta config atual do frontend
      const boxes = document.querySelectorAll("#arrange-instruments input[type=checkbox]:checked");
      const instruments = Array.from(boxes).map((b) => b.value);
      const arrangeStyle = document.getElementById("arrange-style");
      const arrangeDrums = document.getElementById("arrange-include-drums");
      const arrangeDynamics = document.getElementById("arrange-dynamics");
      const arrangeInclude = document.getElementById("arrange-include-originals");
      const arrangeCleanup = document.getElementById("arrange-cleanup");
      const scoreTempo = document.getElementById("score-tempo");
      const scoreTimesig = document.getElementById("score-timesig");
      const scoreQuant = document.getElementById("score-quant");
      const scoreKeymode = document.getElementById("score-keymode");
      const scoreCleanup = document.getElementById("score-cleanup");

      const payload = {
        instruments,
        arrangement_style: arrangeStyle ? arrangeStyle.value : "automatic",
        include_drums: arrangeDrums ? arrangeDrums.checked : true,
        include_original_parts: arrangeInclude ? arrangeInclude.checked : true,
        dynamics: arrangeDynamics ? arrangeDynamics.value : "automatic",
        cleanup_profile: scoreCleanup ? scoreCleanup.value : (arrangeCleanup ? arrangeCleanup.value : "natural"),
        time_signature: scoreTimesig ? scoreTimesig.value : "4/4",
        quantization: scoreQuant ? scoreQuant.value : "1/16",
        key_mode: scoreKeymode ? scoreKeymode.value : "auto",
        tempo: scoreTempo && scoreTempo.value ? Number(scoreTempo.value) : null,
      };

      btnPipeline.disabled = true;
      btnPipeline.textContent = "Processando...";
      updatePipelineStatus("Iniciando pipeline...", "info");
      if (pipelineProgress) pipelineProgress.classList.remove("hidden");

      try {
        const resp = await fetch(`/api/pipeline/${encodeURIComponent(currentFileId)}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await resp.json().catch(() => null);
        if (resp.status === 409) {
          updatePipelineStatus((data && data.detail) || "Pipeline já em andamento.", "error");
          btnPipeline.disabled = false;
          btnPipeline.textContent = "Processar Música Completa";
          return;
        }
        if (!resp.ok || !data || !data.job_id) {
          updatePipelineStatus((data && (data.detail || data.message)) || "Não foi possível iniciar.", "error");
          btnPipeline.disabled = false;
          btnPipeline.textContent = "Processar Música Completa";
          return;
        }
        currentPipelineJobId = data.job_id;
        updatePipelineStatus("Pipeline agendado.", "info");
        if (pipelinePollingInterval) clearInterval(pipelinePollingInterval);
        pipelinePollingInterval = setInterval(() => pollPipelineJob(currentPipelineJobId), 1500);
        setTimeout(() => pollPipelineJob(currentPipelineJobId), 500);
      } catch (err) {
        updatePipelineStatus("Erro ao iniciar pipeline.", "error");
        btnPipeline.disabled = false;
        btnPipeline.textContent = "Processar Música Completa";
      }
    });
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    // BUG FIX #9: cancela pollings de jobs do arquivo anterior
    // Antes: polling de job A continuava após enviar arquivo B,
    // causando cross-file contamination (stems de A exibidos com currentFileId = B)
    cancelAllPolling();
    clearStatus();
    hideAudioInfo();
    hideMusicInfo();
    hideSeparateSection();
    hideStemsInfo();
    hideTranscribeSection();
    hideTranscriptionInfo();
    hideDrumsSection();
    hideScoreSection();
    lastMusic = null;
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
