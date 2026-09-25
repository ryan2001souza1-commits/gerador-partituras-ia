document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("upload-form");
  const fileInput = document.getElementById("file-input");
  const btn = document.getElementById("btn-analisar");
  const statusEl = document.getElementById("status");

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

  function getExtension(filename) {
    const idx = filename.lastIndexOf(".");
    if (idx === -1) return "";
    return filename.slice(idx).toLowerCase();
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearStatus();

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

      if (response.ok && data && data.success) {
        showStatus(data.message || "Arquivo enviado com sucesso.", "success");
      } else {
        const msg =
          (data && (data.detail || data.message)) ||
          "Erro ao enviar o arquivo. Tente novamente.";
        showStatus(msg, "error");
      }
    } catch (err) {
      showStatus("Erro de conexão. Verifique se o servidor está ativo.", "error");
    } finally {
      btn.disabled = false;
      btn.textContent = originalText;
    }
  });
});
