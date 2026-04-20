class DatasetCard extends HTMLElement {
  connectedCallback() {
    this.datasetName = this.getAttribute("name");

    this.innerHTML = `
      <div class="dataset-card">
        <h3>${this.datasetName}</h3>
        <div class="status">Checking...</div>
        <progress value="0" max="100" style="width:100%"></progress>
        <div class="pipeline-status" style="font-size:0.85em; color:#888; margin: 4px 0 8px;"></div>
        <div class="controls">
          <button class="download">Download</button>
          <button class="normalize secondary" style="display:none">Normalize</button>
          <button class="preview secondary">Preview</button>
          <button class="delete danger">Delete</button>
        </div>
      </div>
    `;

    this.statusEl        = this.querySelector(".status");
    this.progressEl      = this.querySelector("progress");
    this.pipelineEl      = this.querySelector(".pipeline-status");
    this.downloadButton  = this.querySelector(".download");
    this.normalizeButton = this.querySelector(".normalize");
    this.previewButton   = this.querySelector(".preview");
    this.deleteButton    = this.querySelector(".delete");

    this.downloadButton.addEventListener("click",  () => this.startDownload());
    this.normalizeButton.addEventListener("click", () => this.startNormalize());
    this.previewButton.addEventListener("click",   () => {
      window.location.href = `/datasets/preview?name=${encodeURIComponent(this.datasetName)}`;
    });
    this.deleteButton.addEventListener("click", () => this.deleteDataset());

    this.refreshStatus();
  }

  disconnectedCallback() {
    if (this.progressTimer) {
      clearInterval(this.progressTimer);
      this.progressTimer = null;
    }
    if (this.normalizeTimer) {
      clearInterval(this.normalizeTimer);
      this.normalizeTimer = null;
    }
  }

  async refreshStatus() {
    const res = await fetch(`/api/datasets/${this.datasetName}/status`);
    const data = await res.json();

    const downloaded = data.downloaded ?? data.exists ?? false;
    const normalized = data.normalized ?? false;
    const tokenized  = data.tokenized  ?? false;

    const steps = [
      downloaded ? "raw ✓" : "raw ✗",
      normalized ? "normalized ✓" : "normalized ✗",
      tokenized  ? "tokenized ✓"  : "tokenized ✗",
    ];
    this.pipelineEl.textContent = steps.join("  →  ");

    if (data.downloading) {
      this.statusEl.textContent = "Downloading...";
      this.downloadButton.disabled  = true;
      this.normalizeButton.style.display = "none";
      this.deleteButton.disabled    = true;
      this.trackProgress();
      return;
    }

    if (downloaded) {
      this.statusEl.textContent    = normalized ? "Ready" : "Downloaded — needs normalization";
      this.progressEl.value        = 100;
      this.downloadButton.disabled = true;
      this.deleteButton.disabled   = false;
      this.normalizeButton.style.display = normalized ? "none" : "inline-block";
      this.normalizeButton.disabled = false;
    } else {
      this.statusEl.textContent    = "Not downloaded";
      this.progressEl.value        = 0;
      this.downloadButton.disabled = false;
      this.deleteButton.disabled   = true;
      this.normalizeButton.style.display = "none";
    }

    this.previewButton.disabled = !downloaded;
  }

  async startDownload() {
    this.downloadButton.disabled = true;
    this.deleteButton.disabled   = true;
    this.normalizeButton.style.display = "none";
    this.normalizeButton.disabled = true;
    let res;
    try {
      res = await fetch(`/api/datasets/${this.datasetName}/download`, { method: "POST" });
    } catch (err) {
      this.statusEl.textContent = `Download failed: ${err.message}`;
      this.downloadButton.disabled = false;
      this.deleteButton.disabled = false;
      return;
    }
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      this.statusEl.textContent = `Download error ${res.status}: ${body?.detail ?? res.statusText}`;
      this.downloadButton.disabled = false;
      this.deleteButton.disabled = false;
      return;
    }
    this.statusEl.textContent = "Starting download...";
    this.trackProgress();
  }

  async startNormalize() {
    this.normalizeButton.disabled = true;
    this.statusEl.textContent = "Normalizing...";
    await fetch(`/api/datasets/${this.datasetName}/normalize`, { method: "POST" });
    this.pollNormalize();
  }

  pollNormalize() {
    if (this.normalizeTimer) return;
    this.normalizeTimer = setInterval(async () => {
      const res  = await fetch(`/api/datasets/${this.datasetName}/status`);
      const data = await res.json();
      if (data.normalized) {
        clearInterval(this.normalizeTimer);
        this.normalizeTimer = null;
        this.refreshStatus();
      }
    }, 1500);
  }

  async trackProgress() {
    if (this.progressTimer) return;
    this.progressTimer = setInterval(async () => {
      try {
        const res  = await fetch(`/api/datasets/${this.datasetName}/progress`);
        const data = await res.json();

        const downloaded = Number(data.downloaded_bytes ?? 0);
        const total      = Number(data.total_bytes);

        if (Number.isFinite(total) && total > 0) {
          const percent = (downloaded / total) * 100;
          if (Number.isFinite(percent)) {
            this.progressEl.value     = percent;
            this.statusEl.textContent = `Downloading ${percent.toFixed(1)}%`;
          }
        } else {
          this.statusEl.textContent = `${downloaded.toLocaleString()} bytes downloaded`;
        }

        if (data.complete) {
          clearInterval(this.progressTimer);
          this.progressTimer = null;
          this.refreshStatus();
        }
      } catch (err) {
        console.error("Progress update failed", err);
      }
    }, 1000);
  }

  async deleteDataset() {
    this.deleteButton.disabled = true;
    let res;
    try {
      res = await fetch(`/api/datasets/${this.datasetName}`, { method: "DELETE" });
    } catch (err) {
      this.statusEl.textContent = `Delete failed: ${err.message}`;
      this.deleteButton.disabled = false;
      return;
    }
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      this.statusEl.textContent = `Delete error ${res.status}: ${body?.detail ?? res.statusText}`;
      this.deleteButton.disabled = false;
      return;
    }
    if (this.progressTimer) { clearInterval(this.progressTimer); this.progressTimer = null; }
    if (this.normalizeTimer) { clearInterval(this.normalizeTimer); this.normalizeTimer = null; }
    this.progressEl.value        = 0;
    this.statusEl.textContent    = "Deleted";
    this.normalizeButton.style.display = "none";
    this.downloadButton.disabled = false;
    this.pipelineEl.textContent  = "";
  }
}

customElements.define("dataset-card", DatasetCard);