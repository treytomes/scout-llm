class DatasetCard extends HTMLElement {
  connectedCallback() {
    this.datasetName = this.getAttribute("name");

    this.innerHTML = `
      <div class="dataset-card">
        <h3>${this.datasetName}</h3>
        <div class="status">Checking...</div>

        <progress value="0" max="100" style="width:100%"></progress>

        <div class="controls">
          <button class="download">Download</button>
          <button class="preview secondary">Preview</button>
          <button class="delete danger">Delete</button>
        </div>
      </div>
    `;

    this.statusEl = this.querySelector(".status");
    this.progressEl = this.querySelector("progress");

    this.downloadButton = this.querySelector(".download");
    this.previewButton = this.querySelector(".preview");
    this.deleteButton = this.querySelector(".delete");

    this.downloadButton
        .addEventListener("click", () => this.startDownload());

    this.previewButton.addEventListener("click", () => {
      window.location.href = `/datasets/preview?name=${encodeURIComponent(this.datasetName)}`;
    });

    this.deleteButton
        .addEventListener("click", () => this.deleteDataset());

    this.refreshStatus();
  }

  async refreshStatus() {
    const res = await fetch(`/api/datasets/${this.datasetName}/status`);
    const data = await res.json();

    if (data.downloaded) {
      this.statusEl.textContent = "Downloaded";
      this.progressEl.value = 100;

      this.downloadButton.disabled = true;
      this.deleteButton.disabled = false;
    } else if (data.downloading) {
      this.statusEl.textContent = "Downloading...";

      this.downloadButton.disabled = true;
      this.deleteButton.disabled = true;

      this.trackProgress();
    } else {
      this.statusEl.textContent = "Not downloaded";
      this.progressEl.value = 0;

      this.downloadButton.disabled = false;
      this.deleteButton.disabled = true;
    }

    this.previewBtn.disabled = !data.downloaded;
  }

  async startDownload() {
    this.downloadButton.disabled = true;
    this.deleteButton.disabled = true;
    
    await fetch(`/api/datasets/${this.datasetName}/download`, {
      method: "POST"
    });

    this.statusEl.textContent = "Starting download...";
    this.trackProgress();
  }

  async trackProgress() {
    if (this.progressTimer) return;

    this.progressTimer = setInterval(async () => {
      try {
        const res = await fetch(`/api/datasets/${this.datasetName}/progress`);
        const data = await res.json();

        const downloaded = Number(data.downloaded_bytes ?? 0);
        const total = Number(data.total_bytes);

        if (Number.isFinite(total) && total > 0) {
          const percent = (downloaded / total) * 100;

          if (Number.isFinite(percent)) {
            this.progressEl.value = percent;
            this.statusEl.textContent = `Downloading ${percent.toFixed(1)}%`;
          }
        } else {
          this.statusEl.textContent = `${downloaded.toLocaleString()} bytes downloaded`;
        }

        if (data.complete) {
          clearInterval(this.progressTimer);
          this.progressTimer = null;

          this.progressEl.value = 100;
          this.statusEl.textContent = "Downloaded";
          this.previewBtn.disabled = false;
        }

      } catch (err) {
        console.error("Progress update failed", err);
      }

    }, 1000);
  }

  async deleteDataset() {
    await fetch(`/api/datasets/${this.datasetName}`, {
      method: "DELETE"
    });

    this.progressEl.value = 0;
    this.statusEl.textContent = "Deleted";
  }
}

customElements.define("dataset-card", DatasetCard);