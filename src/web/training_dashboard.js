import { insertNav } from "./nav.js";

let chartInstance = null;
let statusPoller = null;
let liveMode = false;  // true while a job is running — live metrics feed into chart

// ── Training control ──────────────────────────────────────────────────────────

async function fetchStatus() {
    const res = await fetch("/api/training/status");
    return res.json();
}

function applyStatus(status) {
    const indicator  = document.getElementById("trainingIndicator");
    const statusText = document.getElementById("trainingStatusText");
    const startBtn   = document.getElementById("startBtn");
    const stopBtn    = document.getElementById("stopBtn");
    const progressWrap  = document.getElementById("trainingProgressWrap");
    const progressBar   = document.getElementById("trainingProgress");
    const progressLabel = document.getElementById("trainingProgressLabel");
    const errorEl    = document.getElementById("trainingError");

    indicator.className = "training-indicator";

    if (status.error) {
        indicator.classList.add("error");
        statusText.textContent = `Error: ${status.error}`;
        errorEl.textContent = status.error;
        errorEl.style.display = "block";
    } else {
        errorEl.style.display = "none";
    }

    if (status.running) {
        indicator.classList.add("running");
        const m = status.latest_metrics;
        const step = m?.step ?? 0;
        const maxSteps = status.max_steps ?? 0;
        statusText.textContent = `Training — ${status.dataset} — step ${step.toLocaleString()} / ${maxSteps.toLocaleString()}`;
        startBtn.style.display = "none";
        stopBtn.style.display  = "inline-block";

        if (maxSteps > 0) {
            progressWrap.style.display = "block";
            progressBar.value = (step / maxSteps) * 100;
            const eta = m?.eta;
            progressLabel.textContent = eta != null ? `ETA ${formatEta(eta)}` : "";
        }

        if (m) updateMetricsFromObject(m);
        if (liveMode) appendLivePoint(m);

    } else if (status.completed) {
        statusText.textContent = status.error ? "Stopped with error" : `Completed — ${status.dataset}`;
        startBtn.style.display = "inline-block";
        stopBtn.style.display  = "none";
        progressWrap.style.display = "none";
        stopPolling();
        liveMode = false;
        loadTrainingLogs();  // refresh log list now that a new log exists

    } else {
        statusText.textContent = status.message ?? "Idle";
        startBtn.style.display = "inline-block";
        stopBtn.style.display  = "none";
        progressWrap.style.display = "none";
    }
}

function formatEta(seconds) {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

function startPolling() {
    if (statusPoller) return;
    statusPoller = setInterval(async () => {
        const status = await fetchStatus();
        applyStatus(status);
    }, 2000);
}

function stopPolling() {
    if (statusPoller) {
        clearInterval(statusPoller);
        statusPoller = null;
    }
}

function showTrainingDialog() {
    document.getElementById("trainingDialog").style.display = "block";
    document.getElementById("startBtn").style.display = "none";
}

function hideTrainingDialog() {
    document.getElementById("trainingDialog").style.display = "none";
    document.getElementById("startBtn").style.display = "inline-block";
}

async function startTraining() {
    const dataset = document.getElementById("tdDataset").value.trim() || "scout_dialogue";
    const maxSteps = parseInt(document.getElementById("tdSteps").value, 10) || 400;
    const lr = parseFloat(document.getElementById("tdLr").value) || 5e-5;
    const warmup = parseInt(document.getElementById("tdWarmup").value, 10) || 50;
    const batchSize = parseInt(document.getElementById("tdBatch").value, 10) || 8;
    const moduleConfig = document.getElementById("tdModuleConfig").value;
    const freezeModule0 = document.getElementById("tdFreezeModule0").checked;
    const freezeLC = document.getElementById("tdFreezeLC").checked;
    const resetOptimizer = document.getElementById("tdResetOptimizer").checked;

    const freezeModules = freezeModule0 ? [0] : null;

    hideTrainingDialog();
    document.getElementById("startBtn").disabled = true;

    try {
        const res = await fetch("/api/training/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                dataset_name: dataset,
                batch_size: batchSize,
                max_steps: maxSteps,
                module_config: moduleConfig,
                reset_optimizer: resetOptimizer,
                freeze_modules: freezeModules,
                freeze_language_core: freezeLC,
            }),
        });
        if (!res.ok) {
            const body = await res.json().catch(() => ({}));
            document.getElementById("trainingStatusText").textContent =
                `Start failed: ${body?.detail ?? res.statusText}`;
            document.getElementById("startBtn").disabled = false;
            return;
        }
        liveMode = true;
        resetChart();
        startPolling();
    } catch (err) {
        document.getElementById("trainingStatusText").textContent = `Start failed: ${err.message}`;
        document.getElementById("startBtn").disabled = false;
    }
}

async function stopTraining() {
    const btn = document.getElementById("stopBtn");
    btn.disabled = true;
    await fetch("/api/training/stop", { method: "POST" }).catch(() => {});
    btn.disabled = false;
}

// ── Metrics display ───────────────────────────────────────────────────────────

function updateMetricsFromObject(m) {
    document.getElementById("m-step").textContent = m.step?.toLocaleString() ?? "—";
    document.getElementById("m-loss").textContent = m.loss?.toFixed(3) ?? "—";
    document.getElementById("m-val").textContent  = m.val_loss?.toFixed(3) ?? "—";
    document.getElementById("m-tps").textContent  = Math.round(m.tokens_per_sec ?? 0);
    document.getElementById("m-eta").textContent  = m.eta != null ? formatEta(m.eta) : "";
}

function updateMetrics(rows) {
    if (!rows.length) return;
    updateMetricsFromObject(rows[rows.length - 1]);
}

// ── Chart ─────────────────────────────────────────────────────────────────────

function resetChart() {
    if (chartInstance) { chartInstance.destroy(); chartInstance = null; }
}

function renderChart(rows) {
    resetChart();
    const ctx = document.getElementById("lossChart");
    chartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: rows.map(r => r.step),
            datasets: [
                { label: "train", data: rows.map(r => r.loss),     borderColor: "#378ADD", tension: 0.3, pointRadius: 2 },
                { label: "val",   data: rows.map(r => r.val_loss), borderColor: "#1D9E75", borderDash: [5,3], tension: 0.3, pointRadius: 2 },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: "#9aa3ad", maxTicksLimit: 10 }, grid: { color: "#2a2f3a" } },
                y: { ticks: { color: "#9aa3ad" }, grid: { color: "#2a2f3a" } },
            },
        },
    });
}

function appendLivePoint(m) {
    if (!m || !chartInstance) return;
    chartInstance.data.labels.push(m.step);
    chartInstance.data.datasets[0].data.push(m.loss);
    chartInstance.data.datasets[1].data.push(m.val_loss ?? null);
    chartInstance.update("none");
}

// ── Log table ─────────────────────────────────────────────────────────────────

async function loadTrainingLogs() {
    const res = await fetch("/api/training/logs");
    const logs = await res.json();

    const body = document.getElementById("trainingLogsBody");
    body.innerHTML = "";

    if (!logs.length) {
        body.innerHTML = `<tr><td colspan="4" class="empty-state">No training logs yet.</td></tr>`;
        return;
    }

    logs.forEach(log => {
        const tr = document.createElement("tr");
        const btn = document.createElement("button");
        btn.className = "secondary";
        btn.textContent = "Load";
        btn.addEventListener("click", () => loadTrainingLog(log.filename));

        tr.innerHTML = `<td>${log.filename}</td><td>${log.date}</td><td>${log.index}</td>`;
        const td = document.createElement("td");
        td.appendChild(btn);
        tr.appendChild(td);
        body.appendChild(tr);
    });
}

async function loadTrainingLog(filename) {
    const res = await fetch(`/api/training/logs/${filename}`);
    const log = await res.json();
    renderLogTable(log.entries);
    updateMetrics(log.entries);
    renderChart(log.entries);
    liveMode = false;
}

function renderLogTable(rows) {
    const body = document.getElementById("logDataBody");
    body.innerHTML = "";

    rows.forEach(r => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${r.step?.toLocaleString()}</td>
            <td>${r.loss?.toFixed(4) ?? "—"}</td>
            <td>${r.avg_loss?.toFixed(4) ?? "—"}</td>
            <td>${r.lr?.toExponential(2) ?? "—"}</td>
            <td>${r.val_loss ? r.val_loss.toFixed(4) : "—"}</td>
            <td>${Math.round(r.tokens_per_sec ?? 0)}</td>
        `;
        body.appendChild(tr);
    });
}

// ── Init ──────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", async () => {
    insertNav();

    document.getElementById("startBtn").addEventListener("click", showTrainingDialog);
    document.getElementById("tdConfirmBtn").addEventListener("click", startTraining);
    document.getElementById("tdCancelBtn").addEventListener("click", hideTrainingDialog);
    document.getElementById("stopBtn").addEventListener("click", stopTraining);

    loadTrainingLogs();

    const status = await fetchStatus();
    applyStatus(status);

    if (status.running) {
        liveMode = false;  // existing job — don't append live points, just poll status
        startPolling();
    }
});