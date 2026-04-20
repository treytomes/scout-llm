let chartInstance = null;

async function loadTrainingLogs() {
    const res=await fetch("/api/training/logs");
    const logs=await res.json();

    const body=document.getElementById("trainingLogsBody");
    body.innerHTML="";

    logs.forEach(log => {
        const tr = document.createElement("tr");

        const btn = document.createElement("button");
        btn.textContent = "Load";
        btn.addEventListener("click", () => loadTrainingLog(log.filename));

        tr.innerHTML = `
            <td>${log.filename}</td>
            <td>${log.date}</td>
            <td>${log.index}</td>
        `;

        const td = document.createElement("td");
        td.appendChild(btn);
        tr.appendChild(td);

        body.appendChild(tr);
    });
}

async function loadTrainingLog(filename) {
    const res=await fetch(`/api/training/logs/${filename}`);
    const log=await res.json();

    renderLogTable(log.entries);
    updateMetrics(log.entries);
    renderChart(log.entries);
}

function renderLogTable(rows) {
    const body=document.getElementById("logDataBody");
    body.innerHTML="";

    rows.forEach(r=>{
        const tr=document.createElement("tr");

        tr.innerHTML=`
            <td>${r.step}</td>
            <td>${r.loss?.toFixed(4) ?? "—"}</td>
            <td>${r.avg_loss?.toFixed(4) ?? "—"}</td>
            <td>${r.lr?.toExponential(2) ?? "—"}</td>
            <td>${r.val_loss ? r.val_loss.toFixed(4) : "—"}</td>
            <td>${Math.round(r.tokens_per_sec ?? 0)}</td>
        `;

        body.appendChild(tr);
    });
}

function updateMetrics(rows) {
    if (!rows.length) return;

    const last = rows[rows.length - 1];

    document.getElementById("m-step").textContent=last.step;
    document.getElementById("m-loss").textContent=last.loss?.toFixed(3) ?? "—";
    document.getElementById("m-val").textContent=last.val_loss?.toFixed(3) ?? "—";
    document.getElementById("m-tps").textContent=Math.round(last.tokens_per_sec ?? 0);
}

function renderChart(rows) {
    const ctx=document.getElementById("lossChart");

    if (chartInstance) {
        chartInstance.destroy();
    }

    const labels = rows.map(r => r.step);
    const train = rows.map(r => r.loss);
    const val = rows.map(r => r.val_loss);

    chartInstance=new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [
                {
                    label: "train",
                    data: train,
                    borderColor: "#378ADD",
                    tension: 0.3
                },
                {
                    label: "val",
                    data: val,
                    borderColor: "#1D9E75",
                    borderDash: [5,3],
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

document.addEventListener("DOMContentLoaded", () => {
    loadTrainingLogs();
});
