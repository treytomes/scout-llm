const params = new URLSearchParams(window.location.search);
const datasetName = params.get("name");

let page = 0;
let limit = 20;
let totalRows = 0;

async function loadPage() {
  let res, data;
  try {
    res = await fetch(
      `/api/datasets/${datasetName}/preview?limit=${limit}&page=${page}`
    );
    data = await res.json();
  } catch (err) {
    document.getElementById("metaDataset").textContent = `Error loading page: ${err.message}`;
    return;
  }
  if (!res.ok) {
    document.getElementById("metaDataset").textContent = `Error ${res.status}: ${data?.detail ?? res.statusText}`;
    return;
  }

  totalRows = data.total_rows;

  document.getElementById("title").textContent =
    `Dataset Preview: ${datasetName}`;

  document.getElementById("metaDataset").textContent =
    `Dataset: ${data.name}`;

  document.getElementById("metaSplit").textContent =
    `Split: ${data.split_name ?? "unknown"}`;

  document.getElementById("metaRows").textContent =
    `Rows: ${data.total_rows}`;

  document.getElementById("metaPage").textContent =
    `Page: ${data.page}`;

  document.getElementById("metaPageCount").textContent =
    `# Pages: ${Math.ceil(totalRows / limit)}`;

  renderTable(data.rows);
}

function renderTable(rows) {

  const table = document.getElementById("dataTable");
  table.innerHTML = "";

  if (!rows.length) return;

  const columns = Object.keys(rows[0]);

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");

  const th = document.createElement("th");
  th.textContent = "#";
  headRow.appendChild(th);

  for (const c of columns) {
    const th = document.createElement("th");
    th.textContent = c;
    headRow.appendChild(th);
  }

  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");

  for (var rowIndex = 0; rowIndex < rows.length; rowIndex++) {
    const row = rows[rowIndex];
    const tr = document.createElement("tr");

    const td = document.createElement("td");
    td.textContent = page * limit + rowIndex;
    tr.appendChild(td);

    for (const c of columns) {
      const td = document.createElement("td");
      const raw = JSON.stringify(row[c]);
      td.textContent = raw.length > 300 ? raw.slice(0, 300) + "…" : raw;
      td.title = raw.length > 300 ? raw : "";
      tr.appendChild(td);
    }

    tbody.appendChild(tr);
  }

  table.appendChild(tbody);
}

function lastPage() {
  return Math.max(0, Math.ceil(totalRows / limit) - 1);
}

function formatNumber(n) {
  return Number(n).toLocaleString();
}

function renderPlanTable(data) {
  const table = document.getElementById("planTable");
  table.innerHTML = "";

  const rows = [
    ["Dataset", data.dataset],
    ["Split", data.split],
    ["Total Tokens", formatNumber(data.total_tokens)],
    ["Vocabulary Size", formatNumber(data.vocab_size)],
    ["Sequence Length", formatNumber(data.sequence_length)],
    ["Batch Size", formatNumber(data.batch_size)],
    ["Tokens / Step", formatNumber(data.tokens_per_step)],
    ["Training Samples", formatNumber(data.training_samples)],
    ["Steps / Epoch", formatNumber(data.steps_per_epoch)]
  ];

  const tbody = document.createElement("tbody");

  for (const [label, value] of rows) {
    const tr = document.createElement("tr");

    const tdLabel = document.createElement("td");
    tdLabel.textContent = label;

    const tdValue = document.createElement("td");
    tdValue.textContent = value;

    tr.appendChild(tdLabel);
    tr.appendChild(tdValue);
    tbody.appendChild(tr);
  }

  table.appendChild(tbody);
}

async function generatePlan() {
  const blockSize = Number(document.getElementById("blockSizeInput").value);
  const batchSize = Number(document.getElementById("batchSizeInput").value);

  let res, data;
  try {
    res = await fetch(`/api/datasets/${datasetName}/training_plan`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ block_size: blockSize, batch_size: batchSize })
    });
    data = await res.json();
  } catch (err) {
    alert(`Training plan failed: ${err.message}`);
    return;
  }
  if (!res.ok) {
    alert(`Training plan error ${res.status}: ${data?.detail ?? res.statusText}`);
    return;
  }
  renderPlanTable(data);
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("firstBtn").onclick = () => {
    page = 0;
    loadPage();
  };

  document.getElementById("prevBtn").onclick = () => {
    if (page > 0) page--;
    loadPage();
  };

  document.getElementById("nextBtn").onclick = () => {
    if (page < lastPage()) page++;
    loadPage();
  };

  document.getElementById("lastBtn").onclick = () => {
    page = lastPage();
    loadPage();
  };

  document.getElementById("homeButton").onclick= () => {
    window.location.href = '/';
  }

  document.getElementById("planBtn").onclick = generatePlan;
  
  loadPage();
});