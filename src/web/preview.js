const params = new URLSearchParams(window.location.search);
const datasetName = params.get("name");
console.log(params);

let page = 0;
let limit = 20;
let totalRows = 0;

async function loadPage() {
  const res = await fetch(
    `/api/datasets/${datasetName}/preview?limit=${limit}&page=${page}`
  );

  const data = await res.json();

  totalRows = data.total_rows;

  document.getElementById("title").textContent =
    `Dataset Preview: ${datasetName}`;

  document.getElementById("metaDataset").textContent =
    `Dataset: ${data.dataset}`;

  document.getElementById("metaSplit").textContent =
    `Split: ${data.split ?? "unknown"}`;

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
      td.textContent = JSON.stringify(row[c]);
      tr.appendChild(td);
    }

    tbody.appendChild(tr);
  }

  table.appendChild(tbody);
}

function lastPage() {
  return Math.floor(totalRows / limit);
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
    page++;
    loadPage();
  };

  document.getElementById("lastBtn").onclick = () => {
    page = lastPage();
    loadPage();
  };

  document.getElementById("homeButton").onclick= () => {
    window.location.href = '/';
  }

  loadPage();
});