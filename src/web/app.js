async function callHello() {
    const res = await fetch("/api/hello");
    const data = await res.json();
    show(data);
}

async function checkStatus() {
    const res = await fetch("/api/status");
    const data = await res.json();
    show(data);
}

function show(data) {
    document.getElementById("output").textContent =
        JSON.stringify(data, null, 2);
}

async function loadDatasets() {
    const res = await fetch("/api/datasets");
    const datasets = await res.json();

    const container = document.getElementById("datasets");
    container.innerHTML = "";

    for (const ds of datasets) {
        const card = document.createElement("dataset-card");
        card.setAttribute("name", ds.name);
        container.appendChild(card);
    }
}

function tokenizerTest() {
    window.location.href = '/tokenizer';
}

document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("helloButton").addEventListener("click", callHello);
    document.getElementById("statusButton").addEventListener("click", checkStatus);
    document.getElementById("tokenizerButton").addEventListener("click", tokenizerTest);

    loadDatasets();
});
