import { insertNav } from "./nav.js";

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

document.addEventListener("DOMContentLoaded", () => {
    insertNav();
    loadDatasets();
});