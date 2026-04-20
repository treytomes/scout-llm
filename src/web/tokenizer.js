import { insertNav } from "./nav.js";

async function loadInfo() {
    const res = await fetch("/api/tokenizer/info");
    const data = await res.json();
    document.getElementById("tokenizerName").textContent = data.name;
}

async function tokenize() {
    const text = document.getElementById("inputText").value;

    const res = await fetch("/api/tokenizer/tokenize", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({text})
    });

    const data = await res.json();

    document.getElementById("tokenCount").textContent = data.token_count;
    document.getElementById("tokenStrings").textContent = JSON.stringify(data.token_strings, null, 2);
    document.getElementById("tokenIds").textContent = JSON.stringify(data.tokens, null, 2);

    renderTokenBoxes(data.token_strings, data.tokens);
    renderRoundTrip(text, data);
    renderAlignment(text, data.token_strings, data.offsets);
}

function renderTokenBoxes(strings, ids) {
    const container = document.getElementById("tokenBoxes");
    container.innerHTML = "";

    for (let i = 0; i < strings.length; i++) {

        const el = document.createElement("span");
        el.className = "token-box";

        const token = strings[i];

        if (token.trim() === "") {
            el.classList.add("whitespace");
        }

        if (token.startsWith("<") && token.endsWith(">")) {
            el.classList.add("special");
        }

        const visible = token
            .replace(/ /g, "·")
            .replace(/\n/g, "↵")
            .replace(/\t/g, "→");

        el.textContent = visible;
        el.title = "id: " + ids[i] + " | raw: " + token;

        container.appendChild(el);
    }
}

function renderRoundTrip(original, data) {
    const reconstructed = data.decoded_text;

    const pre = document.getElementById("reconstructedText");
    const status = document.getElementById("roundTripStatus");

    pre.textContent = reconstructed;

    if (original === reconstructed) {
        status.textContent = "Exact match";
        status.style.color = "#10b981";
    } else {
        status.textContent = "Mismatch";
        status.style.color = "#ef4444";
    }
}

function renderAlignment(text, tokens, offsets) {
    const container = document.getElementById("alignmentView");
    container.innerHTML = "";

    for (let i = 0; i < tokens.length; i++) {
        const [start, end] = offsets[i];

        const span = document.createElement("span");
        span.className = "token-box";

        const piece = text.slice(start, end)
            .replace(/ /g, "·")
            .replace(/\n/g, "↵")
            .replace(/\t/g, "→");

        span.textContent = piece || "∅";
        span.title = tokens[i] + " | chars " + start + "-" + end;

        container.appendChild(span);
    }
}

document.addEventListener("DOMContentLoaded", () => {
    insertNav();
    loadInfo();

    document.getElementById("tokenizeButton").onclick = () => {
        tokenize();
    }
});