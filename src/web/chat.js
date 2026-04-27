import { insertNav } from "./nav.js";

insertNav();

let activeConversationId = null;
let isStreaming = false;
let activeCheckpoint = "latest.pt";
let generationDefaults = null;
let generationParams = {};

// ── API ──────────────────────────────────────────────────────────────────────

async function apiListConversations() {
    const res = await fetch("/api/chat/conversations");
    return res.json();
}

async function apiNewConversation() {
    const res = await fetch("/api/chat/conversations", { method: "POST" });
    return res.json();
}

async function apiGetConversation(id) {
    const res = await fetch(`/api/chat/conversations/${id}`);
    if (!res.ok) return null;
    return res.json();
}

async function apiDeleteConversation(id) {
    await fetch(`/api/chat/conversations/${id}`, { method: "DELETE" });
}

async function apiListCheckpoints() {
    const res = await fetch("/api/chat/checkpoints");
    return res.json();
}

async function apiGenerationDefaults() {
    const res = await fetch("/api/chat/generation-defaults");
    return res.json();
}

async function apiRenameConversation(id, title) {
    await fetch(`/api/chat/conversations/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
    });
}

// ── Sidebar ──────────────────────────────────────────────────────────────────

async function refreshSidebar() {
    const convs = await apiListConversations();
    const list = document.getElementById("convList");
    list.innerHTML = "";

    for (const conv of convs) {
        const item = document.createElement("div");
        item.className = "conv-item" + (conv.id === activeConversationId ? " active" : "");
        item.dataset.id = conv.id;

        const title = document.createElement("span");
        title.className = "conv-title";
        title.textContent = conv.title;
        title.title = "Double-click to rename";
        title.addEventListener("dblclick", (e) => {
            e.stopPropagation();
            const input = document.createElement("input");
            input.className = "conv-title-input";
            input.value = conv.title;
            title.replaceWith(input);
            input.focus();
            input.select();

            const commit = async () => {
                const newTitle = input.value.trim() || conv.title;
                await apiRenameConversation(conv.id, newTitle);
                refreshSidebar();
            };
            input.addEventListener("blur", commit);
            input.addEventListener("keydown", (e) => {
                if (e.key === "Enter")  { input.blur(); }
                if (e.key === "Escape") { input.value = conv.title; input.blur(); }
            });
        });

        const meta = document.createElement("span");
        meta.className = "conv-meta";
        meta.textContent = conv.message_count > 0 ? `${conv.message_count}` : "";

        const del = document.createElement("button");
        del.className = "conv-delete";
        del.textContent = "×";
        del.title = "Delete conversation";
        del.addEventListener("click", async (e) => {
            e.stopPropagation();
            await apiDeleteConversation(conv.id);
            if (activeConversationId === conv.id) {
                activeConversationId = null;
                renderNoConversation();
            }
            refreshSidebar();
        });

        item.appendChild(title);
        item.appendChild(meta);
        item.appendChild(del);

        item.addEventListener("click", () => loadConversation(conv.id));
        list.appendChild(item);
    }
}

// ── Chat panel ───────────────────────────────────────────────────────────────

function renderNoConversation() {
    const panel = document.getElementById("chatPanel");
    panel.innerHTML = `
        <div class="token-bar" id="tokenBar" style="display:none"></div>
        <div class="no-conversation" id="noConversation">Select or start a conversation.</div>
    `;
}

function renderChatPanel() {
    const panel = document.getElementById("chatPanel");
    panel.innerHTML = `
        <div class="token-bar" id="tokenBar" style="display:none">
            <span class="token-bar-label">Context</span>
            <div class="token-bar-track"><div class="token-bar-fill" id="tokenBarFill"></div></div>
            <span class="token-bar-count" id="tokenBarCount"></span>
        </div>
        <div class="messages" id="messages"></div>
        <div class="input-area">
            <textarea id="msgInput" placeholder="Say something to Scout…" rows="1"></textarea>
            <button id="sendBtn">Send</button>
        </div>
    `;

    const input = document.getElementById("msgInput");
    const sendBtn = document.getElementById("sendBtn");

    // Auto-resize textarea
    input.addEventListener("input", () => {
        input.style.height = "auto";
        input.style.height = Math.min(input.scrollHeight, 160) + "px";
    });

    // Send on Enter (Shift+Enter for newline)
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener("click", sendMessage);
}

function formatTimestamp(isoString) {
    if (!isoString) return "";
    try {
        const d = new Date(isoString);
        return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    } catch {
        return "";
    }
}

function appendMessage(role, content, streaming = false, timestamp = null) {
    const messages = document.getElementById("messages");

    // Remove empty state if present
    const empty = messages.querySelector(".empty-state");
    if (empty) empty.remove();

    const msg = document.createElement("div");
    msg.className = `message ${role}`;

    const meta = document.createElement("div");
    meta.className = "message-meta";

    const speaker = document.createElement("span");
    speaker.className = "message-speaker";
    speaker.textContent = role === "user" ? "Trey" : "Scout";

    const ts = document.createElement("span");
    ts.className = "message-timestamp";
    ts.textContent = timestamp ? formatTimestamp(timestamp) : formatTimestamp(new Date().toISOString());

    meta.appendChild(speaker);
    meta.appendChild(ts);

    const bubble = document.createElement("div");
    bubble.className = "message-bubble" + (streaming ? " streaming" : "");
    bubble.textContent = content;

    msg.appendChild(meta);
    msg.appendChild(bubble);
    messages.appendChild(msg);
    messages.scrollTop = messages.scrollHeight;

    return bubble;
}

const BLOCK_SIZE = 1024;  // must match config.py

function estimateTokens(messages) {
    // Rough estimate: characters / 4, plus speaker tags
    let chars = 0;
    for (const msg of messages) {
        chars += (msg.content || "").length + 12; // 12 for [Speaker] + newlines
    }
    return Math.round(chars / 4);
}

function updateTokenBar(messages) {
    const bar = document.getElementById("tokenBar");
    const fill = document.getElementById("tokenBarFill");
    const count = document.getElementById("tokenBarCount");

    if (!messages || messages.length === 0) {
        bar.style.display = "none";
        return;
    }

    const tokens = estimateTokens(messages);
    const pct = Math.min(tokens / BLOCK_SIZE, 1.0);

    bar.style.display = "flex";
    fill.style.width = (pct * 100).toFixed(1) + "%";
    fill.className = "token-bar-fill" + (pct > 0.9 ? " danger" : pct > 0.7 ? " warn" : "");
    count.textContent = `~${tokens.toLocaleString()} / ${BLOCK_SIZE.toLocaleString()} tokens`;
}

function renderMessages(messages) {
    const container = document.getElementById("messages");
    container.innerHTML = "";

    if (messages.length === 0) {
        container.innerHTML = `<div class="empty-state">Start the conversation.</div>`;
        updateTokenBar([]);
        return;
    }

    for (const msg of messages) {
        appendMessage(msg.role, msg.content, false, msg.timestamp);
    }
    updateTokenBar(messages);
}

async function loadConversation(id) {
    activeConversationId = id;

    const conv = await apiGetConversation(id);
    if (!conv) return;

    renderChatPanel();
    renderMessages(conv.messages);
    refreshSidebar();

    document.getElementById("msgInput").focus();
}

async function sendMessage() {
    if (isStreaming || !activeConversationId) return;

    const input = document.getElementById("msgInput");
    const text = input.value.trim();
    if (!text) return;

    input.value = "";
    input.style.height = "auto";
    input.disabled = true;
    document.getElementById("sendBtn").disabled = true;
    isStreaming = true;

    const userTs = new Date().toISOString();
    appendMessage("user", text, false, userTs);

    const bubble = appendMessage("assistant", "", true, new Date().toISOString());

    let accumulated = "";

    try {
        const res = await fetch(`/api/chat/conversations/${activeConversationId}/message`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ conversation_id: activeConversationId, message: text, checkpoint: activeCheckpoint, active_modules: activeModules, generation: generationParams }),
        });

        if (!res.ok) {
            bubble.classList.remove("streaming");
            bubble.textContent = "Error: could not reach Scout.";
            return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith("data: ")) continue;
                const payload = JSON.parse(line.slice(6));

                if (payload.done) {
                    bubble.classList.remove("streaming");
                    break;
                }

                if (payload.token) {
                    accumulated += payload.token;
                    bubble.textContent = accumulated;
                    document.getElementById("messages").scrollTop = document.getElementById("messages").scrollHeight;
                }
            }
        }

    } catch (err) {
        bubble.classList.remove("streaming");
        bubble.textContent = "Error: connection failed.";
        console.error(err);
    } finally {
        isStreaming = false;
        input.disabled = false;
        document.getElementById("sendBtn").disabled = false;
        input.focus();
        refreshSidebar();
        // Refresh token bar from server-side message list (has accurate content)
        const conv = await apiGetConversation(activeConversationId);
        if (conv) updateTokenBar(conv.messages);
    }
}

// ── Generation settings ──────────────────────────────────────────────────────

function bindSlider(id, valId, decimals, onChange) {
    const slider = document.getElementById(id);
    const display = document.getElementById(valId);
    const update = () => {
        const v = parseFloat(slider.value);
        display.textContent = decimals > 0 ? v.toFixed(decimals) : String(Math.round(v));
        onChange(v);
    };
    slider.addEventListener("input", update);
    return (value) => {
        slider.value = value;
        update();
    };
}

async function initGenerationSettings() {
    generationDefaults = await apiGenerationDefaults();
    generationParams = { ...generationDefaults };

    const setTemperature = bindSlider("genTemperature", "genTemperatureVal", 2,
        v => { generationParams.temperature = v; });
    const setVocabulary = bindSlider("genVocabulary", "genVocabularyVal", 0,
        v => { generationParams.vocabulary = Math.round(v); });
    const setRepPenalty = bindSlider("genRepPenalty", "genRepPenaltyVal", 2,
        v => { generationParams.rep_penalty = v; });
    const setMaxTokens = bindSlider("genMaxTokens", "genMaxTokensVal", 0,
        v => { generationParams.max_new_tokens = Math.round(v); });

    setTemperature(generationDefaults.temperature);
    setVocabulary(generationDefaults.vocabulary);
    setRepPenalty(generationDefaults.rep_penalty);
    setMaxTokens(generationDefaults.max_new_tokens);

    document.getElementById("genResetBtn").addEventListener("click", () => {
        setTemperature(generationDefaults.temperature);
        setVocabulary(generationDefaults.vocabulary);
        setRepPenalty(generationDefaults.rep_penalty);
        setMaxTokens(generationDefaults.max_new_tokens);
    });
}

// ── Module toggles ──────────────────────────────────────────────────────────

let activeModules = null;        // null = all modules (server default)
let _checkpointModuleMap = {};   // filename → num_modules

function updateModuleToggles(numModules) {
    const container = document.getElementById("moduleToggles");
    container.innerHTML = "";

    if (numModules <= 1) return;   // nothing to toggle with a single module

    const labelEl = document.createElement("span");
    labelEl.className = "module-toggles-label";
    labelEl.textContent = "Modules";
    container.appendChild(labelEl);

    const names = ["Linguistic foundation", "Conversational", "Reflective", "Inner voice"];

    for (let i = 0; i < numModules; i++) {
        const row = document.createElement("div");
        row.className = "module-toggle-row";

        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.id = `moduleCb${i}`;
        cb.checked = true;
        // Module 0 cannot be disabled — it's the foundation everything runs on
        if (i === 0) cb.disabled = true;

        const lbl = document.createElement("label");
        lbl.htmlFor = `moduleCb${i}`;
        lbl.innerHTML = `Module ${i} <span class="module-tag">${names[i] || ""}</span>`;

        cb.addEventListener("change", syncActiveModules);
        row.appendChild(cb);
        row.appendChild(lbl);
        container.appendChild(row);
    }

    syncActiveModules();
}

function syncActiveModules() {
    const checkboxes = document.querySelectorAll(".module-toggle-row input[type=checkbox]");
    if (checkboxes.length === 0) {
        activeModules = null;
        return;
    }
    activeModules = [];
    checkboxes.forEach((cb, i) => { if (cb.checked) activeModules.push(i); });
}

// ── Checkpoint selector ──────────────────────────────────────────────────────

async function initCheckpointSelector() {
    const select = document.getElementById("checkpointSelect");
    const checkpoints = await apiListCheckpoints();

    select.innerHTML = "";
    for (const ckpt of checkpoints) {
        const opt = document.createElement("option");
        opt.value = ckpt.filename;
        opt.textContent = ckpt.label;
        if (ckpt.filename === "latest.pt") opt.selected = true;
        _checkpointModuleMap[ckpt.filename] = ckpt.num_modules || 1;
        select.appendChild(opt);
    }

    activeCheckpoint = select.value;
    updateModuleToggles(_checkpointModuleMap[activeCheckpoint] || 1);

    select.addEventListener("change", () => {
        activeCheckpoint = select.value;
        updateModuleToggles(_checkpointModuleMap[activeCheckpoint] || 1);
    });
}

// ── Init ─────────────────────────────────────────────────────────────────────

document.getElementById("newConvBtn").addEventListener("click", async () => {
    const conv = await apiNewConversation();
    activeConversationId = conv.id;
    renderChatPanel();
    renderMessages([]);
    await refreshSidebar();
    document.getElementById("msgInput").focus();
});

// Deep-link: /chat/?conversation=<id>
await Promise.all([initCheckpointSelector(), initGenerationSettings()]);

const params = new URLSearchParams(window.location.search);
const deepLinkId = params.get("conversation");
if (deepLinkId) {
    await refreshSidebar();
    await loadConversation(deepLinkId);
} else {
    refreshSidebar();
}