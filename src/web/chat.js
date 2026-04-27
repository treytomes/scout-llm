import { insertNav } from "./nav.js";

insertNav();

let activeConversationId = null;
let activeConversationStatus = "active";
let isStreaming = false;
let activeCheckpoint = "latest.pt";
let generationDefaults = null;
let generationParams = {};
let userName = "Trey";
let dreamPollInterval = null;

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
    const res = await fetch(`/api/chat/conversations/${id}`, { method: "DELETE" });
    return res.ok;
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

async function apiEditMessage(conversationId, messageIndex, content) {
    const res = await fetch(`/api/chat/conversations/${conversationId}/messages/${messageIndex}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
    });
    return res.ok;
}

async function apiSetStatus(conversationId, status) {
    const res = await fetch(`/api/chat/conversations/${conversationId}/status`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status }),
    });
    return res.ok;
}

async function apiStartDream(conversationId) {
    const res = await fetch(`/api/chat/conversations/${conversationId}/dream`, {
        method: "POST",
    });
    return res.ok;
}

async function apiDreamStatus(conversationId) {
    const res = await fetch(`/api/chat/conversations/${conversationId}/dream`);
    if (!res.ok) return null;
    return res.json();
}

async function apiCountTokens(text) {
    try {
        const res = await fetch("/api/tokenizer/tokenize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });
        if (!res.ok) return null;
        const data = await res.json();
        return data.token_count;
    } catch {
        return null;
    }
}

// ── Status helpers ────────────────────────────────────────────────────────────

function statusLabel(status) {
    return { active: "", full: "Full", training: "Training…", locked: "🔒" }[status] || "";
}

function isEditable(status) {
    return status === "active" || status === "full";
}

function canSend(status) {
    return status === "active";
}

// ── Sidebar ──────────────────────────────────────────────────────────────────

async function refreshSidebar() {
    const convs = await apiListConversations();
    const list = document.getElementById("convList");
    list.innerHTML = "";

    for (const conv of convs) {
        const status = conv.status || "active";
        const item = document.createElement("div");
        item.className = "conv-item" +
            (conv.id === activeConversationId ? " active" : "") +
            (status !== "active" ? ` conv-${status}` : "");
        item.dataset.id = conv.id;

        const title = document.createElement("span");
        title.className = "conv-title";
        title.textContent = conv.title;
        title.title = status === "locked" ? "Locked — permanent record" : "Double-click to rename";

        if (status !== "locked") {
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
        }

        const meta = document.createElement("span");
        meta.className = "conv-meta";
        const badge = statusLabel(status);
        meta.textContent = badge || (conv.message_count > 0 ? `${conv.message_count}` : "");

        const del = document.createElement("button");
        del.className = "conv-delete";
        del.textContent = "×";
        del.title = status === "locked" ? "Locked — cannot delete" : "Delete conversation";
        del.disabled = status === "locked";
        del.addEventListener("click", async (e) => {
            e.stopPropagation();
            if (status === "locked") return;
            const ok = await apiDeleteConversation(conv.id);
            if (ok) {
                if (activeConversationId === conv.id) {
                    activeConversationId = null;
                    activeConversationStatus = "active";
                    renderNoConversation();
                }
                refreshSidebar();
            }
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

function renderChatPanel(status = "active") {
    const panel = document.getElementById("chatPanel");
    const locked = status === "locked";
    const training = status === "training";
    const full = status === "full";
    const sendDisabled = !canSend(status);

    panel.innerHTML = `
        <div class="chat-header" id="chatHeader">
            <div class="chat-header-user">
                <span class="chat-header-label">You are</span>
                <span class="chat-header-username" id="userNameDisplay" title="Click to change">${escapeHtml(userName)}</span>
            </div>
            <div class="token-bar" id="tokenBar" style="display:none">
                <span class="token-bar-label">Context</span>
                <div class="token-bar-track"><div class="token-bar-fill" id="tokenBarFill"></div></div>
                <span class="token-bar-count" id="tokenBarCount"></span>
            </div>
            <div class="conv-status-controls" id="convStatusControls">
                ${locked ? `<span class="conv-status-badge locked">🔒 Locked</span>` : ""}
                ${training ? `<span class="conv-status-badge training">Training…</span>` : ""}
                ${full && !training ? `<span class="conv-status-badge full">Full</span>` : ""}
                ${full && !training && !locked ? `<button class="conv-action-btn dream-btn" id="dreamBtn">Commit to memory</button>` : ""}
                ${(full || status === "active") && !training && !locked ? `<button class="conv-action-btn mark-full-btn" id="markFullBtn">${full ? "Reopen" : "Mark full"}</button>` : ""}
            </div>
        </div>
        <div class="messages" id="messages"></div>
        ${!locked && !training ? `
        <div class="input-area">
            <textarea id="msgInput" placeholder="${sendDisabled ? "Context full — edit messages or commit to memory." : "Say something to Scout…"}" rows="1" ${sendDisabled ? "disabled" : ""}></textarea>
            <button id="sendBtn" ${sendDisabled ? "disabled" : ""}>Send</button>
        </div>` : `
        <div class="input-area input-area-disabled">
            <span class="input-area-msg">${locked ? "This conversation is locked. It is part of Scout's permanent record." : "Training in progress…"}</span>
        </div>`}
    `;

    if (!locked) {
        document.getElementById("userNameDisplay").addEventListener("click", openUsernameEditor);
    }

    const markFullBtn = document.getElementById("markFullBtn");
    if (markFullBtn) {
        markFullBtn.addEventListener("click", async () => {
            const newStatus = status === "full" ? "active" : "full";
            const ok = await apiSetStatus(activeConversationId, newStatus);
            if (ok) {
                activeConversationStatus = newStatus;
                await loadConversation(activeConversationId);
            }
        });
    }

    const dreamBtn = document.getElementById("dreamBtn");
    if (dreamBtn) {
        dreamBtn.addEventListener("click", async () => {
            if (!confirm("Commit this conversation to Scout's memory?\n\nThis will run a short training session and then permanently lock the conversation. This cannot be undone.")) return;
            const ok = await apiStartDream(activeConversationId);
            if (ok) {
                activeConversationStatus = "training";
                await loadConversation(activeConversationId);
                startDreamPoll(activeConversationId);
            }
        });
    }

    if (!canSend(status)) return;

    const input = document.getElementById("msgInput");
    const sendBtn = document.getElementById("sendBtn");

    input.addEventListener("input", () => {
        input.style.height = "auto";
        input.style.height = Math.min(input.scrollHeight, 160) + "px";
    });

    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener("click", sendMessage);
}

function openDreamModal() {
    document.getElementById("dreamModalBackdrop").style.display = "flex";
    document.getElementById("dreamModalLog").innerHTML = "";
    document.getElementById("dreamModalProgressFill").style.width = "0%";
    document.getElementById("dreamModalPhase").textContent = "starting";
    document.getElementById("dreamModalElapsed").textContent = "";
}

function closeDreamModal() {
    document.getElementById("dreamModalBackdrop").style.display = "none";
}

function dreamLog(text) {
    const log = document.getElementById("dreamModalLog");
    const line = document.createElement("div");
    line.className = "dream-log-line";
    line.textContent = text;
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
}

function startDreamPoll(conversationId) {
    stopDreamPoll();
    openDreamModal();
    let lastPhase = null;

    dreamPollInterval = setInterval(async () => {
        if (conversationId !== activeConversationId) { stopDreamPoll(); return; }
        const s = await apiDreamStatus(conversationId);
        if (!s) return;

        // Update progress bar
        const fill = document.getElementById("dreamModalProgressFill");
        if (fill) fill.style.width = `${s.progress}%`;

        // Update phase label
        const phaseEl = document.getElementById("dreamModalPhase");
        if (phaseEl) phaseEl.textContent = s.phase;

        // Log phase transitions
        if (s.phase !== lastPhase) {
            lastPhase = s.phase;
            if (s.phase === "sft") {
                const mode = s.use_lora ? "LoRA adapter" : "direct weights";
                dreamLog(`SFT pass — ${s.sft_steps} steps [${mode}]`);
            } else if (s.phase === "dpo") {
                dreamLog(`DPO pass — ${s.dpo_steps} steps`);
            } else if (s.phase === "locking") {
                dreamLog("Saving…");
            } else if (s.phase === "done") {
                dreamLog("Complete. Conversation locked.");
            }
        }

        // Update elapsed
        const elapsedEl = document.getElementById("dreamModalElapsed");
        if (elapsedEl && s.elapsed != null) {
            const secs = Math.round(s.elapsed);
            elapsedEl.textContent = secs < 60 ? `${secs}s` : `${Math.floor(secs/60)}m ${secs%60}s`;
        }

        if (s.error) {
            dreamLog(`Error: ${s.error}`);
        }

        if (s.completed) {
            stopDreamPoll();
            if (phaseEl) phaseEl.classList.add("done");
            if (!s.error) {
                setTimeout(async () => {
                    closeDreamModal();
                    await loadConversation(conversationId);
                    refreshSidebar();
                }, 1500);
            }
        }
    }, 2000);
}

function stopDreamPoll() {
    if (dreamPollInterval) { clearInterval(dreamPollInterval); dreamPollInterval = null; }
}

function openUsernameEditor() {
    const display = document.getElementById("userNameDisplay");
    if (!display) return;
    const input = document.createElement("input");
    input.className = "chat-header-username-input";
    input.value = userName;
    display.replaceWith(input);
    input.focus();
    input.select();

    const commit = () => {
        const newName = input.value.trim() || userName;
        userName = newName;
        const span = document.createElement("span");
        span.className = "chat-header-username";
        span.id = "userNameDisplay";
        span.title = "Click to change";
        span.textContent = newName;
        span.addEventListener("click", openUsernameEditor);
        input.replaceWith(span);
    };
    input.addEventListener("blur", commit);
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter")  { input.blur(); }
        if (e.key === "Escape") { input.value = userName; input.blur(); }
    });
}

function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
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

function appendMessage(role, content, streaming = false, timestamp = null, messageIndex = null, msgUserName = null) {
    const messages = document.getElementById("messages");

    const empty = messages.querySelector(".empty-state");
    if (empty) empty.remove();

    const msg = document.createElement("div");
    msg.className = `message ${role}`;
    if (messageIndex !== null) msg.dataset.index = messageIndex;

    const meta = document.createElement("div");
    meta.className = "message-meta";

    const speaker = document.createElement("span");
    speaker.className = "message-speaker";
    if (role === "user") {
        speaker.textContent = msgUserName || userName;
    } else {
        speaker.textContent = "Scout";
    }

    const ts = document.createElement("span");
    ts.className = "message-timestamp";
    ts.textContent = timestamp ? formatTimestamp(timestamp) : formatTimestamp(new Date().toISOString());

    meta.appendChild(speaker);
    meta.appendChild(ts);

    if (isEditable(activeConversationStatus)) {
        const editBtn = document.createElement("button");
        editBtn.className = "message-edit-btn";
        editBtn.textContent = "Edit";
        editBtn.addEventListener("click", () => startEditMessage(msg));
        meta.appendChild(editBtn);
    }

    const bubble = document.createElement("div");
    bubble.className = "message-bubble" + (streaming ? " streaming" : "");
    bubble.textContent = content;

    msg.appendChild(meta);
    msg.appendChild(bubble);
    messages.appendChild(msg);
    messages.scrollTop = messages.scrollHeight;

    return bubble;
}

function startEditMessage(msgEl) {
    const bubble = msgEl.querySelector(".message-bubble");
    if (!bubble || bubble.querySelector("textarea")) return;

    const originalText = bubble.textContent;
    const index = parseInt(msgEl.dataset.index, 10);

    msgEl.classList.add("editing");
    bubble.innerHTML = "";

    const textarea = document.createElement("textarea");
    textarea.className = "message-edit-textarea";
    textarea.value = originalText;
    textarea.rows = Math.max(4, originalText.split("\n").length + 1);

    const actions = document.createElement("div");
    actions.className = "message-edit-actions";

    const saveBtn = document.createElement("button");
    saveBtn.textContent = "Save";
    saveBtn.className = "message-edit-save";

    const cancelBtn = document.createElement("button");
    cancelBtn.textContent = "Cancel";
    cancelBtn.className = "message-edit-cancel";

    actions.appendChild(saveBtn);
    actions.appendChild(cancelBtn);
    bubble.appendChild(textarea);
    bubble.appendChild(actions);
    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);

    const cancel = () => {
        msgEl.classList.remove("editing");
        bubble.textContent = originalText;
    };

    cancelBtn.addEventListener("click", cancel);

    saveBtn.addEventListener("click", async () => {
        const newText = textarea.value.trim();
        if (!newText) return;
        msgEl.classList.remove("editing");
        bubble.textContent = newText;
        if (!isNaN(index) && activeConversationId) {
            await apiEditMessage(activeConversationId, index, newText);
        }
    });

    textarea.addEventListener("keydown", (e) => {
        if (e.key === "Escape") { cancel(); }
    });
}

const BLOCK_SIZE = 1024;

async function updateTokenBar(messages) {
    const bar = document.getElementById("tokenBar");
    const fill = document.getElementById("tokenBarFill");
    const count = document.getElementById("tokenBarCount");

    if (!bar || !messages || messages.length === 0) {
        if (bar) bar.style.display = "none";
        return;
    }

    const promptParts = [];
    for (const msg of messages) {
        const speaker = msg.role === "user" ? (msg.user_name || "Trey") : "Scout";
        promptParts.push(`[${speaker}] ${msg.content}`);
    }
    promptParts.push("[Scout]");
    const promptText = promptParts.join("\n\n");

    const tokenCount = await apiCountTokens(promptText);
    if (tokenCount === null) {
        if (bar) bar.style.display = "none";
        return;
    }

    const pct = Math.min(tokenCount / BLOCK_SIZE, 1.0);

    bar.style.display = "flex";
    fill.style.width = (pct * 100).toFixed(1) + "%";
    fill.className = "token-bar-fill" + (pct >= 1.0 ? " danger" : pct > 0.7 ? " warn" : "");
    count.textContent = `${tokenCount.toLocaleString()} / ${BLOCK_SIZE.toLocaleString()} tokens`;

    // Auto-mark full when context hits 100%
    if (
        pct >= 1.0 &&
        activeConversationStatus === "active" &&
        activeConversationId
    ) {
        const ok = await apiSetStatus(activeConversationId, "full");
        if (ok) {
            activeConversationStatus = "full";
            await loadConversation(activeConversationId);
        }
    }
}

function renderMessages(messages, status = "active") {
    const container = document.getElementById("messages");
    container.innerHTML = "";

    if (messages.length === 0) {
        container.innerHTML = `<div class="empty-state">Start the conversation.</div>`;
        updateTokenBar([]);
        return;
    }

    for (let i = 0; i < messages.length; i++) {
        const msg = messages[i];
        appendMessage(msg.role, msg.content, false, msg.timestamp, i, msg.user_name);
    }
    updateTokenBar(messages);
}

async function loadConversation(id) {
    activeConversationId = id;
    stopDreamPoll();

    const conv = await apiGetConversation(id);
    if (!conv) return;

    activeConversationStatus = conv.status || "active";

    renderChatPanel(activeConversationStatus);
    renderMessages(conv.messages, activeConversationStatus);
    refreshSidebar();

    // Resume dream poll if training was in progress when we navigated away
    if (activeConversationStatus === "training") {
        startDreamPoll(id);
    }

    if (canSend(activeConversationStatus)) {
        document.getElementById("msgInput").focus();
    }
}

async function sendMessage() {
    if (isStreaming || !activeConversationId) return;
    if (!canSend(activeConversationStatus)) return;

    const input = document.getElementById("msgInput");
    const text = input.value.trim();
    if (!text) return;

    input.value = "";
    input.style.height = "auto";
    input.disabled = true;
    document.getElementById("sendBtn").disabled = true;
    isStreaming = true;

    const userTs = new Date().toISOString();
    appendMessage("user", text, false, userTs, null, userName);

    const bubble = appendMessage("assistant", "", true, new Date().toISOString(), null, null);

    let accumulated = "";

    try {
        const res = await fetch(`/api/chat/conversations/${activeConversationId}/message`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                conversation_id: activeConversationId,
                message: text,
                checkpoint: activeCheckpoint,
                active_modules: activeModules,
                generation: generationParams,
                user_name: userName,
            }),
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
        const conv = await apiGetConversation(activeConversationId);
        if (conv) {
            activeConversationStatus = conv.status || "active";
            renderMessages(conv.messages, activeConversationStatus);
        }
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

let activeModules = null;
let _checkpointModuleMap = {};

function updateModuleToggles(numModules) {
    const container = document.getElementById("moduleToggles");
    container.innerHTML = "";

    if (numModules <= 1) return;

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
    activeConversationStatus = "active";
    renderChatPanel("active");
    renderMessages([]);
    await refreshSidebar();
    document.getElementById("msgInput").focus();
});

await Promise.all([initCheckpointSelector(), initGenerationSettings()]);

const params = new URLSearchParams(window.location.search);
const deepLinkId = params.get("conversation");
if (deepLinkId) {
    await refreshSidebar();
    await loadConversation(deepLinkId);
} else {
    refreshSidebar();
}
