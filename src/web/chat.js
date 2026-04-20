let activeConversationId = null;
let isStreaming = false;

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
    panel.innerHTML = `<div class="no-conversation" id="noConversation">Select or start a conversation.</div>`;
}

function renderChatPanel() {
    const panel = document.getElementById("chatPanel");
    panel.innerHTML = `
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

function appendMessage(role, content, streaming = false) {
    const messages = document.getElementById("messages");

    // Remove empty state if present
    const empty = messages.querySelector(".empty-state");
    if (empty) empty.remove();

    const msg = document.createElement("div");
    msg.className = `message ${role}`;

    const speaker = document.createElement("div");
    speaker.className = "message-speaker";
    speaker.textContent = role === "user" ? "Trey" : "Scout";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble" + (streaming ? " streaming" : "");
    bubble.textContent = content;

    msg.appendChild(speaker);
    msg.appendChild(bubble);
    messages.appendChild(msg);
    messages.scrollTop = messages.scrollHeight;

    return bubble;
}

function renderMessages(messages) {
    const container = document.getElementById("messages");
    container.innerHTML = "";

    if (messages.length === 0) {
        container.innerHTML = `<div class="empty-state">Start the conversation.</div>`;
        return;
    }

    for (const msg of messages) {
        appendMessage(msg.role, msg.content);
    }
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

    appendMessage("user", text);

    const bubble = appendMessage("assistant", "", true);

    let accumulated = "";

    try {
        const res = await fetch(`/api/chat/conversations/${activeConversationId}/message`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ conversation_id: activeConversationId, message: text }),
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
    }
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
const params = new URLSearchParams(window.location.search);
const deepLinkId = params.get("conversation");
if (deepLinkId) {
    await refreshSidebar();
    await loadConversation(deepLinkId);
} else {
    refreshSidebar();
}