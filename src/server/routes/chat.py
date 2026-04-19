import json
import sys
import torch
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

import config
from chat.conversation_store import (
    list_conversations,
    create_conversation,
    get_conversation,
    append_message,
    delete_conversation,
)
from .models.chat_models import ChatMessageRequest

api_router = APIRouter(prefix="/api/chat")
view_router = APIRouter(prefix="/chat")

# Model is loaded once and reused across requests
_model = None
_tokenizer = None


def _get_model_and_tokenizer():
    global _model, _tokenizer
    if _model is None:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from model.loader import load_model
        from ai_clients.tokenizer import load_tokenizer

        checkpoint = config.CHECKPOINT_DIR / "latest.pt"
        if not checkpoint.exists():
            raise RuntimeError("No checkpoint found at data/checkpoints/latest.pt")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = load_model(checkpoint, device)
        _tokenizer = load_tokenizer()

    return _model, _tokenizer


def _format_prompt(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        speaker = "Trey" if msg["role"] == "user" else "Scout"
        parts.append(f"[{speaker}] {msg['content']}")
    parts.append("[Scout]")
    return "\n\n".join(parts)


def _stream_response(conversation_id: str, prompt: str):
    """Generator that streams tokens via SSE and persists the full response."""
    from cli_repl import stream_generate

    model, tokenizer = _get_model_and_tokenizer()
    device = next(model.parameters()).device

    stop_sequences = ["[Trey]", "[Scout]"]
    # Hold back this many chars so a stop sequence straddling two yields is
    # never emitted before we can detect it.
    max_stop_len = max(len(s) for s in stop_sequences)

    emitted = ""
    buffer = ""

    for piece in stream_generate(model, tokenizer, prompt, device):
        buffer += piece

        # Check for any complete stop sequence in the buffer
        stop_idx = None
        for stop in stop_sequences:
            idx = buffer.find(stop)
            if idx != -1:
                if stop_idx is None or idx < stop_idx:
                    stop_idx = idx

        if stop_idx is not None:
            safe = buffer[:stop_idx]
            delta = safe[len(emitted):]
            if delta:
                emitted += delta
                yield f"data: {json.dumps({'token': delta})}\n\n"
            break

        # Emit everything except the trailing holdback window
        safe = buffer[:-max_stop_len] if len(buffer) > max_stop_len else ""
        if safe:
            delta = safe[len(emitted):]
            if delta:
                emitted += delta
                yield f"data: {json.dumps({'token': delta})}\n\n"

    else:
        # Stream ended without hitting a stop sequence — flush the buffer
        delta = buffer[len(emitted):]
        if delta:
            emitted += delta
            yield f"data: {json.dumps({'token': delta})}\n\n"

    full_response = emitted.strip()
    if full_response:
        append_message(conversation_id, "assistant", full_response)

    yield f"data: {json.dumps({'done': True})}\n\n"


@view_router.get("/")
def index():
    return FileResponse(config.WEB_DIR / "chat.html")


@api_router.get("/conversations")
def get_conversations():
    return list_conversations()


@api_router.post("/conversations")
def new_conversation():
    return create_conversation()


@api_router.get("/conversations/{conversation_id}")
def get_conversation_detail(conversation_id: str):
    conv = get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@api_router.delete("/conversations/{conversation_id}")
def delete_conversation_endpoint(conversation_id: str):
    if not delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "ok"}


@api_router.post("/conversations/{conversation_id}/message")
def send_message(conversation_id: str, req: ChatMessageRequest):
    conv = get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    append_message(conversation_id, "user", req.message)
    conv = get_conversation(conversation_id)

    try:
        _get_model_and_tokenizer()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    prompt = _format_prompt(conv["messages"])

    return StreamingResponse(
        _stream_response(conversation_id, prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )