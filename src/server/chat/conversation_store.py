"""
chat/conversation_store.py

Persists chat conversations as JSON files under data/conversations/.
Each conversation is a single JSON file with metadata and messages.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import config

CONVERSATIONS_DIR = config.DATA_PATH / "conversations"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conv_path(conversation_id: str) -> Path:
    return CONVERSATIONS_DIR / f"{conversation_id}.json"


def list_conversations() -> list[dict]:
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    convs = []
    for path in sorted(CONVERSATIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            convs.append({
                "id": data["id"],
                "title": data.get("title", "Untitled"),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "message_count": len(data.get("messages", [])),
            })
        except Exception:
            continue
    return convs


def create_conversation() -> dict:
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    conv_id = str(uuid.uuid4())
    now = _now()
    conv = {
        "id": conv_id,
        "title": "New conversation",
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    _conv_path(conv_id).write_text(json.dumps(conv, indent=2), encoding="utf-8")
    return conv


def get_conversation(conversation_id: str) -> dict | None:
    path = _conv_path(conversation_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def append_message(conversation_id: str, role: str, content: str) -> dict:
    conv = get_conversation(conversation_id)
    if conv is None:
        raise ValueError(f"Conversation not found: {conversation_id}")

    message = {"role": role, "content": content, "timestamp": _now()}
    conv["messages"].append(message)
    conv["updated_at"] = _now()

    # Auto-title from first user message
    if conv["title"] == "New conversation" and role == "user":
        conv["title"] = content[:60] + ("…" if len(content) > 60 else "")

    _conv_path(conversation_id).write_text(json.dumps(conv, indent=2), encoding="utf-8")
    return message


def rename_conversation(conversation_id: str, title: str) -> dict | None:
    conv = get_conversation(conversation_id)
    if conv is None:
        return None
    conv["title"] = title.strip() or "Untitled"
    conv["updated_at"] = _now()
    _conv_path(conversation_id).write_text(json.dumps(conv, indent=2), encoding="utf-8")
    return conv


def delete_conversation(conversation_id: str) -> bool:
    path = _conv_path(conversation_id)
    if path.exists():
        path.unlink()
        return True
    return False