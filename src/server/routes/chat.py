import json
import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

import config
from chat.conversation_store import (
    list_conversations,
    create_conversation,
    get_conversation,
    append_message,
    rename_conversation,
    delete_conversation,
    update_message,
    set_conversation_status,
)
from .models.chat_models import (
    ChatMessageRequest, RenameConversationRequest, GenerationParams,
    EditMessageRequest, SetStatusRequest,
)

# Active dream cycle jobs keyed by conversation_id
_dream_jobs: dict = {}

api_router = APIRouter(prefix="/api/chat")
view_router = APIRouter(prefix="/chat")

# Model cache — keyed by checkpoint filename so swapping is lazy
_model_cache: dict = {}
_tokenizer = None

# Module-count cache — avoids repeated torch.load on every page load
_module_count_cache: dict = {}


def _parse_step_from_filename(filename: str) -> tuple[int, int]:
    """Extract (step, phase) from filenames like model_40000.pt or model_p1_200.pt."""
    import re
    m = re.match(r"model_p(\d+)_(\d+)\.pt$", filename)
    if m:
        return int(m.group(2)), int(m.group(1))
    m = re.match(r"model_(\d+)\.pt$", filename)
    if m:
        return int(m.group(1)), 0
    return 0, 0


def _read_metadata() -> dict:
    import json
    meta_path = config.CHECKPOINT_DIR / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {}


def _count_modules_in_checkpoint(filename: str) -> int:
    """Return module count for a checkpoint, using in-memory cache to avoid repeated loads."""
    global _module_count_cache
    if filename in _module_count_cache:
        return _module_count_cache[filename]

    from model.loader import count_modules_in_state
    import torch as _torch

    path = config.CHECKPOINT_DIR / filename
    raw = _torch.load(path, map_location="cpu", weights_only=False)
    state = raw.get("model", raw)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    count = count_modules_in_state(state) or 1
    _module_count_cache[filename] = count
    return count


def _list_checkpoints():
    """Return checkpoint metadata sorted by step descending — no torch.load."""
    ckpt_dir = config.CHECKPOINT_DIR
    meta = _read_metadata()
    checkpoints = []

    for name in ("latest.pt", "phase1_start.pt", "scout_v0_step40k.pt", "scout_phase1_start.pt", "scout_phase1_end.pt"):
        path = ckpt_dir / name
        if not path.exists():
            continue
        info = meta.get(name, {})
        step = info.get("step")
        phase = info.get("phase", 0)
        if step is None and name == "scout_v0_step40k.pt":
            step = 40000
        elif step is None and name == "scout_phase1_start.pt":
            step = 5400
        elif step is None and name == "scout_phase1_end.pt":
            step = 4664
        num_modules = info.get("num_modules") or _count_modules_in_checkpoint(name)
        label = _checkpoint_label(name, step, phase)
        checkpoints.append({"filename": name, "step": step, "phase": phase,
                             "num_modules": num_modules, "label": label})

    numbered = []
    for path in ckpt_dir.glob("model_*.pt"):
        step, phase_from_name = _parse_step_from_filename(path.name)
        info = meta.get(path.name, {})
        phase = info.get("phase", phase_from_name)
        # Infer module count from phase in filename — avoids torch.load on every file.
        # Phase 0 files have 1 module; phase 1+ files have phase+1 modules.
        num_modules = info.get("num_modules") or (phase_from_name + 1) or _count_modules_in_checkpoint(path.name)
        label = f"step {step:,}" + (f" · phase {phase}" if phase else "")
        numbered.append({"filename": path.name, "step": step, "phase": phase,
                         "num_modules": num_modules, "label": label})

    numbered.sort(key=lambda c: (c["phase"], c["step"]), reverse=True)
    checkpoints.extend(numbered)

    return checkpoints


def _checkpoint_label(filename: str, step: int | None, phase: int) -> str:
    step_str = f"step {step:,}" if step is not None else "unknown step"
    if filename == "latest.pt":
        return f"latest ({step_str})"
    if filename == "phase1_start.pt":
        return f"phase 1 start ({step_str})"
    if filename == "scout_v0_step40k.pt":
        return f"📌 v0 · step 40,000 (pre-reset)"
    if filename == "scout_phase1_start.pt":
        return f"📌 phase 1 start · step 5,400 (reset point)"
    if filename == "scout_phase1_end.pt":
        return f"📌 phase 1 end · step 4,664 (current)"
    return step_str + (f" · phase {phase}" if phase else "")


def _get_model_and_tokenizer(checkpoint_filename: str = "latest.pt"):
    global _model_cache, _tokenizer

    ckpt_path = config.CHECKPOINT_DIR / checkpoint_filename
    if not ckpt_path.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint_filename}")

    if checkpoint_filename not in _model_cache:
        from model.loader import load_model
        from ai_clients.tokenizer import load_tokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model_cache[checkpoint_filename] = load_model(ckpt_path, device)

        if _tokenizer is None:
            _tokenizer = load_tokenizer()

    return _model_cache[checkpoint_filename], _tokenizer


def _format_prompt(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        if msg["role"] == "user":
            speaker = msg.get("user_name") or "Trey"
        else:
            speaker = "Scout"
        parts.append(f"[{speaker}] {msg['content']}")
    parts.append("[Scout]")
    return "\n\n".join(parts)


def _active_modules_to_skip(model, active_modules: list[int] | None) -> set[int] | None:
    """Convert an active-modules list to a skip set for model.forward()."""
    if active_modules is None:
        return None
    total = len(model.expert_modules)
    all_indices = set(range(total))
    skip = all_indices - set(active_modules)
    return skip if skip else None


def _stream_response(conversation_id: str, prompt: str, checkpoint_filename: str = "latest.pt",
                     generation: GenerationParams = None, active_modules: list[int] = None,
                     generation_log: dict = None):
    """Generator that streams tokens via SSE and persists the full response."""
    from cli_repl import stream_generate

    model, tokenizer = _get_model_and_tokenizer(checkpoint_filename)
    device = next(model.parameters()).device

    gen = generation or GenerationParams()
    gen_kwargs = {
        "temperature": gen.temperature,
        "top_k": gen.vocabulary,
        "rep_penalty": gen.rep_penalty,
        "max_new_tokens": gen.max_new_tokens,
        "skip_modules": _active_modules_to_skip(model, active_modules),
    }

    # Resolve active_modules to actual list for logging (default = all modules)
    resolved_active = active_modules if active_modules is not None else list(range(len(model.expert_modules)))

    stop_sequences = ["[Trey]", "[Scout]"]
    # Hold back this many chars so a stop sequence straddling two yields is
    # never emitted before we can detect it.
    max_stop_len = max(len(s) for s in stop_sequences)

    emitted = ""
    buffer = ""

    for piece in stream_generate(model, tokenizer, prompt, device, **gen_kwargs):
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
        append_message(conversation_id, "assistant", full_response,
                       checkpoint=checkpoint_filename, active_modules=resolved_active,
                       generation=generation_log)

    yield f"data: {json.dumps({'done': True})}\n\n"


@view_router.get("/")
def index():
    return FileResponse(config.WEB_DIR / "chat.html")


@api_router.get("/generation-defaults")
def generation_defaults():
    return {
        "temperature": config.TEMPERATURE,
        "vocabulary": config.TOP_K,
        "rep_penalty": config.REP_PENALTY,
        "max_new_tokens": config.MAX_NEW_TOKENS,
    }


@api_router.get("/checkpoints")
def list_checkpoints():
    return _list_checkpoints()


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


@api_router.patch("/conversations/{conversation_id}")
def rename_conversation_endpoint(conversation_id: str, req: RenameConversationRequest):
    conv = rename_conversation(conversation_id, req.title)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"id": conv["id"], "title": conv["title"]}


@api_router.delete("/conversations/{conversation_id}")
def delete_conversation_endpoint(conversation_id: str):
    result = delete_conversation(conversation_id)
    if not result:
        conv = get_conversation(conversation_id)
        if conv and conv.get("status") == "locked":
            raise HTTPException(status_code=403, detail="Conversation is locked")
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "ok"}


@api_router.post("/conversations/{conversation_id}/message")
def send_message(conversation_id: str, req: ChatMessageRequest):
    conv = get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    status = conv.get("status", "active")
    if status in ("locked", "training"):
        raise HTTPException(status_code=403, detail=f"Conversation is {status}")

    checkpoint_filename = req.checkpoint or "latest.pt"

    try:
        model, _ = _get_model_and_tokenizer(checkpoint_filename)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Resolve active_modules: default = all modules in the checkpoint
    num_modules = len(model.expert_modules)
    active_modules = req.active_modules if req.active_modules is not None else list(range(num_modules))

    # Build a resolved generation log with effective values (filling in config defaults)
    gen = req.generation or GenerationParams()
    generation_log = {
        "temperature": gen.temperature if gen.temperature is not None else config.TEMPERATURE,
        "vocabulary": gen.vocabulary if gen.vocabulary is not None else config.TOP_K,
        "rep_penalty": gen.rep_penalty if gen.rep_penalty is not None else config.REP_PENALTY,
        "max_new_tokens": gen.max_new_tokens if gen.max_new_tokens is not None else config.MAX_NEW_TOKENS,
    }

    append_message(conversation_id, "user", req.message,
                   checkpoint=checkpoint_filename, active_modules=active_modules,
                   generation=generation_log, user_name=req.user_name)
    conv = get_conversation(conversation_id)

    prompt = _format_prompt(conv["messages"])

    return StreamingResponse(
        _stream_response(conversation_id, prompt, checkpoint_filename, req.generation, active_modules, generation_log),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@api_router.patch("/conversations/{conversation_id}/messages/{message_index}")
def edit_message(conversation_id: str, message_index: int, req: EditMessageRequest):
    conv = get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conv.get("status") == "locked":
        raise HTTPException(status_code=403, detail="Conversation is locked")
    msg = update_message(conversation_id, message_index, req.content)
    if msg is None:
        raise HTTPException(status_code=404, detail="Message not found")
    return msg


@api_router.patch("/conversations/{conversation_id}/status")
def set_status(conversation_id: str, req: SetStatusRequest):
    conv = get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    current = conv.get("status", "active")
    if current == "locked":
        raise HTTPException(status_code=403, detail="Conversation is locked")
    # Only allow manual transitions: active ↔ full
    if req.status not in ("active", "full"):
        raise HTTPException(status_code=400, detail="Manual status must be 'active' or 'full'")
    updated = set_conversation_status(conversation_id, req.status)
    return {"id": conversation_id, "status": updated.get("status")}


@api_router.post("/conversations/{conversation_id}/dream")
def start_dream_cycle(conversation_id: str):
    conv = get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    status = conv.get("status", "active")
    if status == "locked":
        raise HTTPException(status_code=403, detail="Conversation is already locked")
    if status == "training":
        raise HTTPException(status_code=409, detail="Dream cycle already running")

    from train.dream_cycle import DreamCycleJob

    # Mark as training immediately so UI reflects it
    set_conversation_status(conversation_id, "training")

    job = DreamCycleJob(conversation_id=conversation_id)
    _dream_jobs[conversation_id] = job
    job.start()

    return {"status": "started"}


@api_router.get("/conversations/{conversation_id}/dream")
def dream_status(conversation_id: str):
    job = _dream_jobs.get(conversation_id)
    if job is None:
        return {"running": False, "completed": False, "phase": "none", "progress": 0}
    return job.status()