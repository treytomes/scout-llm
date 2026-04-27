from pydantic import BaseModel
from typing import Optional


class GenerationParams(BaseModel):
    temperature: Optional[float] = None
    vocabulary: Optional[int] = None   # top-k; None = use config default
    rep_penalty: Optional[float] = None
    max_new_tokens: Optional[int] = None


class ChatMessageRequest(BaseModel):
    conversation_id: str
    message: str
    checkpoint: Optional[str] = None          # filename e.g. "latest.pt"
    active_modules: Optional[list[int]] = None  # e.g. [0] to bypass module 1
    generation: Optional[GenerationParams] = None


class RenameConversationRequest(BaseModel):
    title: str