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
    checkpoint: Optional[str] = None
    active_modules: Optional[list[int]] = None
    generation: Optional[GenerationParams] = None
    user_name: Optional[str] = None


class RenameConversationRequest(BaseModel):
    title: str


class EditMessageRequest(BaseModel):
    content: str


class SetStatusRequest(BaseModel):
    status: str