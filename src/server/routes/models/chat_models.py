from pydantic import BaseModel


class ChatMessageRequest(BaseModel):
    conversation_id: str
    message: str


class RenameConversationRequest(BaseModel):
    title: str