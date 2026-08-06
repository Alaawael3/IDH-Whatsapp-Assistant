from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    user_id: str = Field(..., min_length=1, max_length=128, description="Stable per-user ID (e.g. WhatsApp wa_id)")


class ChatResponse(BaseModel):
    answer: str


class VoiceChatResponse(BaseModel):
    answer: str
    transcribed_question: str | None = None


class DocumentChatResponse(BaseModel):
    answer: str


class ProfileResponse(BaseModel):
    national_id: str | None
    name: str | None
    birth_date: str | None
    age: str | None
    diseases: list[str]


class ForgetUserResponse(BaseModel):
    status: str
    user_id: str


class HealthResponse(BaseModel):
    status: str
    weaviate_ready: bool
