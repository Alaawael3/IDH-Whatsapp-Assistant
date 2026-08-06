from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request, status

from app.core.config import Settings, get_settings
from app.services.chat_service import ChatService
from app.services.history_store import ChatHistoryStore
from app.services.memory import MemoryService


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def get_memory_service(request: Request) -> MemoryService:
    return request.app.state.memory_service


def get_history_store(request: Request) -> ChatHistoryStore:
    return request.app.state.history_store


def get_whatsapp_client(request: Request):
    """Returns None if WhatsApp isn't configured -- callers must check."""
    return getattr(request.app.state, "whatsapp_client", None)


def get_message_dedup(request: Request):
    return request.app.state.message_dedup


async def verify_api_key(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """No-op if API_KEY isn't configured (e.g. local dev). Once set, every
    /api/* request must include a matching X-API-Key header. For production,
    prefer this behind a proper gateway/WAF with per-client keys, but this
    gives you a working baseline out of the box."""
    if not settings.api_key:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
