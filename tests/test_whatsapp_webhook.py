from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import get_chat_service, get_history_store, get_message_dedup, get_whatsapp_client
from app.api.routes_whatsapp import router as whatsapp_router
from app.core.config import get_settings
from app.services.dedup import MessageDedup
from app.services.history_store import ChatHistoryStore


def _build_app(*, whatsapp_client, chat_service, verify_token="test-verify-token"):
    app = FastAPI()
    app.include_router(whatsapp_router)

    def _settings_override():
        s = get_settings.__wrapped__()  # bypass lru_cache, but we won't call this directly
        return s

    # Patch get_settings used inside the route module directly since it's
    # called via a plain import, not Depends().
    import app.api.routes_whatsapp as route_module

    class _FakeSettings:
        whatsapp_verify_token = verify_token

    route_module.get_settings = lambda: _FakeSettings()

    app.dependency_overrides[get_chat_service] = lambda: chat_service
    app.dependency_overrides[get_history_store] = lambda: ChatHistoryStore()
    app.dependency_overrides[get_whatsapp_client] = lambda: whatsapp_client
    app.dependency_overrides[get_message_dedup] = lambda: MessageDedup()
    return app


def test_webhook_verification_succeeds_with_correct_token():
    app = _build_app(whatsapp_client=None, chat_service=None)
    client = TestClient(app)

    response = client.get(
        "/webhook",
        params={"hub.mode": "subscribe", "hub.verify_token": "test-verify-token", "hub.challenge": "12345"},
    )

    assert response.status_code == 200
    assert response.text == "12345"


def test_webhook_verification_fails_with_wrong_token():
    app = _build_app(whatsapp_client=None, chat_service=None)
    client = TestClient(app)

    response = client.get(
        "/webhook",
        params={"hub.mode": "subscribe", "hub.verify_token": "wrong", "hub.challenge": "12345"},
    )

    assert response.status_code == 403


def test_webhook_receive_acks_immediately_even_without_client_configured():
    app = _build_app(whatsapp_client=None, chat_service=None)
    client = TestClient(app)

    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {"id": "wamid.1", "from": "201234567890", "type": "text", "text": {"body": "hi"}}
                            ]
                        }
                    }
                ]
            }
        ]
    }
    response = client.post("/webhook", json=payload)
    assert response.status_code == 200


def test_webhook_receive_dispatches_text_message_to_chat_service():
    whatsapp_client = AsyncMock()
    # whatsapp_client.verify_signature = Mock(return_value=True)

    chat_service = AsyncMock()
    chat_service.handle_incoming_message.return_value = "reply text"

    app = _build_app(whatsapp_client=whatsapp_client, chat_service=chat_service)
    client = TestClient(app)

    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "id": "wamid.abc",
                                    "from": "201234567890",
                                    "type": "text",
                                    "text": {"body": "عايز اعرف سعر تحليل السكر"},
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }

    response = client.post("/webhook", json=payload)
    assert response.status_code == 200

    # BackgroundTasks run synchronously within TestClient's request cycle
    chat_service.handle_incoming_message.assert_awaited_once()
    call_kwargs = chat_service.handle_incoming_message.await_args.kwargs
    assert call_kwargs["question"] == "عايز اعرف سعر تحليل السكر"
    assert call_kwargs["user_id"] == "201234567890"
    whatsapp_client.send_text_message.assert_awaited_once_with("201234567890", "reply text")


def test_webhook_receive_ignores_status_updates():
    whatsapp_client = AsyncMock()
    # whatsapp_client.verify_signature = Mock(return_value=True)
    chat_service = AsyncMock()

    app = _build_app(whatsapp_client=whatsapp_client, chat_service=chat_service)
    client = TestClient(app)

    payload = {
        "entry": [{"changes": [{"value": {"statuses": [{"id": "wamid.1", "status": "delivered"}]}}]}]
    }
    response = client.post("/webhook", json=payload)
    assert response.status_code == 200
    chat_service.handle_incoming_message.assert_not_awaited()


def test_webhook_receive_rejects_invalid_signature():
    whatsapp_client = AsyncMock()
    # whatsapp_client.verify_signature = Mock(return_value=False)
    chat_service = AsyncMock()

    app = _build_app(whatsapp_client=whatsapp_client, chat_service=chat_service)
    client = TestClient(app)

    payload = {"entry": []}
    response = client.post("/webhook", json=payload, headers={"X-Hub-Signature-256": "sha256=bad"})
    assert response.status_code == 401
