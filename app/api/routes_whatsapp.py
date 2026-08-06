from __future__ import annotations

from enum import verify
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)

from app.api.deps import get_chat_service, get_history_store, get_message_dedup, get_whatsapp_client
from app.core.config import get_settings
from app.core.logging import get_logger
from app.services import prompts
from app.services.chat_service import ChatService
from app.services.dedup import MessageDedup
from app.services.history_store import ChatHistoryStore
from app.services.whatsapp_client import WhatsAppClient, WhatsAppError, suffix_for_mime_type
import traceback

log = get_logger(__name__)

router = APIRouter(prefix="/webhook")

# Message types we can actually act on. Anything else (location, contacts,
# stickers, reactions, unmodeled interactive replies, etc.) gets a short
# "can't handle that" reply instead of silently doing nothing.
_SUPPORTED_TYPES = {"text", "audio", "image", "document", "interactive"}

_UNSUPPORTED_TYPE_MESSAGES = {
    "ar": "معلش، مقدرش أتعامل مع النوع ده من الرسايل دلوقتي. ممكن تبعت سؤالك كتابةً؟",
    "en": "Sorry, I can't handle that type of message right now. Could you send your question as text?",
}


@router.get("")
async def verify_webhook(request: Request):
    params = request.query_params

    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    settings = get_settings()

    print("Mode:", mode)
    print("Received token:", repr(token))
    print("Expected token:", repr(settings.whatsapp_verify_token))

    if mode == "subscribe" and token == settings.whatsapp_verify_token:
        return Response(content=challenge, media_type="text/plain")

    return Response(content="Forbidden", status_code=403)


@router.post("")
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: str | None = Header(default=None),
    chat_service: ChatService = Depends(get_chat_service),
    history_store: ChatHistoryStore = Depends(get_history_store),
    whatsapp_client: WhatsAppClient | None = Depends(get_whatsapp_client),
    dedup: MessageDedup = Depends(get_message_dedup),
) -> Response:
    """Receives inbound WhatsApp events. Always returns 200 quickly (Meta
    retries aggressively on non-200 or slow responses) -- the actual reply
    is generated and sent in a background task, after this handler returns.
    """
    if whatsapp_client is None:
        # Configured to receive webhooks but no send/media credentials set --
        # ack anyway so Meta doesn't retry, but log loudly since this means
        # messages are being silently dropped.
        log.error("whatsapp_webhook_received_but_client_not_configured")
        return Response(status_code=status.HTTP_200_OK)

    raw_body = await request.body()
    # if not whatsapp_client.verify_signature(raw_body, x_hub_signature_256):
    #     raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature")

    payload = await request.json()

    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for message in value.get("messages", []):
                background_tasks.add_task(
                    _process_message,
                    message,
                    chat_service,
                    history_store,
                    whatsapp_client,
                    dedup,
                )
            # value.get("statuses") (delivery/read receipts) is intentionally
            # ignored -- nothing to reply to.

    return Response(status_code=status.HTTP_200_OK)


async def _send_reply(
    whatsapp_client: WhatsAppClient,
    chat_service: ChatService,
    wa_id: str,
    answer: str,
) -> None:
    """Sends `answer` as plain text, except at the two moments the bot is
    presenting a fixed set of choices -- the language pick and the service
    menu -- which go out as tappable WhatsApp buttons/list instead so the
    user can respond with a tap rather than typing."""
    log.info(
        "LANGUAGE_CHOICE_MESSAGE=%r",
        prompts.LANGUAGE_CHOICE_MESSAGE,
    )
    # if answer in (prompts.LANGUAGE_CHOICE_MESSAGE):
    if answer.strip() == prompts.LANGUAGE_CHOICE_MESSAGE.strip():
        await whatsapp_client.send_button_message(
            wa_id, answer, prompts.LANGUAGE_CHOICE_BUTTONS
        )
        return
    log.info(
        "LANGUAGE_CHOICE_MESSAGE=%r",
        prompts.LANGUAGE_CHOICE_MESSAGE,
    )
    if any(marker in answer for marker in prompts.MENU_MARKERS):
        lang = chat_service.language_store.get(wa_id) or "ar"
        # Keep just the greeting line as the list body -- the numbered
        # options themselves now live in the tappable rows.
        body = answer.split("\n", 1)[0]
        await whatsapp_client.send_list_message(
            wa_id,
            body,
            prompts.MENU_LIST_BUTTON_TEXT[lang],
            prompts.MENU_LIST_ROWS[lang],
        )
        return

    await whatsapp_client.send_text_message(wa_id, answer)


async def _process_message(
    message: dict[str, Any],
    chat_service: ChatService,
    history_store: ChatHistoryStore,
    whatsapp_client: WhatsAppClient,
    dedup: MessageDedup,
) -> None:
    message_id = message.get("id")
    wa_id = message.get("from")
    msg_type = message.get("type")

    if not message_id or not wa_id:
        log.warning("whatsapp_message_missing_id_or_sender", message=message)
        return

    if await dedup.seen_before(message_id):
        log.info("whatsapp_duplicate_message_skipped", message_id=message_id)
        return

    log.info("whatsapp_message_received", message_id=message_id, wa_id=wa_id, type=msg_type)

    try:
        await whatsapp_client.mark_as_read(message_id)

        if msg_type not in _SUPPORTED_TYPES:
            lang = "ar"
            await whatsapp_client.send_text_message(
                wa_id, _UNSUPPORTED_TYPE_MESSAGES.get(lang, _UNSUPPORTED_TYPE_MESSAGES["en"])
            )
            return

        chat_history = await history_store.get(wa_id)
        media_path = None

        try:
            if msg_type == "text":
                answer = await chat_service.handle_incoming_message(
                    question=message["text"]["body"],
                    chat_history=chat_history,
                    user_id=wa_id,
                )

            elif msg_type == "interactive":
                # A tap on a reply button or list row. The id we sent back
                # (e.g. "ar"/"en" for the language buttons, "1"-"4" for the
                # menu rows) is exactly what parse_language_choice /
                # parse_menu_choice already know how to read, so we just
                # feed it through as if the user had typed it.
                interactive = message.get("interactive", {})
                reply = interactive.get("button_reply") or interactive.get("list_reply")
                choice_id = reply.get("id") if reply else None
                if not choice_id:
                    log.warning("whatsapp_interactive_missing_reply", message=message)
                    await whatsapp_client.send_text_message(
                        wa_id, _UNSUPPORTED_TYPE_MESSAGES["ar"]
                    )
                    return
                answer = await chat_service.handle_incoming_message(
                    question=choice_id,
                    chat_history=chat_history,
                    user_id=wa_id,
                )

            elif msg_type == "audio":
                media = message["audio"]
                media_path = await whatsapp_client.download_media(
                    media["id"], suffix=suffix_for_mime_type(media.get("mime_type"))
                )
                answer = await chat_service.handle_incoming_message(
                    chat_history=chat_history, user_id=wa_id, audio_path=str(media_path)
                )

            else:  # image or document -- both go through the medical-report path
                media = message[msg_type]
                media_path = await whatsapp_client.download_media(
                    media["id"], suffix=suffix_for_mime_type(media.get("mime_type"))
                )
                caption = media.get("caption")
                answer = await chat_service.handle_incoming_message(
                    question=caption,
                    chat_history=chat_history,
                    user_id=wa_id,
                    document_path=str(media_path),
                )

            await history_store.set(wa_id, chat_history)
            log.info("answer=%r", answer)
            await _send_reply(whatsapp_client, chat_service, wa_id, answer)

        finally:
            if media_path is not None:
                media_path.unlink(missing_ok=True)

    except WhatsAppError as e:
        log.error("whatsapp_send_or_media_failed", message_id=message_id, error=str(e))

    except Exception as e:
        log.error(
            "whatsapp_message_processing_failed",
            message_id=message_id,
            error=repr(e),
        )

        traceback.print_exc()
    
        try:
            await whatsapp_client.send_text_message(
                wa_id,
                "معلش، حصلت مشكلة في معالجة رسالتك. ممكن تجرب تاني؟",
            )
        except WhatsAppError:
            pass