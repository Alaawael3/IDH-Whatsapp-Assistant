# from __future__ import annotations

# import hashlib
# import hmac
# import tempfile
# from pathlib import Path

# import httpx
# from tenacity import retry, stop_after_attempt, wait_exponential

# from app.core.config import Settings
# from app.core.logging import get_logger

# log = get_logger(__name__)



# _MAX_MEDIA_BYTES = 20 * 1024 * 1024  # WhatsApp's own media size ceiling is 16-100MB by type; 20MB is a safe app-level cap


# class WhatsAppError(Exception):
#     """Raised on any failed call to the WhatsApp Cloud API (send or media)."""


# class WhatsAppClient:
#     """Thin wrapper around the Meta WhatsApp Cloud API: send text messages,
#     download inbound media (voice notes, images, PDFs), and verify webhook
#     signatures.

#     Requires WHATSAPP_ACCESS_TOKEN + WHATSAPP_PHONE_NUMBER_ID. Construction
#     raises if either is missing -- the caller (main.py) only builds this when
#     both are configured, same pattern as SpeechService/DocumentService.
#     """

#     def __init__(self, settings: Settings):
#         if not settings.whatsapp_access_token or not settings.whatsapp_phone_number_id:
#             raise RuntimeError(
#                 "WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_NUMBER_ID must both be set "
#                 "to use the WhatsApp integration."
#             )
#         self._token = settings.whatsapp_access_token
#         self._phone_number_id = settings.whatsapp_phone_number_id
#         # self._app_secret = settings.whatsapp_app_secret
#         self._base_url = f"https://graph.facebook.com/{settings.whatsapp_api_version}"
#         self._client = httpx.AsyncClient(
#             headers={"Authorization": f"Bearer {self._token}"}, timeout=20
#         )

#     async def aclose(self) -> None:
#         await self._client.aclose()

#     # -- outbound -------------------------------------------------------------

#     @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
#     async def send_text_message(self, to: str, text: str) -> None:
#         """Sends a plain text message to a WhatsApp user (`to` = their wa_id,
#         e.g. "201234567890", no leading '+')."""
#         url = f"{self._base_url}/{self._phone_number_id}/messages"
#         payload = {
#                "messaging_product": "whatsapp",
#                "recipient_type": "individual",
#                "to": to,
#                "type": "text",
#                "text": {
#                    "preview_url": False,
#                    "body": text,
#                },
#            }
#         response = await self._client.post(url, json=payload)
#         if response.status_code >= 400:
#             raise WhatsAppError(
#                 f"send_text_message failed ({response.status_code}): {response.text}"
#             )

#     async def mark_as_read(self, message_id: str) -> None:
#         """Best-effort read receipt. Failures here shouldn't block a reply,
#         so this swallows errors rather than raising."""
#         url = f"{self._base_url}/{self._phone_number_id}/messages"
#         payload = {
#             "messaging_product": "whatsapp",
#             "status": "read",
#             "message_id": message_id,
#         }
#         try:
#             await self._client.post(url, json=payload)
#         except Exception as e:
#             log.warning("mark_as_read_failed", message_id=message_id, error=str(e))

#     # -- inbound media ----------------------------------------------------------

#     async def download_media(self, media_id: str, suffix: str = "") -> Path:
#         """Downloads a WhatsApp media object (voice note, image, document) to
#         a temp file and returns its path. Caller is responsible for deleting
#         the file when done (see routes_whatsapp.py's use of it).

#         Two-step Cloud API flow: GET /{media_id} to resolve a short-lived
#         download URL, then GET that URL (still needs the same auth header).
#         """
#         meta_url = f"{self._base_url}/{media_id}"
#         meta_response = await self._client.get(meta_url)
#         if meta_response.status_code >= 400:
#             raise WhatsAppError(
#                 f"media metadata lookup failed ({meta_response.status_code}): {meta_response.text}"
#             )
#         media_url = meta_response.json().get("url")
#         if not media_url:
#             raise WhatsAppError(f"media metadata response had no url: {meta_response.text}")

#         tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
#         try:
#             async with self._client.stream("GET", media_url) as stream:
#                 if stream.status_code >= 400:
#                     raise WhatsAppError(f"media download failed ({stream.status_code})")
#                 size = 0
#                 async for chunk in stream.aiter_bytes(1024 * 1024):
#                     size += len(chunk)
#                     if size > _MAX_MEDIA_BYTES:
#                         raise WhatsAppError("media exceeds size limit")
#                     tmp.write(chunk)
#         finally:
#             tmp.close()

#         return Path(tmp.name)

#     # -- security ---------------------------------------------------------------

#     def verify_signature(self, raw_body: bytes, signature_header: str | None) -> bool:
#         """Verifies the X-Hub-Signature-256 header Meta sends on every
#         webhook POST. Returns True (and logs a warning) if WHATSAPP_APP_SECRET
#         isn't configured, so local dev via ngrok isn't blocked -- set the
#         secret before going to production.
#         """
#         # if not self._app_secret:
#         #     log.warning("whatsapp_signature_verification_skipped", reason="WHATSAPP_APP_SECRET not set")
#         #     return True
#         if not signature_header or not signature_header.startswith("sha256="):
#             return False

#         expected = hmac.new(self._app_secret.encode(), raw_body, hashlib.sha256).hexdigest()
#         provided = signature_header.removeprefix("sha256=")
#         return hmac.compare_digest(expected, provided)


# _MIME_SUFFIX = {
#     "audio/ogg": ".ogg",
#     "audio/ogg; codecs=opus": ".ogg",
#     "audio/mpeg": ".mp3",
#     "audio/mp4": ".m4a",
#     "audio/amr": ".amr",
#     "image/jpeg": ".jpg",
#     "image/png": ".png",
#     "application/pdf": ".pdf",
# }


# def suffix_for_mime_type(mime_type: str | None) -> str:
#     if not mime_type:
#         return ""
#     return _MIME_SUFFIX.get(mime_type.strip(), "")


from __future__ import annotations

import hashlib
import hmac
import tempfile
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)



_MAX_MEDIA_BYTES = 20 * 1024 * 1024  # WhatsApp's own media size ceiling is 16-100MB by type; 20MB is a safe app-level cap


class WhatsAppError(Exception):
    """Raised on any failed call to the WhatsApp Cloud API (send or media)."""


class WhatsAppClient:
    """Thin wrapper around the Meta WhatsApp Cloud API: send text messages,
    download inbound media (voice notes, images, PDFs), and verify webhook
    signatures.

    Requires WHATSAPP_ACCESS_TOKEN + WHATSAPP_PHONE_NUMBER_ID. Construction
    raises if either is missing -- the caller (main.py) only builds this when
    both are configured, same pattern as SpeechService/DocumentService.
    """

    def __init__(self, settings: Settings):
        if not settings.whatsapp_access_token or not settings.whatsapp_phone_number_id:
            raise RuntimeError(
                "WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_NUMBER_ID must both be set "
                "to use the WhatsApp integration."
            )
        self._token = settings.whatsapp_access_token
        self._phone_number_id = settings.whatsapp_phone_number_id
        # self._app_secret = settings.whatsapp_app_secret
        self._base_url = f"https://graph.facebook.com/{settings.whatsapp_api_version}"
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self._token}"}, timeout=20
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- outbound -------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def send_text_message(self, to: str, text: str) -> None:
        """Sends a plain text message to a WhatsApp user (`to` = their wa_id,
        e.g. "201234567890", no leading '+')."""
        url = f"{self._base_url}/{self._phone_number_id}/messages"
        payload = {
               "messaging_product": "whatsapp",
               "recipient_type": "individual",
               "to": to,
               "type": "text",
               "text": {
                   "preview_url": False,
                   "body": text,
               },
           }
        response = await self._client.post(url, json=payload)
        if response.status_code >= 400:
            raise WhatsAppError(
                f"send_text_message failed ({response.status_code}): {response.text}"
            )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def send_button_message(
        self, to: str, body_text: str, buttons: list[tuple[str, str]]
    ) -> None:
        """Sends an interactive reply-buttons message. `buttons` is a list of
        (id, title) pairs -- WhatsApp allows 1-3 reply buttons, each title
        capped at 20 characters. The `id` comes back verbatim in the
        `button_reply` of the user's tap, so callers should use ids their
        own parsing logic already understands (e.g. "ar"/"en")."""
        log.info("body_text=%r", body_text)

        if not 1 <= len(buttons) <= 3:
            raise ValueError("WhatsApp reply-button messages support 1-3 buttons")
        url = f"{self._base_url}/{self._phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {"text": body_text},
                "action": {
                    "buttons": [
                        {"type": "reply", "reply": {"id": btn_id, "title": title[:20]}}
                        for btn_id, title in buttons
                    ]
                },
            },
        }
        log.info(payload)
        response = await self._client.post(url, json=payload)
        if response.status_code >= 400:
            log.info(
                "body=%r buttons=%r",
                body_text,
                buttons,
            )
            raise WhatsAppError(
                f"send_button_message failed ({response.status_code}): {response.text}"
            )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def send_list_message(
        self,
        to: str,
        body_text: str,
        button_text: str,
        rows: list[tuple[str, str, str | None]],
        section_title: str | None = None,
    ) -> None:
        """Sends an interactive list message. `rows` is a list of
        (id, title, description) tuples -- WhatsApp allows 1-10 rows, titles
        capped at 24 characters and descriptions at 72. Like
        `send_button_message`, the `id` comes back verbatim in the user's
        `list_reply`."""
        if not 1 <= len(rows) <= 10:
            raise ValueError("WhatsApp list messages support 1-10 rows")
        section: dict = {
            "rows": [
                {
                    "id": row_id,
                    "title": title[:24],
                    **({"description": desc[:72]} if desc else {}),
                }
                for row_id, title, desc in rows
            ]
        }
        if section_title:
            section["title"] = section_title[:24]

        url = f"{self._base_url}/{self._phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "interactive",
            "interactive": {
                "type": "list",
                "body": {"text": body_text},
                "action": {
                    "button": button_text[:20],
                    "sections": [section],
                },
            },
        }
        response = await self._client.post(url, json=payload)
        log.info(payload)
        if response.status_code >= 400:
            raise WhatsAppError(
                f"send_list_message failed ({response.status_code}): {response.text}"
            )

    async def mark_as_read(self, message_id: str) -> None:
        """Best-effort read receipt. Failures here shouldn't block a reply,
        so this swallows errors rather than raising."""
        url = f"{self._base_url}/{self._phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
        }
        try:
            await self._client.post(url, json=payload)
        except Exception as e:
            log.warning("mark_as_read_failed", message_id=message_id, error=str(e))

    # -- inbound media ----------------------------------------------------------

    async def download_media(self, media_id: str, suffix: str = "") -> Path:
        """Downloads a WhatsApp media object (voice note, image, document) to
        a temp file and returns its path. Caller is responsible for deleting
        the file when done (see routes_whatsapp.py's use of it).

        Two-step Cloud API flow: GET /{media_id} to resolve a short-lived
        download URL, then GET that URL (still needs the same auth header).
        """
        meta_url = f"{self._base_url}/{media_id}"
        meta_response = await self._client.get(meta_url)
        if meta_response.status_code >= 400:
            raise WhatsAppError(
                f"media metadata lookup failed ({meta_response.status_code}): {meta_response.text}"
            )
        media_url = meta_response.json().get("url")
        if not media_url:
            raise WhatsAppError(f"media metadata response had no url: {meta_response.text}")

        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            async with self._client.stream("GET", media_url) as stream:
                if stream.status_code >= 400:
                    raise WhatsAppError(f"media download failed ({stream.status_code})")
                size = 0
                async for chunk in stream.aiter_bytes(1024 * 1024):
                    size += len(chunk)
                    if size > _MAX_MEDIA_BYTES:
                        raise WhatsAppError("media exceeds size limit")
                    tmp.write(chunk)
        finally:
            tmp.close()

        return Path(tmp.name)

    # -- security ---------------------------------------------------------------

    def verify_signature(self, raw_body: bytes, signature_header: str | None) -> bool:
        """Verifies the X-Hub-Signature-256 header Meta sends on every
        webhook POST. Returns True (and logs a warning) if WHATSAPP_APP_SECRET
        isn't configured, so local dev via ngrok isn't blocked -- set the
        secret before going to production.
        """
        # if not self._app_secret:
        #     log.warning("whatsapp_signature_verification_skipped", reason="WHATSAPP_APP_SECRET not set")
        #     return True
        if not signature_header or not signature_header.startswith("sha256="):
            return False

        expected = hmac.new(self._app_secret.encode(), raw_body, hashlib.sha256).hexdigest()
        provided = signature_header.removeprefix("sha256=")
        return hmac.compare_digest(expected, provided)


_MIME_SUFFIX = {
    "audio/ogg": ".ogg",
    "audio/ogg; codecs=opus": ".ogg",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/amr": ".amr",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "application/pdf": ".pdf",
}


def suffix_for_mime_type(mime_type: str | None) -> str:
    if not mime_type:
        return ""
    return _MIME_SUFFIX.get(mime_type.strip(), "")