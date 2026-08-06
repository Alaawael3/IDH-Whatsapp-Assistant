from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional

from app.core.logging import get_logger

log = get_logger(__name__)

NUDGE_MESSAGES = {
    "ar": "لسه موجود؟ 🙂",
    "en": "Are you there?",
}

SESSION_CLOSED_MESSAGES = {
    "ar": "يبدو إن المحادثة اتقفلت لعدم الرد.",
    "en": "It looks like the conversation timed out.",
}

SendMessageFn = Callable[[str, str], Awaitable[None]]


class _UserSession:
    __slots__ = ("nudge_task", "close_task", "lock", "closed")

    def __init__(self) -> None:
        self.nudge_task: Optional[asyncio.Task] = None
        self.close_task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()
        self.closed = False


class SessionManager:
    """Tracks per-user inactivity after the bot sends a message.

    This is the async/FastAPI-native rewrite of the notebook's
    threading.Timer-based SessionManager -- same semantics (30s nudge / 60s
    close by default), but using asyncio tasks so it plays nicely inside a
    single Uvicorn worker's event loop.

    IMPORTANT (scaling note): like the original, this state is in-process
    only. It works correctly with a single worker/replica. If you scale to
    multiple workers or replicas, move this to a shared store (Redis +
    a scheduled job, e.g. via APScheduler+RedisJobStore or a Celery beat
    task) so the nudge/close clock is consistent no matter which worker
    handles the next request. The interface below is designed so that swap
    only touches this file.

    Flow for one user turn:
        1. User message arrives -> await disarm(user_id) FIRST.
        2. Bot computes and sends its reply.
        3. If the reply isn't the natural end of the conversation, call
           arm(user_id, chat_history, lang) to start the clock again.
    """

    def __init__(
        self,
        send_message_fn: SendMessageFn,
        nudge_seconds: int = 30,
        timeout_seconds: int = 60,
    ):
        self.send_message_fn = send_message_fn
        self.nudge_seconds = nudge_seconds
        self.timeout_seconds = timeout_seconds
        self._sessions: dict[str, _UserSession] = {}
        self._global_lock = asyncio.Lock()

    async def _get_session(self, user_id: str) -> _UserSession:
        async with self._global_lock:
            session = self._sessions.get(user_id)
            if session is None:
                session = _UserSession()
                self._sessions[user_id] = session
            return session

    async def disarm(self, user_id: str) -> None:
        """Cancel pending nudge/close tasks. Call the moment a new user
        message arrives, before doing anything else with it."""
        session = self._sessions.get(user_id)
        if session is None:
            return
        async with session.lock:
            if session.nudge_task and not session.nudge_task.done():
                session.nudge_task.cancel()
            if session.close_task and not session.close_task.done():
                session.close_task.cancel()
            session.closed = False

    async def arm(
        self,
        user_id: str,
        chat_history: list,
        lang: str,
        end_of_chat: bool = False,
        on_close: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> None:
        """Call right after the bot sends a message to the user.

        end_of_chat: True when this bot message is itself a natural close of
            the conversation (e.g. emergency handoff) -- no clock is started
            since we're not waiting on the user for anything.
        on_close: optional async callback fired with user_id when the
            session actually times out.
        """
        if end_of_chat:
            return

        session = await self._get_session(user_id)

        async with session.lock:
            if session.nudge_task and not session.nudge_task.done():
                session.nudge_task.cancel()
            if session.close_task and not session.close_task.done():
                session.close_task.cancel()
            session.closed = False

            async def _send_nudge() -> None:
                try:
                    await asyncio.sleep(self.nudge_seconds)
                    if session.closed:
                        return
                    await self.send_message_fn(
                        user_id, NUDGE_MESSAGES.get(lang, NUDGE_MESSAGES["en"])
                    )
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    log.error("session_nudge_failed", user_id=user_id, error=str(e))

            async def _close_session() -> None:
                try:
                    await asyncio.sleep(self.timeout_seconds)
                    async with session.lock:
                        if session.closed:
                            return
                        session.closed = True
                    chat_history.clear()
                    await self.send_message_fn(
                        user_id,
                        SESSION_CLOSED_MESSAGES.get(lang, SESSION_CLOSED_MESSAGES["en"]),
                    )
                    if on_close:
                        await on_close(user_id)
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    log.error("session_close_failed", user_id=user_id, error=str(e))

            session.nudge_task = asyncio.create_task(_send_nudge())
            session.close_task = asyncio.create_task(_close_session())

    async def cancel_all(self, user_id: str) -> None:
        """Fully stop tracking a user (e.g. on an explicit user-initiated end)."""
        await self.disarm(user_id)
        async with self._global_lock:
            self._sessions.pop(user_id, None)
