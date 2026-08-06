from __future__ import annotations

import asyncio


class ChatHistoryStore:
    """Holds each user's short-term chat_history (list of {"role","content"})
    between HTTP requests.

    The notebook passed `chat_history` around as a plain Python list kept
    alive by the caller's process/session. A stateless HTTP API has no such
    place to keep it, so this store fills that gap.

    IMPORTANT (scaling note): in-memory + single-process, same caveat as
    SessionManager. For multi-worker/multi-replica deployments, swap this
    for a Redis-backed store (e.g. a JSON blob per user_id with a TTL) --
    the interface below is intentionally small so that's a drop-in change.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[dict[str, str]]] = {}
        self._lock = asyncio.Lock()

    async def get(self, user_id: str) -> list[dict[str, str]]:
        async with self._lock:
            return list(self._store.get(user_id, []))

    async def set(self, user_id: str, chat_history: list[dict[str, str]]) -> None:
        async with self._lock:
            self._store[user_id] = list(chat_history)

    async def clear(self, user_id: str) -> None:
        async with self._lock:
            self._store.pop(user_id, None)
