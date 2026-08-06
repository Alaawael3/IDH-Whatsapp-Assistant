from __future__ import annotations

import asyncio
from collections import OrderedDict


class MessageDedup:
    """Tracks recently-seen WhatsApp message IDs (`wamid`s) so a webhook
    retry (Meta redelivers if it doesn't get a fast 200) doesn't produce a
    duplicate reply.

    In-memory, bounded to `max_size` (oldest evicted first) -- fine for a
    single instance. If you move to multiple instances/workers, back this
    with Redis (SETNX + short TTL) instead so dedup is shared across them.
    """

    def __init__(self, max_size: int = 5000):
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._max_size = max_size
        self._lock = asyncio.Lock()

    async def seen_before(self, message_id: str) -> bool:
        """Returns True if this message_id was already processed (and
        records it as seen either way, so a caller can rely on a single
        call to both check and mark)."""
        async with self._lock:
            if message_id in self._seen:
                self._seen.move_to_end(message_id)
                return True
            self._seen[message_id] = None
            if len(self._seen) > self._max_size:
                self._seen.popitem(last=False)
            return False
