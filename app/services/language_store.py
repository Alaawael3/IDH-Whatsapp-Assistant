from __future__ import annotations

"""Per-conversation language preference.

This is intentionally NOT long-term memory (not saved to mem0). It only
tracks which language the user picked for the *current* conversation, and
is cleared the moment that conversation's session closes (see
SessionManager's `on_close` callback), so the next conversation asks again.

Scaling note: same caveat as SessionManager -- this is in-process/single
worker state. If you scale to multiple workers/replicas, move this to the
same shared store (Redis) you'd use for SessionManager, keyed the same way.
"""


class LanguagePreferenceStore:
    def __init__(self) -> None:
        self._prefs: dict[str, str] = {}

    def get(self, user_id: str) -> str | None:
        return self._prefs.get(user_id)

    def set(self, user_id: str, lang: str) -> None:
        self._prefs[user_id] = lang

    def clear(self, user_id: str) -> None:
        self._prefs.pop(user_id, None)
