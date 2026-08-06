from __future__ import annotations

"""Per-conversation menu state.

The service menu (prompts.MENU_MESSAGES) is plain numbered text -- it isn't
a real tappable UI. Without this store, a bare "1" typed in reply to that
menu had nothing to anchor it: it fell straight into the same LLM router as
any free-text question, which is what let a generic "1) branches" selection
get silently turned into a specific, unrequested branch recommendation (see
the "branches" bug -- either mis-routed to chitchat and hallucinated, or
routed to retrieval with no location filter and RAG just returned whatever
scored highest).

This store lets ChatService track, per session, "we just sent the menu and
are waiting on a raw digit" or "we just asked which area/test/disease and
are waiting on that answer" -- so those replies are handled deterministically
in code instead of being guessed at by the router LLM.

Like LanguagePreferenceStore, this is NOT long-term memory -- it's cleared
the moment the session closes (see SessionManager's `on_close` callback) or
once the corresponding sub-flow completes.

Scaling note: same caveat as SessionManager/LanguagePreferenceStore -- this
is in-process/single-worker state. If you scale to multiple workers or
replicas, move this to the same shared store (Redis) keyed the same way.
"""

# States a session can be waiting in after the menu (or a menu option) was
# sent. `None` (i.e. no entry in the store) means "not currently waiting on
# a menu-shaped reply -- treat the next message as a normal question."
AWAITING_MENU_CHOICE = "awaiting_menu_choice"
AWAITING_BRANCH_AREA = "awaiting_branch_area"
AWAITING_TEST_NAME = "awaiting_test_name"
AWAITING_DISEASE_NAME = "awaiting_disease_name"


class MenuStateStore:
    def __init__(self) -> None:
        self._state: dict[str, str] = {}

    def get(self, user_id: str) -> str | None:
        return self._state.get(user_id)

    def set(self, user_id: str, state: str) -> None:
        self._state[user_id] = state

    def clear(self, user_id: str) -> None:
        self._state.pop(user_id, None)