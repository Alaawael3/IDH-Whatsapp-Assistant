from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from mem0 import Memory

from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)

# --- What we're allowed to keep long-term -----------------------------------
# IMPORTANT: long-term (mem0) storage is restricted to exactly these fields.
# Raw Q&A turns are never written to mem0 -- that's short-term memory only
# (see get_short_term_history / append_short_term_turn below).
#
# We never ask the user for their age directly. We ask for birth date,
# validate it, store it, and derive `age` from it automatically. Both
# `birth_date` and the derived `age` are stored as separate facts.
PROFILE_LABELS: dict[str, str] = {
    "national_id": "National ID",
    "name": "Name",
    "birth_date": "Birth Date",
    "age": "Age",
}
DISEASE_LABEL = "Disease"

# `age` is intentionally excluded -- it's derived from birth_date, never
# collected from the user directly.
REQUIRED_PROFILE_FIELDS = ("name", "birth_date", "national_id")

SHORT_TERM_MAX_MESSAGES = 5

_NATIONAL_ID_LENGTH = 14
_ALL_ZERO_NATIONAL_ID = "0" * _NATIONAL_ID_LENGTH

_BIRTH_DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y")

_MIN_AGE = 0
_MAX_AGE = 120


# ============================================================
# VALIDATORS -- pure functions, unchanged from the notebook
# ============================================================


def validate_national_id(value: str | None) -> tuple[bool, str | None]:
    """A valid Egyptian-style national ID: exactly 14 numeric digits, not all
    zeros. Returns (is_valid, error_reason); error_reason is None when valid.
    """
    if value is None:
        return False, "missing"

    candidate = str(value).strip()

    if not re.fullmatch(r"\d+", candidate):
        return False, "must contain digits only"

    if len(candidate) != _NATIONAL_ID_LENGTH:
        return False, f"must be exactly {_NATIONAL_ID_LENGTH} digits"

    if candidate == _ALL_ZERO_NATIONAL_ID:
        return False, "cannot be all zeros"

    return True, None


def validate_birth_date(value: str | None) -> tuple[bool, date | None, str | None]:
    """Parses a birth date from common formats and checks it's a real,
    plausible birth date. Returns (is_valid, parsed_date_or_None, error_reason).
    """
    if value is None:
        return False, None, "missing"

    candidate = str(value).strip()
    parsed: date | None = None

    for fmt in _BIRTH_DATE_FORMATS:
        try:
            parsed = datetime.strptime(candidate, fmt).date()
            break
        except ValueError:
            continue

    if parsed is None:
        return False, None, "invalid date format"

    today = date.today()
    if parsed > today:
        return False, None, "birth date cannot be in the future"

    age = compute_age(parsed)
    if age < _MIN_AGE or age > _MAX_AGE:
        return False, None, "birth date implies an unrealistic age"

    return True, parsed, None


def compute_age(birth_date: date) -> int:
    """Whole-years age as of today."""
    today = date.today()
    had_birthday_this_year = (today.month, today.day) >= (birth_date.month, birth_date.day)
    return today.year - birth_date.year - (0 if had_birthday_this_year else 1)


def normalize_birth_date(parsed: date) -> str:
    """Canonical storage format so birth_date round-trips cleanly."""
    return parsed.strftime("%Y-%m-%d")


# ============================================================
# MEMORY SERVICE
# ============================================================


class MemoryService:
    """Wraps mem0 (long-term profile/disease facts) + an in-process
    short-term chat-history helper.

    One instance is created at app startup (see app.core.dependencies) and
    reused across requests -- constructing `Memory` is expensive.
    """

    def __init__(self, settings: Settings):
        vector_store_config: dict[str, Any]
        if settings.qdrant_url:
            vector_store_config = {
                "provider": "qdrant",
                "config": {
                    "url": settings.qdrant_url,
                    "api_key": settings.qdrant_api_key,
                    "collection_name": settings.mem0_collection_name,
                    "embedding_model_dims": settings.mem0_embedding_dims,
                },
            }
        else:
            # Local/dev fallback: embedded Qdrant, no server required.
            vector_store_config = {
                "provider": "qdrant",
                "config": {
                    "path": settings.qdrant_local_path,
                    "collection_name": settings.mem0_collection_name,
                    "embedding_model_dims": settings.mem0_embedding_dims,
                },
            }

        self._mem0_config = {
            "vector_store": vector_store_config,
            "llm": {"provider": "groq", "config": {"model": settings.mem0_llm_model,"api_key": settings.groq_api_key,}},
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": settings.mem0_embedding_model,
                    "embedding_dims": settings.mem0_embedding_dims,
                    "model_kwargs": {"device": "cpu"},
                },
            },
        }
        self.settings = settings
        self.memory = Memory.from_config(self._mem0_config)

    # -- internal -------------------------------------------------------

    @staticmethod
    def _extract_memory_texts(results: Any) -> list[str]:
        items = results.get("results", []) if isinstance(results, dict) else results
        return [item["memory"] if isinstance(item, dict) else str(item) for item in items]

    # -- long-term profile / disease facts -------------------------------

    def get_user_context(
        self, question: str, user_id: str, limit: int | None = None
    ) -> str:
        """Relevant memories for this question, formatted for the LLM prompt.
        Returns "" (never raises) if nothing relevant is stored yet."""
        limit = limit or self.settings.mem0_max_memories_in_prompt
        try:
            results = self.memory.search(
                question,
                filters={"user_id": user_id},
                top_k=limit,
                threshold=self.settings.mem0_search_threshold,
            )
        except Exception:
            log.warning("mem0_search_failed", user_id=user_id)
            return ""

        facts = self._extract_memory_texts(results)
        if not facts:
            return ""
        return "\n".join(f"- {f}" for f in facts)

    def get_all_user_memories(self, user_id: str, limit: int = 50) -> list[str]:
        try:
            results = self.memory.get_all(filters={"user_id": user_id}, top_k=limit)
            return self._extract_memory_texts(results)
        except Exception:
            log.warning("mem0_get_all_failed", user_id=user_id)
            return []

    def save_fact(self, user_id: str, label: str, value: str) -> None:
        """Write a single structured fact, e.g. save_fact(uid, "Name", "Ahmed")
        -> stores "Name: Ahmed". infer=False: we already know exactly what
        this fact is; mem0 shouldn't run its own extraction on it."""
        try:
            self.memory.add(
                [{"role": "user", "content": f"{label}: {value}"}],
                user_id=user_id,
                infer=False,
            )
        except Exception as e:
            log.error("mem0_save_fact_failed", user_id=user_id, label=label, error=str(e))

    def get_profile(self, user_id: str) -> dict[str, Any]:
        """Reconstruct the user's profile from saved facts. If `age` wasn't
        stored directly but `birth_date` was, age is derived on the fly."""
        profile: dict[str, Any] = {
            "national_id": None,
            "name": None,
            "birth_date": None,
            "age": None,
            "diseases": [],
        }
        for mem_text in self.get_all_user_memories(user_id, limit=100):
            stripped = mem_text.strip()
            for field, label in PROFILE_LABELS.items():
                prefix = f"{label}:"
                if stripped.lower().startswith(prefix.lower()):
                    profile[field] = stripped.split(":", 1)[1].strip()
            if stripped.lower().startswith(f"{DISEASE_LABEL}:".lower()):
                disease_value = stripped.split(":", 1)[1].strip()
                if disease_value and disease_value not in profile["diseases"]:
                    profile["diseases"].append(disease_value)

        if not profile.get("age") and profile.get("birth_date"):
            is_valid, parsed, _ = validate_birth_date(profile["birth_date"])
            if is_valid and parsed is not None:
                profile["age"] = str(compute_age(parsed))

        return profile

    @staticmethod
    def missing_profile_fields(profile: dict[str, Any]) -> list[str]:
        return [f for f in REQUIRED_PROFILE_FIELDS if not profile.get(f)]

    def forget_user(self, user_id: str) -> None:
        """Erase everything stored about one user (profile + diseases)."""
        self.memory.delete_all(user_id=user_id)

    def list_user_memories(self, user_id: str, limit: int = 50) -> list[str]:
        return self.get_all_user_memories(user_id, limit=limit)

    # -- short-term (in-request) chat history -----------------------------

    @staticmethod
    def get_short_term_history(
        chat_history: list | None, max_messages: int = SHORT_TERM_MAX_MESSAGES
    ) -> list:
        if not chat_history:
            return []
        return chat_history[-max_messages:]

    @staticmethod
    def append_short_term_turn(
        chat_history: list,
        question: str,
        answer: str,
        max_messages: int = SHORT_TERM_MAX_MESSAGES,
    ) -> list:
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})
        if len(chat_history) > max_messages:
            del chat_history[:-max_messages]
        return chat_history
