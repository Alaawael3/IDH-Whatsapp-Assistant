"""Tests the onboarding gate logic with a fake LLM/memory so no live Groq or
mem0 calls happen. Focuses on the state-machine behavior: which field gets
asked for, validation-failure messaging, and the completion menu.
"""

from __future__ import annotations

from app.services.onboarding import OnboardingService


class _FakeChain:
    """Mimics `prompt | llm | StrOutputParser()` -- .invoke() returns a
    canned string based on what the test wants extracted."""

    def __init__(self, response: str):
        self.response = response

    def invoke(self, _inputs):
        return self.response


class _FakeMemoryService:
    def __init__(self, profile: dict):
        self._profile = profile
        self.saved_facts: list[tuple[str, str, str]] = []

    def get_profile(self, user_id: str) -> dict:
        return dict(self._profile)

    def missing_profile_fields(self, profile: dict) -> list[str]:
        return [f for f in ("name", "birth_date", "national_id") if not profile.get(f)]

    def save_fact(self, user_id: str, label: str, value: str) -> None:
        self.saved_facts.append((user_id, label, value))


def _make_service(profile_extraction_json: str, profile: dict) -> tuple[OnboardingService, _FakeMemoryService]:
    memory = _FakeMemoryService(profile)
    svc = OnboardingService.__new__(OnboardingService)  # bypass __init__ (no real LLM)
    svc.memory = memory
    svc._profile_extraction_chain = _FakeChain(profile_extraction_json)
    svc._disease_extraction_chain = _FakeChain('{"diseases": []}')
    svc._onboarding_request_chain = _FakeChain("")  # force fallback path
    return svc, memory


def test_gate_asks_for_missing_fields_when_profile_empty():
    svc, memory = _make_service(
        profile_extraction_json='{"national_id": null, "name": null, "birth_date": null, "refusing": false}',
        profile={"national_id": None, "name": None, "birth_date": None, "age": None, "diseases": []},
    )
    response, profile = svc.run_onboarding_gate("hi", "user1", "ar")
    assert response is not None
    assert "الاسم" in response or "اسمك" in response


def test_gate_saves_valid_national_id_and_asks_for_rest():
    svc, memory = _make_service(
        profile_extraction_json='{"national_id": "29001011234567", "name": null, "birth_date": null, "refusing": false}',
        profile={"national_id": None, "name": None, "birth_date": None, "age": None, "diseases": []},
    )
    response, profile = svc.run_onboarding_gate("رقمي القومي 29001011234567", "user1", "ar")
    assert profile["national_id"] == "29001011234567"
    assert ("National ID", "29001011234567") not in [(m[1], m[2]) for m in memory.saved_facts if False]
    assert any(m[1] == "National ID" and m[2] == "29001011234567" for m in memory.saved_facts)
    assert response is not None  # still missing name/birth_date


def test_gate_rejects_invalid_national_id_without_saving():
    svc, memory = _make_service(
        profile_extraction_json='{"national_id": "123", "name": null, "birth_date": null, "refusing": false}',
        profile={"national_id": None, "name": None, "birth_date": None, "age": None, "diseases": []},
    )
    response, profile = svc.run_onboarding_gate("رقمي 123", "user1", "ar")
    assert profile["national_id"] is None
    assert not any(m[1] == "National ID" for m in memory.saved_facts)
    assert "14" in response  # error message mentions the 14-digit rule


def test_gate_returns_none_and_profile_when_already_complete():
    svc, memory = _make_service(
        profile_extraction_json="{}",
        profile={
            "national_id": "29001011234567",
            "name": "Ahmed",
            "birth_date": "1998-05-14",
            "age": "27",
            "diseases": [],
        },
    )
    response, profile = svc.run_onboarding_gate("عايز اعرف سعر تحليل", "user1", "ar")
    assert response is None
    assert profile["name"] == "Ahmed"


def test_gate_apology_when_user_refuses():
    svc, memory = _make_service(
        profile_extraction_json='{"national_id": null, "name": null, "birth_date": null, "refusing": true}',
        profile={"national_id": None, "name": None, "birth_date": None, "age": None, "diseases": []},
    )
    response, profile = svc.run_onboarding_gate("مش هقولك", "user1", "ar")
    assert response is not None
    assert "فاهم" in response or "understand" in response.lower()
