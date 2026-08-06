# from __future__ import annotations

# import json
# from typing import Any

# from langchain_core.output_parsers import StrOutputParser

# from app.services import prompts


# def _strip_json_fence(raw: str) -> str:
#     return raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


# def _normalize_name(name: str | None) -> str:
#     return (name or "").strip().lower()


# class MedicalReportService:
#     """Explains an already-parsed medical report to the user, in-character,
#     while carefully gating whether the report is treated as the logged-in
#     customer's own data. Ported from medical_report.py.
#     """

#     def __init__(self, llm):
#         self._patient_identity_chain = prompts.PATIENT_IDENTITY_PROMPT | llm | StrOutputParser()
#         self._report_chain = prompts.MEDICAL_REPORT_PROMPT | llm | StrOutputParser()

#     def extract_patient_identity(self, report_text: str) -> dict[str, Any]:
#         raw = self._patient_identity_chain.invoke({"report_text": report_text})
#         raw = _strip_json_fence(raw)
#         try:
#             data = json.loads(raw)
#         except json.JSONDecodeError:
#             data = {}
#         return {
#             "patient_name": data.get("patient_name"),
#             "patient_age": data.get("patient_age"),
#             "patient_gender": data.get("patient_gender"),
#         }

#     @staticmethod
#     def match_report_to_profile(patient_identity: dict[str, Any], profile: dict[str, Any]) -> str:
#         """Returns one of: "own", "other", "unknown".

#         "own"     -> report identity clearly matches the logged-in customer.
#         "other"   -> report identity clearly does NOT match.
#         "unknown" -> not stated clearly enough to compare -- treated
#                      cautiously like "other" for prompting purposes.
#         """
#         report_name = _normalize_name(patient_identity.get("patient_name"))
#         report_age = patient_identity.get("patient_age")

#         profile_name = _normalize_name(profile.get("name"))
#         profile_age = profile.get("age")

#         if not report_name and report_age is None:
#             return "unknown"

#         name_matches: bool | None = None
#         if report_name and profile_name:
#             name_matches = report_name in profile_name or profile_name in report_name

#         age_matches: bool | None = None
#         if report_age is not None and profile_age is not None:
#             try:
#                 age_matches = abs(int(report_age) - int(profile_age)) <= 1  # allow off-by-one
#             except (TypeError, ValueError):
#                 age_matches = None

#         signals = [s for s in (name_matches, age_matches) if s is not None]
#         if not signals:
#             return "unknown"
#         if all(signals):
#             return "own"
#         return "other"

#     @staticmethod
#     def _format_patient_identity_summary(patient_identity: dict[str, Any]) -> str:
#         parts = []
#         if patient_identity.get("patient_name"):
#             parts.append(f"الاسم: {patient_identity['patient_name']}")
#         if patient_identity.get("patient_age") is not None:
#             parts.append(f"السن: {patient_identity['patient_age']}")
#         if patient_identity.get("patient_gender"):
#             parts.append(f"الجنس: {patient_identity['patient_gender']}")
#         return "، ".join(parts) if parts else "غير مذكورة بوضوح في التقرير"

#     def explain_medical_report(
#         self,
#         report_text: str,
#         question: str | None,
#         lang: str,
#         profile: dict[str, Any],
#         user_context_block: str = "",
#     ) -> tuple[str, str]:
#         """Returns (answer, report_ownership) so the caller can decide
#         whether to fold anything from this turn back into the customer's own
#         long-term profile (only makes sense when report_ownership == "own")."""
#         patient_identity = self.extract_patient_identity(report_text)
#         report_ownership = self.match_report_to_profile(patient_identity, profile)

#         effective_question = (
#             question.strip()
#             if question and question.strip()
#             else prompts.NO_QUESTION_PLACEHOLDER.get(lang, prompts.NO_QUESTION_PLACEHOLDER["ar"])
#         )

#         answer = self._report_chain.invoke(
#             {
#                 "report_ownership": report_ownership,
#                 "patient_identity_summary": self._format_patient_identity_summary(patient_identity),
#                 "report_text": report_text,
#                 "question": effective_question,
#                 "user_context_block": user_context_block,
#             }
#         )

#         return answer, report_ownership



from __future__ import annotations

import json
from typing import Any

from langchain_core.output_parsers import StrOutputParser

from app.services import prompts


def _strip_json_fence(raw: str) -> str:
    return raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


def _normalize_name(name: str | None) -> str:
    return (name or "").strip().lower()


class MedicalReportService:
    """Explains an already-parsed medical report to the user, in-character,
    while carefully gating whether the report is treated as the logged-in
    customer's own data. Ported from medical_report.py.
    """

    def __init__(self, llm):
        self._patient_identity_chain = prompts.PATIENT_IDENTITY_PROMPT | llm | StrOutputParser()
        self._report_chain = prompts.MEDICAL_REPORT_PROMPT | llm | StrOutputParser()

    def extract_patient_identity(self, report_text: str) -> dict[str, Any]:
        raw = self._patient_identity_chain.invoke({"report_text": report_text})
        raw = _strip_json_fence(raw)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        return {
            "patient_name": data.get("patient_name"),
            "patient_age": data.get("patient_age"),
            "patient_gender": data.get("patient_gender"),
        }

    @staticmethod
    def match_report_to_profile(patient_identity: dict[str, Any], profile: dict[str, Any]) -> str:
        """Returns one of: "own", "other", "unknown".

        "own"     -> report identity clearly matches the logged-in customer.
        "other"   -> report identity clearly does NOT match.
        "unknown" -> not stated clearly enough to compare -- treated
                     cautiously like "other" for prompting purposes.
        """
        report_name = _normalize_name(patient_identity.get("patient_name"))
        report_age = patient_identity.get("patient_age")

        profile_name = _normalize_name(profile.get("name"))
        profile_age = profile.get("age")

        if not report_name and report_age is None:
            return "unknown"

        name_matches: bool | None = None
        if report_name and profile_name:
            name_matches = report_name in profile_name or profile_name in report_name

        age_matches: bool | None = None
        if report_age is not None and profile_age is not None:
            try:
                age_matches = abs(int(report_age) - int(profile_age)) <= 1  # allow off-by-one
            except (TypeError, ValueError):
                age_matches = None

        signals = [s for s in (name_matches, age_matches) if s is not None]
        if not signals:
            return "unknown"
        if all(signals):
            return "own"
        return "other"

    @staticmethod
    def _format_patient_identity_summary(patient_identity: dict[str, Any]) -> str:
        parts = []
        if patient_identity.get("patient_name"):
            parts.append(f"الاسم: {patient_identity['patient_name']}")
        if patient_identity.get("patient_age") is not None:
            parts.append(f"السن: {patient_identity['patient_age']}")
        if patient_identity.get("patient_gender"):
            parts.append(f"الجنس: {patient_identity['patient_gender']}")
        return "، ".join(parts) if parts else "غير مذكورة بوضوح في التقرير"

    def explain_medical_report(
        self,
        report_text: str,
        question: str | None,
        lang: str,
        profile: dict[str, Any],
        user_context_block: str = "",
    ) -> tuple[str, str]:
        """Returns (answer, report_ownership) so the caller can decide
        whether to fold anything from this turn back into the customer's own
        long-term profile (only makes sense when report_ownership == "own")."""
        patient_identity = self.extract_patient_identity(report_text)
        report_ownership = self.match_report_to_profile(patient_identity, profile)

        effective_question = (
            question.strip()
            if question and question.strip()
            else prompts.NO_QUESTION_PLACEHOLDER.get(lang, prompts.NO_QUESTION_PLACEHOLDER["ar"])
        )

        answer = self._report_chain.invoke(
            {
                "report_ownership": report_ownership,
                "patient_identity_summary": self._format_patient_identity_summary(patient_identity),
                "report_text": report_text,
                "question": effective_question,
                "user_context_block": user_context_block,
                "language_instruction": prompts.language_instruction(lang),
            }
        )

        return answer, report_ownership