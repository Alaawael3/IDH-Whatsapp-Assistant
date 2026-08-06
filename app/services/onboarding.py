# from __future__ import annotations

# import json
# from typing import Any

# from langchain_core.output_parsers import StrOutputParser

# from app.core.logging import get_logger
# from app.services import prompts
# from app.services.memory import (
#     DISEASE_LABEL,
#     PROFILE_LABELS,
#     MemoryService,
#     compute_age,
#     normalize_birth_date,
#     validate_birth_date,
#     validate_national_id,
# )

# log = get_logger(__name__)


# def _strip_json_fence(raw: str) -> str:
#     return raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


# class OnboardingService:
#     """Gates every interaction until name / birth_date / national_id are on
#     file, then extracts and validates each field as it's provided across
#     turns. Ported from onboarding.py with identical prompts and validation
#     rules.
#     """

#     def __init__(self, llm, memory_service: MemoryService):
#         self.memory = memory_service
#         self._profile_extraction_chain = (
#             prompts.PROFILE_EXTRACTION_PROMPT | llm | StrOutputParser()
#         )
#         self._disease_extraction_chain = (
#             prompts.DISEASE_EXTRACTION_PROMPT | llm | StrOutputParser()
#         )
#         self._onboarding_request_chain = (
#             prompts.ONBOARDING_REQUEST_PROMPT | llm | StrOutputParser()
#         )

#     # -- extraction ---------------------------------------------------------

#     def extract_profile_fields(self, message: str) -> dict[str, Any]:
#         raw = self._profile_extraction_chain.invoke({"message": message})
#         raw = _strip_json_fence(raw)
#         try:
#             data = json.loads(raw)
#         except json.JSONDecodeError:
#             data = {}
#         return {
#             "national_id": data.get("national_id"),
#             "name": data.get("name"),
#             "birth_date": data.get("birth_date"),
#             "refusing": bool(data.get("refusing", False)),
#         }

#     def extract_diseases(self, message: str) -> list[str]:
#         raw = self._disease_extraction_chain.invoke({"message": message})
#         raw = _strip_json_fence(raw)
#         try:
#             data = json.loads(raw)
#             return [d for d in (data.get("diseases", []) or []) if d]
#         except json.JSONDecodeError:
#             return []

#     def remember_diseases_from_message(
#         self, message: str, user_id: str, known_diseases: list[str]
#     ) -> None:
#         for disease in self.extract_diseases(message):
#             if disease not in known_diseases:
#                 self.memory.save_fact(user_id, DISEASE_LABEL, disease)
#                 known_diseases.append(disease)

#     # -- onboarding request message ------------------------------------------

#     def build_onboarding_request(
#         self, profile: dict[str, Any], missing: list[str], lang: str, is_first: bool
#     ) -> str:
#         if lang == "en":
#             known_parts = []
#             if profile.get("name"):
#                 known_parts.append(f"name = {profile['name']}")
#             if profile.get("birth_date"):
#                 known_parts.append(f"date of birth = {profile['birth_date']}")
#             if profile.get("national_id"):
#                 known_parts.append("national ID = provided")
#             known_summary = ", ".join(known_parts) if known_parts else "nothing"
#             missing_summary = ", ".join(prompts.FIELD_NAMES_EN[f] for f in missing)
#         else:
#             known_parts = []
#             if profile.get("name"):
#                 known_parts.append(f"الاسم = {profile['name']}")
#             if profile.get("birth_date"):
#                 known_parts.append(f"تاريخ الميلاد = {profile['birth_date']}")
#             if profile.get("national_id"):
#                 known_parts.append("الرقم القومي = موجود")
#             known_summary = "، ".join(known_parts) if known_parts else "مفيش معلومات معروفة"
#             missing_summary = "، ".join(prompts.FIELD_NAMES_AR[f] for f in missing)

#         try:
#             message = self._onboarding_request_chain.invoke(
#                 {
#                     "known_summary": known_summary,
#                     "missing_summary": missing_summary,
#                     "is_first": "yes" if is_first else "no",
#                 }
#             ).strip()
#             if message:
#                 return message
#         except Exception as e:
#             log.error("onboarding_request_generation_failed", error=str(e))

#         return self._fallback_onboarding_request(lang, missing, is_first)

#     @staticmethod
#     def _fallback_onboarding_request(lang: str, missing: list[str], is_first: bool) -> str:
#         if lang == "en":
#             names = [prompts.FIELD_NAMES_EN[f] for f in missing]
#             if len(names) == 1:
#                 return f"Could you please provide your {names[0]}?"
#             return (
#                 "Could you please provide your " + ", ".join(names[:-1]) + " and " + names[-1] + "?"
#             )

#         if len(missing) == 1:
#             field = missing[0]
#             if field == "national_id":
#                 return "تمام، ممكن أعرف الرقم القومي؟"
#             if field == "name":
#                 return "تمام، ممكن أعرف اسمك؟"
#             if field == "birth_date":
#                 return "تمام، ممكن أعرف تاريخ ميلادك؟"

#         if is_first:
#             return "أهلاً بيك! عشان أقدر أساعدك، ممكن أعرف اسمك وتاريخ ميلادك والرقم القومي؟"

#         return "تمام، ممكن أعرف " + "، ".join(prompts.FIELD_NAMES_AR[f] for f in missing) + "؟"

#     # -- the gate -------------------------------------------------------------

#     def run_onboarding_gate(
#         self, question: str, user_id: str, lang: str
#     ) -> tuple[str | None, dict[str, Any]]:
#         """Returns (response_or_None, profile).

#         response is None once name/birth_date/national_id are all on file --
#         in that case the caller should proceed to normal routing. Otherwise
#         response is the message to send back to the user right now (an
#         onboarding prompt, a validation error, or the menu after the profile
#         first completes).
#         """
#         profile = self.memory.get_profile(user_id)

#         was_empty = not any(
#             [profile.get("name"), profile.get("birth_date"), profile.get("national_id")]
#         )

#         missing_before = self.memory.missing_profile_fields(profile)
#         if not missing_before:
#             return None, profile

#         extracted = self.extract_profile_fields(question)
#         invalid_fields: list[str] = []

#         if extracted.get("name"):
#             self.memory.save_fact(user_id, PROFILE_LABELS["name"], str(extracted["name"]))
#             profile["name"] = str(extracted["name"])

#         if extracted.get("national_id") not in (None, ""):
#             is_valid, _reason = validate_national_id(extracted["national_id"])
#             if is_valid:
#                 self.memory.save_fact(
#                     user_id, PROFILE_LABELS["national_id"], str(extracted["national_id"])
#                 )
#                 profile["national_id"] = str(extracted["national_id"])
#             else:
#                 invalid_fields.append("national_id")

#         if extracted.get("birth_date") not in (None, ""):
#             is_valid, parsed_date, _reason = validate_birth_date(extracted["birth_date"])
#             if is_valid and parsed_date is not None:
#                 normalized = normalize_birth_date(parsed_date)
#                 self.memory.save_fact(user_id, PROFILE_LABELS["birth_date"], normalized)
#                 profile["birth_date"] = normalized

#                 derived_age = compute_age(parsed_date)
#                 self.memory.save_fact(user_id, PROFILE_LABELS["age"], str(derived_age))
#                 profile["age"] = str(derived_age)
#             else:
#                 invalid_fields.append("birth_date")

#         missing_after = self.memory.missing_profile_fields(profile)

#         if invalid_fields:
#             error_lines = [prompts.INVALID_FIELD_MESSAGES[lang][f] for f in invalid_fields]
#             return " ".join(error_lines), profile

#         if missing_after:
#             if extracted.get("refusing"):
#                 return prompts.APOLOGY_MESSAGES[lang], profile
#             response = self.build_onboarding_request(
#                 profile=profile, missing=missing_after, lang=lang, is_first=was_empty
#             )
#             return response, profile

#         menu = prompts.MENU_MESSAGES[lang].format(name=profile.get("name", ""))
#         return menu, profile


from __future__ import annotations

import json
from typing import Any

from langchain_core.output_parsers import StrOutputParser

from app.core.logging import get_logger
from app.services import prompts
from app.services.memory import (
    DISEASE_LABEL,
    PROFILE_LABELS,
    MemoryService,
    compute_age,
    normalize_birth_date,
    validate_birth_date,
    validate_national_id,
)

log = get_logger(__name__)


def _strip_json_fence(raw: str) -> str:
    return raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


class OnboardingService:
    """Gates every interaction until name / birth_date / national_id are on
    file, then extracts and validates each field as it's provided across
    turns. Ported from onboarding.py with identical prompts and validation
    rules.
    """

    def __init__(self, llm, memory_service: MemoryService):
        self.memory = memory_service
        self._profile_extraction_chain = (
            prompts.PROFILE_EXTRACTION_PROMPT | llm | StrOutputParser()
        )
        self._disease_extraction_chain = (
            prompts.DISEASE_EXTRACTION_PROMPT | llm | StrOutputParser()
        )
        self._onboarding_request_chain = (
            prompts.ONBOARDING_REQUEST_PROMPT | llm | StrOutputParser()
        )

    # -- extraction ---------------------------------------------------------

    def extract_profile_fields(self, message: str) -> dict[str, Any]:
        raw = self._profile_extraction_chain.invoke({"message": message})
        raw = _strip_json_fence(raw)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        return {
            "national_id": data.get("national_id"),
            "name": data.get("name"),
            "birth_date": data.get("birth_date"),
            "refusing": bool(data.get("refusing", False)),
        }

    def extract_diseases(self, message: str) -> list[str]:
        raw = self._disease_extraction_chain.invoke({"message": message})
        raw = _strip_json_fence(raw)
        try:
            data = json.loads(raw)
            return [d for d in (data.get("diseases", []) or []) if d]
        except json.JSONDecodeError:
            return []

    def remember_diseases_from_message(
        self, message: str, user_id: str, known_diseases: list[str]
    ) -> None:
        for disease in self.extract_diseases(message):
            if disease not in known_diseases:
                self.memory.save_fact(user_id, DISEASE_LABEL, disease)
                known_diseases.append(disease)

    # -- onboarding request message ------------------------------------------

    def build_onboarding_request(
        self, profile: dict[str, Any], missing: list[str], lang: str, is_first: bool
    ) -> str:
        if lang == "en":
            known_parts = []
            if profile.get("name"):
                known_parts.append(f"name = {profile['name']}")
            if profile.get("birth_date"):
                known_parts.append(f"date of birth = {profile['birth_date']}")
            if profile.get("national_id"):
                known_parts.append("national ID = provided")
            known_summary = ", ".join(known_parts) if known_parts else "nothing"
            missing_summary = ", ".join(prompts.FIELD_NAMES_EN[f] for f in missing)
        else:
            known_parts = []
            if profile.get("name"):
                known_parts.append(f"الاسم = {profile['name']}")
            if profile.get("birth_date"):
                known_parts.append(f"تاريخ الميلاد = {profile['birth_date']}")
            if profile.get("national_id"):
                known_parts.append("الرقم القومي = موجود")
            known_summary = "، ".join(known_parts) if known_parts else "مفيش معلومات معروفة"
            missing_summary = "، ".join(prompts.FIELD_NAMES_AR[f] for f in missing)

        try:
            message = self._onboarding_request_chain.invoke(
                {
                    "known_summary": known_summary,
                    "missing_summary": missing_summary,
                    "is_first": "yes" if is_first else "no",
                    "language_instruction": prompts.language_instruction(lang),
                }
            ).strip()
            if message:
                return message
        except Exception as e:
            log.error("onboarding_request_generation_failed", error=str(e))

        return self._fallback_onboarding_request(lang, missing, is_first)

    @staticmethod
    def _fallback_onboarding_request(lang: str, missing: list[str], is_first: bool) -> str:
        if lang == "en":
            names = [prompts.FIELD_NAMES_EN[f] for f in missing]
            if len(names) == 1:
                return f"Could you please provide your {names[0]}?"
            return (
                "Could you please provide your " + ", ".join(names[:-1]) + " and " + names[-1] + "?"
            )

        if len(missing) == 1:
            field = missing[0]
            if field == "national_id":
                return "تمام، ممكن أعرف الرقم القومي؟"
            if field == "name":
                return "تمام، ممكن أعرف اسمك؟"
            if field == "birth_date":
                return "تمام، ممكن أعرف تاريخ ميلادك؟"

        if is_first:
            return "أهلاً بيك! عشان أقدر أساعدك، ممكن أعرف اسمك وتاريخ ميلادك والرقم القومي؟"

        return "تمام، ممكن أعرف " + "، ".join(prompts.FIELD_NAMES_AR[f] for f in missing) + "؟"

    # -- the gate -------------------------------------------------------------

    def run_onboarding_gate(
        self, question: str, user_id: str, lang: str
    ) -> tuple[str | None, dict[str, Any]]:
        """Returns (response_or_None, profile).

        response is None once name/birth_date/national_id are all on file --
        in that case the caller should proceed to normal routing. Otherwise
        response is the message to send back to the user right now (an
        onboarding prompt, a validation error, or the menu after the profile
        first completes).
        """
        profile = self.memory.get_profile(user_id)

        was_empty = not any(
            [profile.get("name"), profile.get("birth_date"), profile.get("national_id")]
        )

        missing_before = self.memory.missing_profile_fields(profile)
        if not missing_before:
            return None, profile

        extracted = self.extract_profile_fields(question)
        invalid_fields: list[str] = []

        if extracted.get("name"):
            self.memory.save_fact(user_id, PROFILE_LABELS["name"], str(extracted["name"]))
            profile["name"] = str(extracted["name"])

        if extracted.get("national_id") not in (None, ""):
            is_valid, _reason = validate_national_id(extracted["national_id"])
            if is_valid:
                self.memory.save_fact(
                    user_id, PROFILE_LABELS["national_id"], str(extracted["national_id"])
                )
                profile["national_id"] = str(extracted["national_id"])
            else:
                invalid_fields.append("national_id")

        if extracted.get("birth_date") not in (None, ""):
            is_valid, parsed_date, reason = validate_birth_date(extracted["birth_date"])

            if is_valid and parsed_date is not None:
                normalized = normalize_birth_date(parsed_date)
                self.memory.save_fact(user_id, PROFILE_LABELS["birth_date"], normalized)
                profile["birth_date"] = normalized

                derived_age = compute_age(parsed_date)
                self.memory.save_fact(user_id, PROFILE_LABELS["age"], str(derived_age))
                profile["age"] = str(derived_age)

            else:
                if reason == "birth date cannot be in the future":
                    return (
                        "تاريخ الميلاد لا يمكن أن يكون في المستقبل. من فضلك أدخل تاريخ ميلاد صحيح.",
                        profile,
                    )

                if reason == "birth date implies an unrealistic age":
                    return (
                        "تاريخ الميلاد غير صحيح. من فضلك أدخل تاريخ ميلاد حقيقي.",
                        profile,
                    )

                return (
                    "صيغة تاريخ الميلاد غير صحيحة. من فضلك أدخله بالشكل 14-05-1998.",
                    profile,
                )

        missing_after = self.memory.missing_profile_fields(profile)

        if invalid_fields:
            error_lines = [prompts.INVALID_FIELD_MESSAGES[lang][f] for f in invalid_fields]
            return " ".join(error_lines), profile

        if missing_after:
            if extracted.get("refusing"):
                return prompts.APOLOGY_MESSAGES[lang], profile
            response = self.build_onboarding_request(
                profile=profile, missing=missing_after, lang=lang, is_first=was_empty
            )
            return response, profile

        menu = prompts.MENU_MESSAGES[lang].format(name=profile.get("name", ""))
        return menu, profile