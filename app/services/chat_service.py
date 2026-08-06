from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from langchain_core.output_parsers import StrOutputParser

from app.core.logging import get_logger
from app.services import prompts
from app.services.documents import DocumentParseError, DocumentService
from app.services.language_store import LanguagePreferenceStore
from app.services.medical_report import MedicalReportService
from app.services.memory import MemoryService
from app.services.onboarding import OnboardingService
from app.services.retrieval import RetrievalService
from app.services.session_manager import SessionManager
from app.services.speech import SpeechService, TranscriptionError

log = get_logger(__name__)


class ChatService:
    """Single entry point for text, voice, and medical-report (PDF/image)
    messages -- the FastAPI-native rewrite of the notebook's
    `handle_incoming_message` (cell 29) + `answer_user_question` (cell 20).

    Exactly one of `question`, `audio_path`, `document_path` should be
    provided per call, matching the original contract.
    """

    def __init__(
        self,
        llm,
        memory: MemoryService,
        onboarding: OnboardingService,
        retrieval: RetrievalService,
        session_manager: SessionManager,
        medical_report: MedicalReportService,
        speech: SpeechService | None,
        documents: DocumentService | None,
        language_store: LanguagePreferenceStore,
    ):
        self.memory = memory
        self.onboarding = onboarding
        self.retrieval = retrieval
        self.session_manager = session_manager
        self.medical_report = medical_report
        self.speech = speech
        self.documents = documents
        self.language_store = language_store

        self._memory_recall_chain = (
            prompts.MEMORY_RECALL_PROMPT
            | llm
            | StrOutputParser()
        )

        self._router_chain = prompts.ROUTER_PROMPT | llm | StrOutputParser()
        self._chitchat_chain = prompts.CHITCHAT_PROMPT | llm | StrOutputParser()
        self._answer_chain = prompts.ANSWER_PROMPT | llm | StrOutputParser()

    # -- routing --------------------------------------------------------------

    def _language_gate(
        self, session_key: str, incoming_text: str | None, chat_history: list
    ) -> tuple[str | None, str | None]:
        """Returns (status, payload):

        ("ask", message)        -- language still unresolved; send `message`.
        ("resolved_now", lang)  -- this turn's message WAS the language
                                    choice; `lang` is now set for the
                                    conversation. The caller should NOT treat
                                    incoming_text as a real question.
        (None, None)            -- already resolved earlier; proceed as normal.
        """
        lang = self.language_store.get(session_key)
        if lang:
            return None, None

        if chat_history:
            # We've already sent the language question once; this message
            # is the user's answer to it.
            choice = prompts.parse_language_choice(incoming_text)
            if choice:
                self.language_store.set(session_key, choice)
                return "resolved_now", choice
            return "ask", prompts.LANGUAGE_CHOICE_MESSAGE

        return "ask", prompts.LANGUAGE_CHOICE_MESSAGE

    def _post_language_selected_response(self, user_id: str | None, lang: str) -> str:
        """What to say right after the language is picked: identity
        onboarding for a new/incomplete profile, or straight to the service
        menu for a returning user with a complete profile. Never small talk."""
        if not user_id:
            return prompts.MENU_MESSAGES[lang].format(name="")

        profile = self.memory.get_profile(user_id)
        missing = self.memory.missing_profile_fields(profile)
        if missing:
            return self.onboarding.build_onboarding_request(
                profile=profile, missing=missing, lang=lang, is_first=True
            )
        return prompts.MENU_MESSAGES[lang].format(name=profile.get("name", ""))

    def _language_reset_on_close(self):
        async def _on_close(user_id: str) -> None:
            self.language_store.clear(user_id)

        return _on_close

    # -- routing --------------------------------------------------------------

    def route_question(self, question: str, chat_history: list) -> dict:
        raw = self._router_chain.invoke({"question": question, "chat_history": chat_history})
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        log.info("Router output: %s", raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"standalone_question": question, "intent": "needs_retrieval"}

    # -- core answering (text only, after onboarding is satisfied) ------------

    def answer_user_question(
        self, question: str, chat_history: list | None, user_id: str | None
    ) -> str:
        chat_history = self.memory.get_short_term_history(chat_history)
        session_key = user_id or "anonymous"
        # Language was fixed for this conversation by the language gate in
        # handle_incoming_message; detect_lang is only a defensive fallback
        # (e.g. if this is ever called directly, bypassing the gate).
        lang = self.language_store.get(session_key) or prompts.detect_lang(question)

        profile = None
        if user_id:
            onboarding_response, profile = self.onboarding.run_onboarding_gate(
                question, user_id, lang
            )
            if onboarding_response is not None:
                return onboarding_response

            self.onboarding.remember_diseases_from_message(
                question, user_id, profile.setdefault("diseases", [])
            )

        routed = self.route_question(question, chat_history)
        intent = routed.get("intent", "needs_retrieval")
        standalone_question = routed.get("standalone_question", question)

        if intent == "emergency":
            # No memory read/write here on purpose: an acute crisis message
            # shouldn't get folded into this person's standing profile.
            return prompts.EMERGENCY_MESSAGES[lang]

        if intent == "out_of_scope":
            return prompts.OUT_OF_SCOPE_MESSAGES[lang]

        if intent == "chitchat":
            user_context = self.memory.get_user_context(question, user_id) if user_id else ""
            log.info(
                "Chitchat question=%r history=%r",
                standalone_question,
                chat_history,
            )
            return self._chitchat_chain.invoke(
                {
                    "language": lang,
                    "question": standalone_question,
                    "chat_history": chat_history,
                    "user_context_block": prompts.build_user_context_block(user_context),
                }
            )
        
        if intent == "memory_recall":
            return self._answer_memory_question(
                standalone_question,
                user_id,
            )

        # needs_retrieval
        user_context = (
            self.memory.get_user_context(standalone_question, user_id) if user_id else ""
        )
        fused_results = self.retrieval.multi_query_hybrid_retrieve(standalone_question)
        log.info("retrieved_results", fused_results=fused_results)
        context = self.retrieval.format_context(fused_results)

        return self._answer_chain.invoke(
            {
                "language": lang,
                "question": standalone_question,
                "context": context,
                "chat_history": chat_history,
                "user_context_block": prompts.build_user_context_block(user_context),
            }
        )

    def _answer_memory_question(
        self,
        question: str,
        user_id: str | None,
    ) -> str:
        if not user_id:
            return "I don't have any saved information about you yet."
        
        session_key = user_id or "anonymous"
        memories = self.memory.get_user_context(question, user_id)
        lang = self.language_store.get(session_key) or prompts.detect_lang(question)

        return self._memory_recall_chain.invoke(
            {
                "language": lang,
                "question": question,
                "memories": memories,
            }
        )

    # -- unified entry point ---------------------------------------------------

    async def handle_incoming_message(
        self,
        question: str | None = None,
        chat_history: list | None = None,
        user_id: str | None = None,
        audio_path: str | None = None,
        document_path: str | None = None,
    ) -> str:
        if chat_history is None:
            chat_history = []

        # ------------------------------------------------------------
        # Language gate -- runs before anything else, exactly once per
        # conversation. Applies uniformly to text/voice/document turns
        # since none of those should proceed until the language is set.
        # ------------------------------------------------------------
        session_key = user_id or "anonymous"
        gate_status, gate_payload = self._language_gate(session_key, question, chat_history)

        if gate_status == "ask":
            await self.session_manager.disarm(session_key)
            reply = gate_payload
            self.memory.append_short_term_turn(chat_history, question or "[new conversation]", reply)
            await self.session_manager.arm(
                session_key, chat_history, lang="ar", end_of_chat=False,
                on_close=self._language_reset_on_close(),
            )
            return reply

        if gate_status == "resolved_now":
            lang = gate_payload
            await self.session_manager.disarm(session_key)
            reply = self._post_language_selected_response(user_id, lang)
            self.memory.append_short_term_turn(chat_history, question or "[language selected]", reply)
            await self.session_manager.arm(
                session_key, chat_history, lang=lang, end_of_chat=False,
                on_close=self._language_reset_on_close(),
            )
            return reply

        # gate_status is None here -> language already resolved earlier in
        # this conversation; fall through to normal handling below.

        # ------------------------------------------------------------
        # Voice message -> transcribe to text, then fall through to the
        # normal text path below.
        # ------------------------------------------------------------
        if audio_path:
            if self.speech is None:
                raise RuntimeError("Voice messages are not enabled (ELEVENLABS_API_KEY not set).")
            try:
                question, _detected_lang = self.speech.transcribe_audio(audio_path)
            except TranscriptionError as e:
                log.warning("transcription_failed", error=str(e))
                return prompts.VOICE_TRANSCRIPTION_FAILED_MESSAGES["ar"]

        # ------------------------------------------------------------
        # Medical report (PDF/image) -> its own path, separate from the
        # question-answering RAG pipeline.
        # ------------------------------------------------------------
        if document_path:
            if self.documents is None:
                raise RuntimeError(
                    "Document uploads are not enabled (LLAMA_CLOUD_API_KEY not set)."
                )
            await self.session_manager.disarm(session_key)

            lang = self.language_store.get(session_key) or (
                prompts.detect_lang(question) if question else "ar"
            )

            if user_id:
                onboarding_response, profile = self.onboarding.run_onboarding_gate(
                    question or "", user_id, lang
                )
                if onboarding_response is not None:
                    await self.session_manager.arm(user_id, chat_history, lang, end_of_chat=False)
                    return onboarding_response
            else:
                profile = {}

            try:
                report_text = self.documents.parse_medical_document(document_path)
            except DocumentParseError as e:
                log.warning("document_parse_failed", error=str(e))
                answer = prompts.DOCUMENT_PARSE_FAILED_MESSAGES.get(
                    lang, prompts.DOCUMENT_PARSE_FAILED_MESSAGES["ar"]
                )
                self.memory.append_short_term_turn(
                    chat_history, question or "[medical report upload]", answer
                )
                if user_id:
                    await self.session_manager.arm(user_id, chat_history, lang, end_of_chat=False)
                return answer

            # Personalization block only carries the customer's own known
            # facts -- explain_medical_report decides internally whether
            # it's even allowed to use them, based on report_ownership.
            user_context_parts = []
            if profile.get("name"):
                user_context_parts.append(f"اسم العميل المسجل: {profile['name']}")
            if profile.get("age"):
                user_context_parts.append(f"سن العميل المسجل: {profile['age']}")
            if profile.get("diseases"):
                user_context_parts.append(
                    "أمراض معروفة سبق وذكرها العميل: " + ", ".join(profile["diseases"])
                )
            user_context_block = (
                "معلومات عن العميل:\n" + "\n".join(user_context_parts) + "\n\n"
                if user_context_parts
                else ""
            )

            answer, report_ownership = self.medical_report.explain_medical_report(
                report_text, question, lang, profile, user_context_block
            )

            log.info(
                "medical_report_result answer=%r ownership=%r",
                answer,
                report_ownership,
            )
            # Only fold disease mentions back into the customer's own
            # profile when we're reasonably confident the report is about
            # them.
            if user_id and question and report_ownership == "own":
                self.onboarding.remember_diseases_from_message(
                    question, user_id, profile.setdefault("diseases", [])
                )

            self.memory.append_short_term_turn(
                chat_history, question or "[medical report upload]", answer
            )
            if user_id:
                await self.session_manager.arm(user_id, chat_history, lang, end_of_chat=False)
            return answer

        # ------------------------------------------------------------
        # Plain text path
        # ------------------------------------------------------------
        if not question:
            raise ValueError(
                "handle_incoming_message requires one of `question`, `audio_path`, or `document_path`."
            )

        await self.session_manager.disarm(session_key)

        answer = self.answer_user_question(question, chat_history, user_id)
        self.memory.append_short_term_turn(chat_history, question, answer)

        lang = self.language_store.get(session_key) or prompts.detect_lang(question)
        end_of_chat = answer in (
            prompts.FORGET_CONFIRMATION_MESSAGES.get(lang),
            prompts.EMERGENCY_MESSAGES.get(lang),
        )
        await self.session_manager.arm(
            session_key, chat_history, lang, end_of_chat=end_of_chat,
            on_close=self._language_reset_on_close(),
        )

        return answer