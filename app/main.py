from __future__ import annotations

import os

# Must run before anything below imports `mem0` (mem0 reads this into a
# module-level constant on first import, so setting it later -- e.g. from
# Settings inside lifespan() -- is too late). Silences mem0's PostHog
# telemetry client, including the "Multiple active PostHog clients
# detected" warning that shows up on every `--reload` restart.
os.environ.setdefault("MEM0_TELEMETRY", "False")

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from app.api.routes_chat import router as chat_router  # noqa: E402
from app.api.routes_health import router as health_router  # noqa: E402
from app.api.routes_whatsapp import router as whatsapp_router  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.services.language_store import LanguagePreferenceStore
from app.services.menu_store import MenuStateStore
from app.services.chat_service import ChatService  # noqa: E402
from app.services.dedup import MessageDedup  # noqa: E402
from app.services.documents import DocumentService  # noqa: E402
from app.services.history_store import ChatHistoryStore  # noqa: E402
from app.services.llm import build_llm, build_openrouter_llm  # noqa: E402
from app.services.medical_report import MedicalReportService  # noqa: E402
from app.services.memory import MemoryService  # noqa: E402
from app.services.onboarding import OnboardingService  # noqa: E402
from app.services.prompts import QUERY_GENERATION_PROMPT  # noqa: E402
from app.services.retrieval import RetrievalService  # noqa: E402
from app.services.session_manager import SessionManager  # noqa: E402
from app.services.speech import SpeechService  # noqa: E402
from app.services.whatsapp_client import WhatsAppClient  # noqa: E402
import os
from dotenv import load_dotenv

load_dotenv()

log = get_logger(__name__)


def _build_send_message_fn(whatsapp_client: WhatsAppClient | None):
    """SessionManager's nudge/close messages go out through this single
    function. If WhatsApp is configured, they're sent for real; otherwise
    they're just logged (e.g. local dev without WhatsApp credentials, or
    when driving the assistant purely through the JSON /api/chat endpoint)."""

    async def _send(user_id: str, text: str) -> None:
        if whatsapp_client is None:
            log.info("proactive_message_logged_only", user_id=user_id, text=text)
            return
        try:
            await whatsapp_client.send_text_message(user_id, text)
        except Exception as e:
            log.error("proactive_message_send_failed", user_id=user_id, error=str(e))

    return _send


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    log.info("app_starting", env=settings.app_env)

    from langchain_core.output_parsers import StrOutputParser

    llm = build_llm(settings)
    llm_medical_report = build_openrouter_llm(settings)
    query_chain = QUERY_GENERATION_PROMPT | llm | StrOutputParser()

    memory_service = MemoryService(settings)
    retrieval_service = RetrievalService(settings, llm_query_chain=query_chain)
    onboarding_service = OnboardingService(llm, memory_service)
    medical_report_service = MedicalReportService(llm_medical_report)

    speech_service = SpeechService(settings) if settings.elevenlabs_api_key else None
    document_service = DocumentService(settings) if settings.llama_cloud_api_key else None
    if speech_service is None:
        log.warning("voice_disabled", reason="ELEVENLABS_API_KEY not set")
    if document_service is None:
        log.warning("documents_disabled", reason="LLAMA_CLOUD_API_KEY not set")

    whatsapp_client: WhatsAppClient | None = None
    if settings.whatsapp_access_token and settings.whatsapp_phone_number_id:
        whatsapp_client = WhatsAppClient(settings)
        log.info("whatsapp_client_configured")
    else:
        log.warning(
            "whatsapp_disabled",
            reason="WHATSAPP_ACCESS_TOKEN / WHATSAPP_PHONE_NUMBER_ID not set",
        )

    session_manager = SessionManager(
        send_message_fn=_build_send_message_fn(whatsapp_client),
        nudge_seconds=settings.session_nudge_seconds,
        timeout_seconds=settings.session_timeout_seconds,
    )

    language_store = LanguagePreferenceStore()
    menu_store = MenuStateStore()

    chat_service = ChatService(
        llm=llm,
        memory=memory_service,
        onboarding=onboarding_service,
        retrieval=retrieval_service,
        session_manager=session_manager,
        medical_report=medical_report_service,
        speech=speech_service,
        documents=document_service,
        language_store=language_store,
        # menu_store=menu_store,
    )
    app.state.settings = settings
    app.state.memory_service = memory_service
    app.state.retrieval_service = retrieval_service
    app.state.language_store = language_store
    app.state.menu_store = menu_store
    app.state.chat_service = chat_service
    app.state.history_store = ChatHistoryStore()
    app.state.whatsapp_client = whatsapp_client
    app.state.message_dedup = MessageDedup()

    log.info("app_ready")
    yield

    log.info("app_shutting_down")
    retrieval_service.close()
    if whatsapp_client is not None:
        await whatsapp_client.aclose()


def create_app() -> FastAPI:
    settings = get_settings()
    print("Groq key:", settings.groq_api_key[:10] if settings.groq_api_key else None)
    print("="*50)
    print("Model:", settings.groq_model)
    print("="*50)
    print(settings.openrouter_api_key)

    app = FastAPI(
        title="IDH Medical Assistant API",
        version="0.1.0",
        lifespan=lifespan,
    )

    origins = ["*"] if settings.cors_origins == "*" else [
        o.strip() for o in settings.cors_origins.split(",") if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(whatsapp_router)

    return app


app = create_app()
