from __future__ import annotations

from langchain_groq import ChatGroq
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import Settings


def build_llm(settings: Settings, temperature: float | None = None) -> ChatGroq:
    """Single factory for all ChatGroq instances so model/temperature/timeout
    policy lives in one place instead of being repeated (and drifting) across
    modules like the notebook's router/onboarding/report LLM instances did.
    """
    return ChatGroq(
        model=settings.groq_model,
        temperature=settings.groq_temperature if temperature is None else temperature,
        max_tokens=settings.groq_max_tokens,
        api_key=settings.groq_api_key,
        timeout=30,
        max_retries=2,
    )

from langchain_openai import ChatOpenAI

def build_openrouter_llm(settings: Settings, temperature: float | None = None):
    return ChatOpenAI(
        model="google/gemma-4-26b-a4b-it:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
        temperature=0,
    )


# Reusable retry decorator for chain.invoke() calls hitting Groq -- covers
# transient network/rate-limit errors. Import and apply where an LLM call
# sits on the hot path of a user-facing request.
llm_retry = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
)
