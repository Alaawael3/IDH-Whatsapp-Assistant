from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for all configuration.

    Everything the notebook hardcoded (API keys, Weaviate URL, model names,
    retrieval thresholds...) now comes from environment variables / .env.
    See .env.example for the full list and description of each field.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- App ---
    app_env: str = "development"
    log_level: str = "INFO"
    api_key: str | None = None  # if set, required as `X-API-Key` header on /api/*
    cors_origins: str = "*"

    # --- Groq ---
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.2
    groq_max_tokens: int = 700

    openrouter_api_key: str 

    # --- Weaviate ---
    weaviate_mode: Literal["cloud", "local", "embedded"] = "cloud"
    weaviate_url: str | None = None
    weaviate_api_key: str | None = None
    weaviate_local_host: str = "localhost"
    weaviate_local_port: int = 8080
    weaviate_local_grpc_port: int = 50051
    weaviate_embedded_data_path: str = "./data/weaviate_data"
    weaviate_collection_name: str = "IDHKnowledge"

    # --- Embeddings ---
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_use_fp16: bool = False

    # --- Retrieval ---
    retrieval_top_k: int = 10
    retrieval_final_k: int = 4
    hybrid_alpha: float = 0.5
    relevance_score_threshold: float = 0.3

    # --- mem0 ---
    mem0_llm_model: str = "llama-3.1-8b-instant"
    mem0_embedding_model: str = "BAAI/bge-m3"
    mem0_embedding_dims: int = 1024
    mem0_collection_name: str = "idh_user_memories"
    mem0_max_memories_in_prompt: int = 5
    mem0_search_threshold: float = 0.3
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_local_path: str = "./data/mem0_qdrant_data"

    # --- ElevenLabs ---
    elevenlabs_api_key: str | None = None
    elevenlabs_stt_model_id: str = "scribe_v2"

    # --- LlamaCloud ---
    llama_cloud_api_key: str | None = None

    # --- HF ---
    hf_token: str | None = None
    hf_hub_offline: bool = False

    # --- Session manager ---
    session_nudge_seconds: int = 60
    session_timeout_seconds: int = 90

    # --- WhatsApp Cloud API ---
    whatsapp_access_token: str | None = None
    whatsapp_phone_number_id: str | None = None
    whatsapp_verify_token: str | None = None
    # App secret used to verify the X-Hub-Signature-256 header on incoming
    # webhooks. Strongly recommended in production; if unset, verification
    # is skipped (logged as a warning) so local testing isn't blocked.
    whatsapp_app_secret: str | None = None
    whatsapp_api_version: str = "v25.0"


@lru_cache
def get_settings() -> Settings:
    return Settings()
