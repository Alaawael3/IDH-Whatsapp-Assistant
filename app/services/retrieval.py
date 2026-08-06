from __future__ import annotations

import os
from typing import Any

import weaviate
from FlagEmbedding import BGEM3FlagModel
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.init import Auth

from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)

# Schema for the knowledge-base collection. Kept identical to the notebook
# so an existing populated collection stays compatible.
WEAVIATE_PROPERTIES = [
    Property(name="chunk_id", data_type=DataType.TEXT),
    Property(name="text", data_type=DataType.TEXT),
    Property(name="document_type", data_type=DataType.TEXT),
    Property(name="chunk_strategy", data_type=DataType.TEXT),
    Property(name="source", data_type=DataType.TEXT),
    Property(name="test_name", data_type=DataType.TEXT),
    Property(name="price_egp", data_type=DataType.TEXT),
    Property(name="disease_name", data_type=DataType.TEXT),
    Property(name="branch_name", data_type=DataType.TEXT),
    Property(name="location", data_type=DataType.TEXT),
    Property(name="phone", data_type=DataType.TEXT),
    Property(name="companyid", data_type=DataType.TEXT),
    Property(name="company", data_type=DataType.TEXT),
    Property(name="area", data_type=DataType.TEXT),
    Property(name="region", data_type=DataType.TEXT),
    Property(name="chunk_index", data_type=DataType.TEXT),
]
KNOWN_PROPERTY_NAMES = {p.name for p in WEAVIATE_PROPERTIES} - {"chunk_id", "text"}


def connect_weaviate(settings: Settings) -> weaviate.WeaviateClient:
    if settings.weaviate_mode == "embedded":
        return weaviate.connect_to_embedded(
            persistence_data_path=settings.weaviate_embedded_data_path
        )
    if settings.weaviate_mode == "local":
        return weaviate.connect_to_local(
            host=settings.weaviate_local_host,
            port=settings.weaviate_local_port,
            grpc_port=settings.weaviate_local_grpc_port,
        )
    if settings.weaviate_mode == "cloud":
        if not settings.weaviate_url or not settings.weaviate_api_key:
            raise RuntimeError(
                "WEAVIATE_URL and WEAVIATE_API_KEY must be set when WEAVIATE_MODE=cloud"
            )
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=settings.weaviate_url,
            auth_credentials=Auth.api_key(settings.weaviate_api_key),
        )
    raise ValueError(f"Unknown WEAVIATE_MODE: {settings.weaviate_mode!r}")


def get_or_create_collection(client: weaviate.WeaviateClient, name: str):
    if client.collections.exists(name):
        return client.collections.use(name)
    try:
        return client.collections.create(
            name=name,
            description="Chunks for diseases, lab tests, prices, and branches",
            vectorizer_config=Configure.Vectorizer.none(),
            properties=WEAVIATE_PROPERTIES,
        )
    except (AttributeError, TypeError):
        return client.collections.create(
            name=name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=WEAVIATE_PROPERTIES,
        )


class RetrievalService:
    """Owns the embedding model + Weaviate client/collection and exposes
    hybrid retrieval with multi-query expansion + Reciprocal Rank Fusion,
    exactly as in the notebook (cells 12-14).

    Construction is expensive (loads the BGE-M3 model) -- build once at app
    startup and reuse for the process lifetime.
    """

    def __init__(self, settings: Settings, llm_query_chain):
        """llm_query_chain: a LangChain runnable that turns a question into
        3 newline-separated search queries (see services/prompts.py ->
        query_chain). Passed in rather than constructed here to avoid a
        circular import between retrieval and the LLM/prompt layer."""
        self.settings = settings
        self.query_chain = llm_query_chain

        if settings.hf_token:
            os.environ.setdefault("HF_TOKEN", settings.hf_token)
        if settings.hf_hub_offline:
            # Skip huggingface_hub's per-file freshness-check network calls
            # entirely and read straight from the local cache. Only safe
            # once the model has actually been downloaded at least once.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        self.client = connect_weaviate(settings)
        if not self.client.is_ready():
            raise RuntimeError("Weaviate client connected but not ready")

        self.collection = get_or_create_collection(self.client, settings.weaviate_collection_name)

        self.embedding_model = BGEM3FlagModel(
            settings.embedding_model_name,
            use_fp16=settings.embedding_use_fp16,
        )

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            log.warning("weaviate_close_failed")

    # -- embeddings -------------------------------------------------------

    def embed_texts(self, texts: list[str], batch_size: int = 12) -> list[list[float]]:
        output = self.embedding_model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            batch_size=batch_size,
        )
        return output["dense_vecs"].tolist()

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    # -- retrieval ----------------------------------------------------------

    def hybrid_retrieve(self, query: str, k: int | None = None) -> list:
        k = k or self.settings.retrieval_top_k
        query_vector = self.embed_text(query)
        response = self.collection.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=self.settings.hybrid_alpha,
            limit=k,
            return_metadata=["score"],
        )
        return response.objects

    @staticmethod
    def reciprocal_rank_fusion(results: list[list], k: int = 60) -> list[dict[str, Any]]:
        fused_scores: dict[str, dict[str, Any]] = {}
        for docs in results:
            for rank, obj in enumerate(docs):
                doc_key = obj.properties["text"]
                if doc_key not in fused_scores:
                    fused_scores[doc_key] = {"obj": obj, "score": 0.0}
                fused_scores[doc_key]["score"] += 1 / (rank + k)

        return sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)

    def multi_query_hybrid_retrieve(
        self, question: str, k: int | None = None, final_k: int | None = None
    ) -> list[dict[str, Any]]:
        k = k or self.settings.retrieval_top_k
        final_k = final_k or self.settings.retrieval_final_k

        generated_queries = self.query_chain.invoke({"question": question})
        queries = [q.strip() for q in generated_queries.split("\n") if q.strip()]
        if not queries:
            queries = [question]

        log.debug("multi_query_generated", queries=queries)

        all_results = [self.hybrid_retrieve(q, k=k) for q in queries]
        fused_results = self.reciprocal_rank_fusion(all_results)
        return fused_results[:final_k]

    # Fields that identify a single, specific "thing" in the knowledge base.
    # Checked in this order because a chunk could carry more than one.
    _DISAMBIGUATION_FIELDS = ("branch_name", "test_name", "disease_name")

    @classmethod
    def ambiguous_field(
        cls, fused_results: list[dict[str, Any]], top_n: int = 3
    ) -> str | None:
        """Heuristic confidence check: True the top results plausibly answer
        *a* question in this general area, but disagree on WHICH specific
        branch/test/disease is meant (e.g. a generic "branches" query
        surfacing several different branches with similar RRF scores).

        multi_query_hybrid_retrieve() always returns its top-k results
        regardless of how generic the query was, so without this check the
        caller has no signal that it should ask a clarifying question
        instead of just presenting the top hit as *the* answer.

        Returns the name of the ambiguous field ("branch_name", "test_name",
        or "disease_name"), or None if the top results agree (or there's
        only one/zero results, which isn't ambiguous).

        This is a coarse heuristic, not a guarantee -- callers that expect a
        legitimate multi-item answer (e.g. "compare X and Y") should not
        rely on this alone.
        """
        top = fused_results[:top_n]
        if len(top) < 2:
            return None
        for field in cls._DISAMBIGUATION_FIELDS:
            values = {
                r["obj"].properties.get(field)
                for r in top
                if r["obj"].properties.get(field)
            }
            if len(values) > 1:
                return field
        return None

    @staticmethod
    def format_context(fused_results: list[dict[str, Any]]) -> str:
        """Turn retrieved objects into a clean context block for the answer LLM."""
        blocks = []
        for i, r in enumerate(fused_results, 1):
            props = r["obj"].properties
            blocks.append(f"[Context {i}]\n" + "\n".join(f"{k}: {v}" for k, v in props.items() if v))
        return "\n\n".join(blocks) if blocks else "No relevant context found."