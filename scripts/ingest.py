"""Build/refresh the IDH knowledge base in Weaviate from source CSVs.

NOTE ON PROVENANCE: the original notebook referenced CSV paths and chunk-
strategy names (medical_test / semantic / branch_info) in its config, but
did not actually contain the ingestion code -- the Weaviate Cloud collection
it queried was already populated by a separate process not included in what
you shared. This script is a new, clean implementation matching the same
Weaviate schema (see app/services/retrieval.py::WEAVIATE_PROPERTIES) so you
have a reproducible way to (re)build the knowledge base. Adjust the chunking
logic below to match your actual source data / previous ingestion process
once you can compare against it.

Usage:
    uv run python scripts/ingest.py \
        --tests data/idh-test-catalog-enriched.csv \
        --diseases data/diseases_comprehensive_knowledge.csv \
        --branches data/All_Branches.csv
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.services.retrieval import (  # noqa: E402
    connect_weaviate,
    get_or_create_collection,
)
from FlagEmbedding import BGEM3FlagModel  # noqa: E402

log = get_logger(__name__)


def _chunk_id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def build_medical_test_chunks(df: pd.DataFrame) -> list[dict]:
    """One chunk per test row: name + description + price + preparation."""
    chunks = []
    for _, row in df.iterrows():
        test_name = str(row.get("test_name", "")).strip()
        if not test_name:
            continue
        text_parts = [f"Test: {test_name}"]
        if row.get("description"):
            text_parts.append(f"Description: {row['description']}")
        if row.get("price_egp"):
            text_parts.append(f"Price: {row['price_egp']} EGP")
        if row.get("preparation"):
            text_parts.append(f"Preparation instructions: {row['preparation']}")

        chunks.append(
            {
                "chunk_id": _chunk_id("test", test_name),
                "text": "\n".join(text_parts),
                "document_type": "medical_test",
                "chunk_strategy": "medical_test",
                "source": "idh-test-catalog-enriched.csv",
                "test_name": test_name,
                "price_egp": str(row.get("price_egp", "")),
            }
        )
    return chunks


def build_disease_chunks(df: pd.DataFrame, min_chars: int = 200) -> list[dict]:
    """Simple paragraph-based chunking per disease entry. For closer parity
    with the notebook's semantic-chunking config (buffer_size, breakpoint
    percentile), swap this for langchain_experimental's SemanticChunker --
    left as a straightforward default here to avoid adding a dependency
    whose exact original parameters we can't verify."""
    chunks = []
    for _, row in df.iterrows():
        disease_name = str(row.get("disease_name", "")).strip()
        content = str(row.get("content", row.get("description", ""))).strip()
        if not disease_name or not content:
            continue

        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        buf = ""
        idx = 0
        for para in paragraphs:
            buf = f"{buf}\n\n{para}".strip() if buf else para
            if len(buf) >= min_chars:
                chunks.append(
                    {
                        "chunk_id": _chunk_id("disease", disease_name, str(idx)),
                        "text": f"Disease: {disease_name}\n\n{buf}",
                        "document_type": "disease_info",
                        "chunk_strategy": "semantic",
                        "source": "diseases_comprehensive_knowledge.csv",
                        "disease_name": disease_name,
                        "chunk_index": str(idx),
                    }
                )
                idx += 1
                buf = ""
        if buf:
            chunks.append(
                {
                    "chunk_id": _chunk_id("disease", disease_name, str(idx)),
                    "text": f"Disease: {disease_name}\n\n{buf}",
                    "document_type": "disease_info",
                    "chunk_strategy": "semantic",
                    "source": "diseases_comprehensive_knowledge.csv",
                    "disease_name": disease_name,
                    "chunk_index": str(idx),
                }
            )
    return chunks


def build_branch_chunks(df: pd.DataFrame) -> list[dict]:
    """One chunk per branch: location, phone, area, region."""
    chunks = []
    for _, row in df.iterrows():
        branch_name = str(row.get("branch_name", row.get("company", ""))).strip()
        if not branch_name:
            continue
        text_parts = [f"Branch: {branch_name}"]
        for label, col in (
            ("Location", "location"),
            ("Phone", "phone"),
            ("Area", "area"),
            ("Region", "region"),
        ):
            if row.get(col):
                text_parts.append(f"{label}: {row[col]}")

        chunks.append(
            {
                "chunk_id": _chunk_id("branch", branch_name),
                "text": "\n".join(text_parts),
                "document_type": "branch_info",
                "chunk_strategy": "branch_info",
                "source": "All_Branches.csv",
                "branch_name": branch_name,
                "location": str(row.get("location", "")),
                "phone": str(row.get("phone", "")),
                "companyid": str(row.get("companyid", "")),
                "company": str(row.get("company", "")),
                "area": str(row.get("area", "")),
                "region": str(row.get("region", "")),
            }
        )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", type=Path, required=False)
    parser.add_argument("--diseases", type=Path, required=False)
    parser.add_argument("--branches", type=Path, required=False)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    all_chunks: list[dict] = []
    if args.tests:
        df = pd.read_csv(args.tests)
        c = build_medical_test_chunks(df)
        log.info("loaded_medical_tests", rows=len(df), chunks=len(c))
        all_chunks += c
    if args.diseases:
        df = pd.read_csv(args.diseases)
        c = build_disease_chunks(df)
        log.info("loaded_diseases", rows=len(df), chunks=len(c))
        all_chunks += c
    if args.branches:
        df = pd.read_csv(args.branches)
        c = build_branch_chunks(df)
        log.info("loaded_branches", rows=len(df), chunks=len(c))
        all_chunks += c

    if not all_chunks:
        log.error("no_input_csvs_provided")
        raise SystemExit("Provide at least one of --tests / --diseases / --branches")

    log.info("embedding_model_loading", model=settings.embedding_model_name)
    embedding_model = BGEM3FlagModel(
        settings.embedding_model_name, use_fp16=settings.embedding_use_fp16
    )

    client = connect_weaviate(settings)
    collection = get_or_create_collection(client, settings.weaviate_collection_name)

    log.info("ingesting", total_chunks=len(all_chunks))
    batch_size = args.batch_size
    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start : start + batch_size]
        texts = [c["text"] for c in batch]
        vectors = embedding_model.encode(
            texts, return_dense=True, return_sparse=True, return_colbert_vecs=False
        )["dense_vecs"].tolist()

        with collection.batch.dynamic() as b:
            for chunk, vector in zip(batch, vectors):
                b.add_object(
                    properties=chunk,
                    uuid=uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"]),
                    vector=vector,
                )
        log.info("batch_ingested", start=start, end=start + len(batch))

    client.close()
    log.info("ingestion_complete", total_chunks=len(all_chunks))


if __name__ == "__main__":
    main()
