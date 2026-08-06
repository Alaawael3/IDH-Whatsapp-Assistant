from __future__ import annotations

import os
from pathlib import Path

from llama_cloud import LlamaCloud

from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)

_MAX_REPORT_CHARS = 12000  # keep the prompt size sane; truncate very long multi-page reports


class DocumentParseError(Exception):
    """Raised when a medical report file (PDF/image) can't be parsed. Caller
    should show the user a friendly retry message, not a raw stack trace."""


class DocumentService:
    def __init__(self, settings: Settings):
        if not settings.llama_cloud_api_key:
            raise RuntimeError(
                "LLAMA_CLOUD_API_KEY is not set. Medical report uploads cannot be parsed "
                "without it -- set it in .env or disable the document endpoint."
            )
        # LlamaCloud() reads LLAMA_CLOUD_API_KEY from the environment itself.
        os.environ.setdefault("LLAMA_CLOUD_API_KEY", settings.llama_cloud_api_key)
        self._client = LlamaCloud()

    def parse_medical_document(self, file_path: str | os.PathLike) -> str:
        """Extracts text from a medical report file -- PDF or image (scan of
        a printed report) -- using LlamaCloud's agentic parsing tier, which
        is stronger at reading structured tables (lab value / reference
        range rows) than the plain OCR tier.

        Returns the extracted text as markdown, concatenated across all
        pages (so tables stay readable). Raises DocumentParseError on any
        failure.
        """
        path = Path(file_path)
        if not path.exists():
            raise DocumentParseError(f"File not found: {file_path}")

        try:
            uploaded_file = self._client.files.create(file=str(path), purpose="parse")
            result = self._client.parsing.parse(
                file_id=uploaded_file.id,
                tier="agentic",
                version="latest",
                expand=["markdown"],
            )
        except Exception as e:
            raise DocumentParseError(f"document parsing failed: {e}") from e

        pages = getattr(getattr(result, "markdown", None), "pages", None)
        if not pages:
            raise DocumentParseError("Parser returned no pages.")

        text = "\n\n".join(
            page.markdown for page in pages if getattr(page, "markdown", None)
        ).strip()

        if not text:
            raise DocumentParseError("Parsed document is empty -- the scan may be unreadable.")

        if len(text) > _MAX_REPORT_CHARS:
            text = text[:_MAX_REPORT_CHARS] + "\n\n[...جزء من التقرير تم اختصاره...]"

        return text
