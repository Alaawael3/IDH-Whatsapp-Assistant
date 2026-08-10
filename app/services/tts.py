from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path

from fishaudio import FishAudio
from fishaudio.utils import save

from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)

# Characters that read badly (or not at all) when spoken aloud by TTS --
# markdown emphasis asterisks and stray "!"/"?" punctuation the LLM
# sometimes leaves in. Stripped before synthesis only; the text reply sent
# alongside the voice note is unaffected.
_TTS_STRIP_CHARS_RE = re.compile(r"[#*!?]")
_WHITESPACE_RE = re.compile(r"[ \t]{2,}")

# Arabic script ranges (main block + supplement + presentation forms A/B).
# Presence of any of these means the reply is Arabic -- at that point any
# Latin/English text and bracket characters left in the string are almost
# always stray artifacts (a leftover English term, a "(see above)" aside,
# etc.) that read jarringly when spoken, not something meant to be heard.
_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
_ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+")
_BRACKET_CHARS_RE = re.compile(r"[()\[\]{}<>]")


def clean_text_for_speech(text: str) -> str:
    """Strips characters that don't belong in synthesized speech (*, !, ?),
    and -- if the text is Arabic -- also strips any embedded English words
    and bracket characters, then collapses the extra spacing that removing
    any of this can leave behind. Pure function, safe to call on any
    string -- never raises."""
    cleaned = _TTS_STRIP_CHARS_RE.sub("", text)
    if _ARABIC_RE.search(cleaned):
        cleaned = _ENGLISH_WORD_RE.sub("", cleaned)
        cleaned = _BRACKET_CHARS_RE.sub("", cleaned)
    return cleaned.strip()


class TTSError(Exception):
    """Raised when text-to-speech synthesis fails (bad text, API error, or
    the resulting audio couldn't be written to disk). Caller should fall
    back to a text-only reply, not propagate a raw stack trace."""


class TextToSpeechService:
    """Wraps Fish Audio's TTS API. One instance is created at app startup
    (see app.core.lifespan) and reused across requests, same pattern as
    SpeechService (STT) and DocumentService.
    """

    def __init__(self, settings: Settings):
        if not settings.fish_api_key:
            raise RuntimeError(
                "FISH_API_KEY is not set. Voice replies cannot be synthesized "
                "without it -- set it in .env or disable voice replies."
            )
        # FishAudio() reads FISH_API_KEY from the environment itself.
        os.environ.setdefault("FISH_API_KEY", settings.fish_api_key)
        self._client = FishAudio()
        self._model = settings.fish_tts_model
        self._default_reference_id = settings.fish_reference_id_ar
        # Per-language voice selection -- "en" gets an English reference
        # voice, "ar" an Arabic one, so the synthesized voice actually
        # matches what's being read out loud instead of one voice reading
        # both languages.
        self._reference_ids_by_lang: dict[str, str] = {
            "en": settings.fish_reference_id_en,
            "ar": settings.fish_reference_id_ar,
        }

    def synthesize_speech(
        self, text: str, lang: str | None = None, reference_id: str | None = None
    ) -> Path:
        """Synthesizes `text` to speech and returns the path to a temp mp3
        file. Caller is responsible for deleting the file once it's been
        uploaded/sent (see routes_whatsapp.py's use of it).

        Voice selection, in priority order:
          1. `reference_id`, if given -- an explicit one-off override.
          2. `lang` ("en"/"ar") -- looked up in the configured per-language
             voices.
          3. FISH_REFERENCE_ID from settings, or the model's own default
             voice if that isn't set either.

        Raises TTSError on any failure.
        """
        cleaned_text = clean_text_for_speech(text)
        if not cleaned_text:
            log.warning("synthesize_speech_empty_after_cleaning", text=text)
            raise TTSError("Text was empty after cleaning -- nothing to synthesize.")

        effective_reference_id = (
            reference_id
            or (self._reference_ids_by_lang.get(lang) if lang else None)
            or self._default_reference_id
        )
        started = time.monotonic()
        log.info(
            "synthesize_speech_start",
            text_len=len(text),
            cleaned_text_len=len(cleaned_text),
            model=self._model,
            lang=lang,
            reference_id=effective_reference_id,
        )

        kwargs: dict[str, str] = {"model": self._model, "text": cleaned_text}
        if effective_reference_id:
            kwargs["reference_id"] = effective_reference_id

        try:
            audio = self._client.tts.convert(**kwargs)
        except Exception as e:
            log.error(
                "synthesize_speech_request_failed",
                model=self._model,
                error=str(e),
            )
            raise TTSError(f"text-to-speech request failed: {e}") from e

        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        try:
            save(audio, str(tmp_path))
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            log.error("synthesize_speech_save_failed", tmp_path=str(tmp_path), error=str(e))
            raise TTSError(f"failed to save synthesized audio: {e}") from e

        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            tmp_path.unlink(missing_ok=True)
            log.warning("synthesize_speech_empty_output", text_len=len(text))
            raise TTSError("Synthesis produced an empty audio file.")

        log.info(
            "synthesize_speech_end",
            tmp_path=str(tmp_path),
            size_bytes=tmp_path.stat().st_size,
            duration_ms=round((time.monotonic() - started) * 1000, 1),
        )
        return tmp_path