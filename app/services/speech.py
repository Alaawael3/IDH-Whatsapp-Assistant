from __future__ import annotations

import os
from pathlib import Path

from elevenlabs.client import ElevenLabs

from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)


class TranscriptionError(Exception):
    """Raised when an audio message can't be transcribed (bad file, API
    error, empty result, etc). Caller should show the user a friendly retry
    message, not a raw stack trace."""


class SpeechService:
    def __init__(self, settings: Settings):
        if not settings.elevenlabs_api_key:
            raise RuntimeError(
                "ELEVENLABS_API_KEY is not set. Voice messages cannot be transcribed "
                "without it -- set it in .env or disable the voice endpoint."
            )
        self._model_id = settings.elevenlabs_stt_model_id
        self._client = ElevenLabs(api_key=settings.elevenlabs_api_key)

    def transcribe_audio(self, audio_path: str | os.PathLike) -> tuple[str, str | None]:
        """Transcribes a voice message to text.

        Returns (text, detected_language_code). We don't rely on the detected
        language for routing -- detect_lang() works on the transcribed text
        itself (checks the Arabic unicode range) -- but it's useful for
        logging/debugging.

        Raises TranscriptionError on any failure: missing file, API error, or
        an empty/whitespace-only transcription.
        """
        path = Path(audio_path)
        if not path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_path}")

        try:
            with open(path, "rb") as audio_file:
                transcription = self._client.speech_to_text.convert(
                    file=audio_file,
                    model_id=self._model_id,
                    language_code=None,  # auto-detect -- important for mixed AR/EN speech
                    diarize=False,
                    tag_audio_events=False,
                )
        except Exception as e:
            raise TranscriptionError(f"speech-to-text request failed: {e}") from e

        text = (getattr(transcription, "text", "") or "").strip()
        if not text:
            raise TranscriptionError("Transcription returned empty text.")

        detected_lang = getattr(transcription, "language_code", None)
        return text, detected_lang
