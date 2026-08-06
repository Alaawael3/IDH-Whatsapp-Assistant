from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.deps import get_chat_service, get_history_store, get_memory_service, verify_api_key
from app.core.logging import get_logger
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    DocumentChatResponse,
    ForgetUserResponse,
    ProfileResponse,
    VoiceChatResponse,
)
from app.services.chat_service import ChatService
from app.services.history_store import ChatHistoryStore
from app.services.memory import MemoryService

log = get_logger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"], dependencies=[Depends(verify_api_key)])

_MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    history_store: ChatHistoryStore = Depends(get_history_store),
) -> ChatResponse:
    """Plain text message."""
    chat_history = await history_store.get(payload.user_id)
    answer = await chat_service.handle_incoming_message(
        question=payload.question,
        chat_history=chat_history,
        user_id=payload.user_id,
    )
    await history_store.set(payload.user_id, chat_history)
    return ChatResponse(answer=answer)


@router.post("/voice", response_model=VoiceChatResponse)
async def chat_voice(
    user_id: str = Form(...),
    audio: UploadFile = File(...),
    chat_service: ChatService = Depends(get_chat_service),
    history_store: ChatHistoryStore = Depends(get_history_store),
) -> VoiceChatResponse:
    """Voice message -- transcribed then handled identically to text."""
    if chat_service.speech is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice messages are not enabled (ELEVENLABS_API_KEY not set).",
        )

    with _persist_upload(audio, max_bytes=_MAX_UPLOAD_BYTES) as tmp_path:
        from app.services.speech import TranscriptionError

        try:
            transcribed_question, _lang = chat_service.speech.transcribe_audio(tmp_path)
        except TranscriptionError:
            log.warning("voice_transcription_failed", user_id=user_id)
            from app.services import prompts

            return VoiceChatResponse(
                answer=prompts.VOICE_TRANSCRIPTION_FAILED_MESSAGES["ar"],
                transcribed_question=None,
            )

        chat_history = await history_store.get(user_id)
        answer = await chat_service.handle_incoming_message(
            question=transcribed_question, chat_history=chat_history, user_id=user_id
        )
        await history_store.set(user_id, chat_history)

    return VoiceChatResponse(answer=answer, transcribed_question=transcribed_question)


@router.post("/document", response_model=DocumentChatResponse)
async def chat_document(
    user_id: str = Form(...),
    question: str | None = Form(default=None),
    document: UploadFile = File(...),
    chat_service: ChatService = Depends(get_chat_service),
    history_store: ChatHistoryStore = Depends(get_history_store),
) -> DocumentChatResponse:
    """Medical test report upload (PDF or image). `question` is optional --
    if omitted, the whole report is proactively explained."""
    with _persist_upload(document, max_bytes=_MAX_UPLOAD_BYTES) as tmp_path:
        chat_history = await history_store.get(user_id)
        try:
            answer = await chat_service.handle_incoming_message(
                question=question,
                chat_history=chat_history,
                user_id=user_id,
                document_path=str(tmp_path),
            )
        except RuntimeError as e:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
        await history_store.set(user_id, chat_history)
    return DocumentChatResponse(answer=answer)


@router.get("/profile/{user_id}", response_model=ProfileResponse)
async def get_profile(
    user_id: str, memory_service: MemoryService = Depends(get_memory_service)
) -> ProfileResponse:
    profile = memory_service.get_profile(user_id)
    return ProfileResponse(**profile)


@router.delete("/profile/{user_id}", response_model=ForgetUserResponse)
async def forget_user(
    user_id: str,
    memory_service: MemoryService = Depends(get_memory_service),
    history_store: ChatHistoryStore = Depends(get_history_store),
) -> ForgetUserResponse:
    """Erase everything stored about a user (profile + diseases + short-term
    history). GDPR/right-to-erasure style endpoint."""
    memory_service.forget_user(user_id)
    await history_store.clear(user_id)
    return ForgetUserResponse(status="deleted", user_id=user_id)


def _persist_upload(upload: UploadFile, max_bytes: int):
    """Streams an UploadFile to a temp file with a size cap, returns a
    context manager yielding the path and cleaning up afterwards."""
    suffix = Path(upload.filename or "").suffix
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    size = 0
    try:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Uploaded file exceeds the size limit.",
                )
            tmp.write(chunk)
    finally:
        tmp.close()

    class _Ctx:
        def __enter__(self):
            return Path(tmp.name)

        def __exit__(self, *exc):
            Path(tmp.name).unlink(missing_ok=True)

    return _Ctx()


# from __future__ import annotations

# import shutil
# import tempfile
# from pathlib import Path

# from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

# from app.api.deps import get_chat_service, get_history_store, get_memory_service, verify_api_key
# from app.core.logging import get_logger
# from app.schemas.chat import (
#     ChatRequest,
#     ChatResponse,
#     DocumentChatResponse,
#     ForgetUserResponse,
#     ProfileResponse,
#     VoiceChatResponse,
# )
# from app.services.chat_service import ChatService
# from app.services.history_store import ChatHistoryStore
# from app.services.memory import MemoryService

# log = get_logger(__name__)

# router = APIRouter(prefix="/api/chat", tags=["chat"], dependencies=[Depends(verify_api_key)])

# _MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


# @router.post("", response_model=ChatResponse)
# async def chat(
#     payload: ChatRequest,
#     chat_service: ChatService = Depends(get_chat_service),
#     history_store: ChatHistoryStore = Depends(get_history_store),
# ) -> ChatResponse:
#     """Plain text message."""
#     chat_history = await history_store.get(payload.user_id)
#     answer = await chat_service.handle_incoming_message(
#         question=payload.question,
#         chat_history=chat_history,
#         user_id=payload.user_id,
#     )
#     await history_store.set(payload.user_id, chat_history)
#     return ChatResponse(answer=answer)


# @router.post("/voice", response_model=VoiceChatResponse)
# async def chat_voice(
#     user_id: str = Form(...),
#     audio: UploadFile = File(...),
#     chat_service: ChatService = Depends(get_chat_service),
#     history_store: ChatHistoryStore = Depends(get_history_store),
# ) -> VoiceChatResponse:
#     """Voice message -- transcribed then handled identically to text."""
#     if chat_service.speech is None:
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#             detail="Voice messages are not enabled (ELEVENLABS_API_KEY not set).",
#         )

#     with _persist_upload(audio, max_bytes=_MAX_UPLOAD_BYTES) as tmp_path:
#         from app.services.speech import TranscriptionError

#         try:
#             transcribed_question, _lang = chat_service.speech.transcribe_audio(tmp_path)
#         except TranscriptionError:
#             log.warning("voice_transcription_failed", user_id=user_id)
#             from app.services import prompts

#             return VoiceChatResponse(
#                 answer=prompts.VOICE_TRANSCRIPTION_FAILED_MESSAGES["ar"],
#                 transcribed_question=None,
#             )

#         chat_history = await history_store.get(user_id)
#         answer = await chat_service.handle_incoming_message(
#             question=transcribed_question, chat_history=chat_history, user_id=user_id
#         )
#         await history_store.set(user_id, chat_history)

#     return VoiceChatResponse(answer=answer, transcribed_question=transcribed_question)


# @router.post("/document", response_model=DocumentChatResponse)
# async def chat_document(
#     user_id: str = Form(...),
#     question: str | None = Form(default=None),
#     document: UploadFile = File(...),
#     chat_service: ChatService = Depends(get_chat_service),
#     history_store: ChatHistoryStore = Depends(get_history_store),
# ) -> DocumentChatResponse:
#     """Medical test report upload (PDF or image). `question` is optional --
#     if omitted, the whole report is proactively explained."""
#     with _persist_upload(document, max_bytes=_MAX_UPLOAD_BYTES) as tmp_path:
#         chat_history = await history_store.get(user_id)
#         try:
#             answer = await chat_service.handle_incoming_message(
#                 question=question,
#                 chat_history=chat_history,
#                 user_id=user_id,
#                 document_path=str(tmp_path),
#             )
#         except RuntimeError as e:
#             raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
#         await history_store.set(user_id, chat_history)
#     return DocumentChatResponse(answer=answer)


# @router.get("/profile/{user_id}", response_model=ProfileResponse)
# async def get_profile(
#     user_id: str, memory_service: MemoryService = Depends(get_memory_service)
# ) -> ProfileResponse:
#     profile = memory_service.get_profile(user_id)
#     return ProfileResponse(**profile)


# @router.delete("/profile/{user_id}", response_model=ForgetUserResponse)
# async def forget_user(
#     user_id: str,
#     memory_service: MemoryService = Depends(get_memory_service),
#     history_store: ChatHistoryStore = Depends(get_history_store),
# ) -> ForgetUserResponse:
#     """Erase everything stored about a user (profile + diseases + short-term
#     history). GDPR/right-to-erasure style endpoint."""
#     memory_service.forget_user(user_id)
#     await history_store.clear(user_id)
#     return ForgetUserResponse(status="deleted", user_id=user_id)


# def _persist_upload(upload: UploadFile, max_bytes: int):
#     """Streams an UploadFile to a temp file with a size cap, returns a
#     context manager yielding the path and cleaning up afterwards."""
#     suffix = Path(upload.filename or "").suffix
#     tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
#     size = 0
#     try:
#         while chunk := upload.file.read(1024 * 1024):
#             size += len(chunk)
#             if size > max_bytes:
#                 raise HTTPException(
#                     status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
#                     detail="Uploaded file exceeds the size limit.",
#                 )
#             tmp.write(chunk)
#     finally:
#         tmp.close()

#     class _Ctx:
#         def __enter__(self):
#             return Path(tmp.name)

#         def __exit__(self, *exc):
#             Path(tmp.name).unlink(missing_ok=True)

#     return _Ctx()
