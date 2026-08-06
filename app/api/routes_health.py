# from __future__ import annotations

# from fastapi import APIRouter, Request

# from app.schemas.chat import HealthResponse

# router = APIRouter(tags=["health"])


# @router.get("/healthz", response_model=HealthResponse)
# async def healthz(request: Request) -> HealthResponse:
#     """Liveness/readiness probe. Checks the Weaviate connection since that's
#     the dependency most likely to silently die under the app."""
#     retrieval = getattr(request.app.state, "retrieval_service", None)
#     weaviate_ready = False
#     if retrieval is not None:
#         try:
#             weaviate_ready = retrieval.client.is_ready()
#         except Exception:
#             weaviate_ready = False
#     return HealthResponse(status="ok", weaviate_ready=weaviate_ready)


from __future__ import annotations

from fastapi import APIRouter, Request

from app.schemas.chat import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
async def healthz(request: Request) -> HealthResponse:
    """Liveness/readiness probe. Checks the Weaviate connection since that's
    the dependency most likely to silently die under the app."""
    retrieval = getattr(request.app.state, "retrieval_service", None)
    weaviate_ready = False
    if retrieval is not None:
        try:
            weaviate_ready = retrieval.client.is_ready()
        except Exception:
            weaviate_ready = False
    return HealthResponse(status="ok", weaviate_ready=weaviate_ready)
