"""Smoke test for the health endpoint. Doesn't boot the real lifespan
(that needs live Groq/Weaviate/mem0 credentials) -- instead builds a bare
FastAPI app with the health router and a fake app.state, matching how a
real deployment's readiness probe would be exercised.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes_health import router as health_router


class _FakeRetrieval:
    class client:
        @staticmethod
        def is_ready():
            return True


def test_healthz_reports_ok():
    app = FastAPI()
    app.include_router(health_router)
    app.state.retrieval_service = _FakeRetrieval()

    client = TestClient(app)
    response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["weaviate_ready"] is True


def test_healthz_handles_missing_retrieval_service():
    app = FastAPI()
    app.include_router(health_router)
    # No app.state.retrieval_service set -- simulates a startup race/failure.

    client = TestClient(app)
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["weaviate_ready"] is False
