import pytest
from fastapi.testclient import TestClient

from souschef import api
from souschef.llm import StubBackend
from souschef.rag import RagPipeline
from souschef.store import LogStore


@pytest.fixture()
def client(index):
    pipe = RagPipeline(index, StubBackend(), store=LogStore(":memory:"))
    api.app.dependency_overrides[api.get_pipeline] = lambda: pipe
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["llm"] == "stub" and body["chunks"] > 0


def test_ask_logs_and_stats(client):
    r = client.post("/ask", json={"question": "How do I cancel my SousChef subscription?"})
    assert r.status_code == 200
    body = r.json()
    assert not body["refused"] and body["sources"] and "total_ms" in body
    assert client.get("/logs").json()[0]["question"].startswith("How do I cancel")
    assert client.get("/stats").json()["queries"] == 1


def test_ask_validation(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question?", "retriever": "x"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question?", "top_k": 99}).status_code == 422


def test_ask_per_request_retriever(client):
    r = client.post("/ask", json={"question": "What is a par level?", "retriever": "bm25", "top_k": 2})
    assert r.json()["retriever"] == "bm25" and len(r.json()["sources"]) == 2
