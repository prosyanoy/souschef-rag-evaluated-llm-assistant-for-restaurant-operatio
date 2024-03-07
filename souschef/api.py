"""FastAPI service: ``uvicorn souschef.api:app``."""
from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Settings
from .prompts import PROMPTS
from .rag import RagPipeline
from .retrievers import METHODS

app = FastAPI(title="SousChef RAG", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    retriever: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=10)


@lru_cache(maxsize=1)
def get_pipeline() -> RagPipeline:
    return RagPipeline.from_settings(Settings.from_env())


@app.get("/health")
def health(p: RagPipeline = Depends(get_pipeline)) -> dict:
    return {"status": "ok", "chunks": len(p.index.chunks), "llm": p.llm.name, "retriever": p.retriever,
            "prompt_version": p.prompt.version, "prompt_versions": sorted(PROMPTS)}


@app.post("/ask")
def ask(req: AskRequest, p: RagPipeline = Depends(get_pipeline)) -> dict:
    if req.retriever is not None and req.retriever not in METHODS:
        raise HTTPException(422, f"retriever must be one of {list(METHODS)}")
    try:
        return p.ask(req.question, retriever=req.retriever, top_k=req.top_k).to_dict()
    except Exception as exc:  # e.g. Ollama not reachable
        raise HTTPException(502, f"generation failed: {exc}") from exc


@app.get("/logs")
def logs(limit: int = 50, p: RagPipeline = Depends(get_pipeline)) -> list[dict]:
    return p.store.recent_queries(min(max(limit, 1), 500))


@app.get("/stats")
def stats(p: RagPipeline = Depends(get_pipeline)) -> dict:
    return p.store.stats()
