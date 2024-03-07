"""The RAG pipeline: retrieve -> prompt -> generate -> parse citations."""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .chunking import load_corpus
from .config import Settings
from .llm import LLMBackend, get_backend
from .prompts import REFUSAL, get_prompt
from .retrievers import RetrievalIndex
from .store import LogStore

_CITE_RE = re.compile(r"\[(\d+)\]")


@dataclass
class RagResult:
    question: str
    answer: str
    refused: bool
    citations: list[int]  # 1-based passage numbers cited in the answer
    sources: list[dict]  # retrieved passages, in prompt order
    retriever: str
    prompt_version: str
    llm_backend: str
    retrieval_ms: float
    generation_ms: float
    total_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


def load_or_build_index(settings: Settings) -> RetrievalIndex:
    """Load the persisted index, or build (and persist) it from the corpus when absent."""
    idx_dir: Path = settings.index_dir
    if (idx_dir / "dense.faiss").exists():
        return RetrievalIndex.load(idx_dir)
    return build_index(settings)


def build_index(settings: Settings) -> RetrievalIndex:
    index = RetrievalIndex.build(load_corpus(settings.corpus_dir), settings.embedding_backend,
                                 settings.embedding_model)
    index.save(settings.index_dir)
    return index


class RagPipeline:
    def __init__(self, index: RetrievalIndex, llm: LLMBackend, *, retriever: str = "hybrid", top_k: int = 4,
                 prompt_version: str = "v2", store: LogStore | None = None):
        self.index, self.llm, self.store = index, llm, store
        self.retriever, self.top_k = retriever, top_k
        self.prompt = get_prompt(prompt_version)

    @classmethod
    def from_settings(cls, settings: Settings, *, with_store: bool = True) -> "RagPipeline":
        return cls(load_or_build_index(settings), get_backend(settings), retriever=settings.retriever,
                   top_k=settings.top_k, prompt_version=settings.prompt_version,
                   store=LogStore(settings.db_path) if with_store else None)

    def ask(self, question: str, *, retriever: str | None = None, top_k: int | None = None,
            log: bool = True) -> RagResult:
        retriever = retriever or self.retriever
        t0 = time.perf_counter()
        hits = self.index.search(question, retriever, top_k or self.top_k)
        t1 = time.perf_counter()
        sources = [{"n": i, "chunk_id": h.chunk.chunk_id, "section_id": h.chunk.section_id,
                    "title": h.chunk.title, "heading": h.chunk.heading, "text": h.chunk.text,
                    "score": round(h.score, 4)} for i, h in enumerate(hits, 1)]
        system, user = self.prompt.render(
            question, [(f"{s['title']} > {s['heading']}", s["text"]) for s in sources])
        answer = self.llm.generate(system, user).strip()
        t2 = time.perf_counter()
        refused = REFUSAL.lower() in answer.lower()
        citations = [] if refused else sorted({int(n) for n in _CITE_RE.findall(answer)})
        result = RagResult(question, answer, refused, citations, sources, retriever, self.prompt.version,
                           self.llm.name, round((t1 - t0) * 1000, 3), round((t2 - t1) * 1000, 3),
                           round((t2 - t0) * 1000, 3))
        if log and self.store:
            self.store.log_query(result)
        return result
