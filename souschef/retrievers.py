"""BM25, dense (FAISS) and hybrid retrieval over a shared chunk list."""
from __future__ import annotations

import json
import math
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .chunking import Chunk
from .text import tokenize

METHODS = ("bm25", "dense", "hybrid")


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int  # 1-based


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.lengths = np.array([len(d) for d in docs], dtype=float)
        self.avg_len = float(self.lengths.mean()) if len(docs) else 0.0
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf))
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    norm = 1 - self.b + self.b * self.lengths[i] / self.avg_len
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * norm)
        return out


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> None: ...
    def encode(self, texts: list[str]) -> np.ndarray: ...


class LsaEmbedder:
    """Local dense embeddings: char n-gram TF-IDF compressed by truncated SVD (LSA)."""

    kind = "lsa"

    def __init__(self, dim: int = 128, seed: int = 7):
        self.dim, self.seed = dim, seed
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                                          lowercase=True)
        self.svd: TruncatedSVD | None = None

    def fit(self, texts: list[str]) -> None:
        x = self.vectorizer.fit_transform(texts)
        dim = max(2, min(self.dim, x.shape[0] - 1, x.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=dim, random_state=self.seed)
        self.svd.fit(x)

    def encode(self, texts: list[str]) -> np.ndarray:
        assert self.svd is not None, "fit() first"
        v = self.svd.transform(self.vectorizer.transform(texts)).astype("float32")
        return _normalize(v)


class SentenceTransformerEmbedder:
    """Optional neural embeddings (needs `pip install sentence-transformers` and the model weights)."""

    kind = "sentence-transformers"

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy: heavy optional dependency
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, texts: list[str]) -> None:  # pre-trained; nothing to fit
        self._load()

    def encode(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.asarray(self._load().encode(texts), dtype="float32"))

    def __getstate__(self):
        return {"model_name": self.model_name, "_model": None}


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (v / norms).astype("float32")


def make_embedder(backend: str, model_name: str) -> Embedder:
    if backend == "lsa":
        return LsaEmbedder()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"Unknown embedding backend {backend!r}")


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    fused: Counter[int] = Counter()
    for ranking in rankings:
        for pos, idx in enumerate(ranking):
            fused[idx] += 1.0 / (k + pos + 1)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


class RetrievalIndex:
    """All three retrievers over one chunk list; persisted to / loaded from a directory."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder, faiss_index: faiss.Index | None = None):
        self.chunks = chunks
        self.embedder = embedder
        self.bm25 = BM25([tokenize(c.full_text) for c in chunks])
        if faiss_index is None:
            vectors = embedder.encode([c.full_text for c in chunks])
            faiss_index = faiss.IndexFlatIP(vectors.shape[1])
            faiss_index.add(vectors)
        self.faiss = faiss_index

    @classmethod
    def build(cls, chunks: list[Chunk], embedding_backend: str = "lsa",
              embedding_model: str = "all-MiniLM-L6-v2") -> "RetrievalIndex":
        embedder = make_embedder(embedding_backend, embedding_model)
        embedder.fit([c.full_text for c in chunks])
        return cls(chunks, embedder)

    # -- persistence -------------------------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "chunks.jsonl").write_text(
            "".join(json.dumps(c.to_dict(), ensure_ascii=False) + "\n" for c in self.chunks), encoding="utf-8")
        faiss.write_index(self.faiss, str(directory / "dense.faiss"))
        with open(directory / "embedder.pkl", "wb") as fh:  # our own artefact; never load untrusted pickles
            pickle.dump(self.embedder, fh)

    @classmethod
    def load(cls, directory: Path) -> "RetrievalIndex":
        chunks = [Chunk(**json.loads(line)) for line in
                  (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        with open(directory / "embedder.pkl", "rb") as fh:
            embedder = pickle.load(fh)
        return cls(chunks, embedder, faiss.read_index(str(directory / "dense.faiss")))

    # -- search ------------------------------------------------------------------------------
    def _bm25_ranking(self, query: str, depth: int) -> list[tuple[int, float]]:
        scores = self.bm25.scores(tokenize(query))
        order = np.argsort(-scores, kind="stable")[:depth]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def _dense_ranking(self, query: str, depth: int) -> list[tuple[int, float]]:
        depth = min(depth, len(self.chunks))
        sims, ids = self.faiss.search(self.embedder.encode([query]), depth)
        return [(int(i), float(s)) for i, s in zip(ids[0], sims[0]) if i >= 0]

    def search(self, query: str, method: str = "hybrid", k: int = 4) -> list[Hit]:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        depth = max(k, 20)
        if method == "bm25":
            ranked = self._bm25_ranking(query, depth)
        elif method == "dense":
            ranked = self._dense_ranking(query, depth)
        else:
            fused = reciprocal_rank_fusion([[i for i, _ in self._bm25_ranking(query, depth)],
                                            [i for i, _ in self._dense_ranking(query, depth)]])
            ranked = fused
        return [Hit(self.chunks[i], s, r + 1) for r, (i, s) in enumerate(ranked[:k])]
