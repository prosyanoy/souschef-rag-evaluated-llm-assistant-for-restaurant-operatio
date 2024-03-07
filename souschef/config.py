"""Runtime configuration, read from environment variables with safe local defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path
    gold_path: Path
    index_dir: Path
    db_path: Path
    baseline_path: Path
    retriever: str  # bm25 | dense | hybrid
    top_k: int
    llm_backend: str  # stub | ollama
    ollama_host: str
    ollama_model: str
    prompt_version: str
    embedding_backend: str  # lsa | sentence-transformers
    embedding_model: str

    @classmethod
    def from_env(cls) -> "Settings":
        data = Path(_env("SOUSCHEF_DATA_DIR", str(ROOT / "data")))
        return cls(
            corpus_dir=data / "corpus",
            gold_path=data / "gold.jsonl",
            index_dir=Path(_env("SOUSCHEF_INDEX_DIR", str(ROOT / ".cache" / "index"))),
            db_path=Path(_env("SOUSCHEF_DB_PATH", str(ROOT / ".cache" / "souschef.db"))),
            baseline_path=ROOT / "eval" / "baseline.json",
            retriever=_env("SOUSCHEF_RETRIEVER", "hybrid"),
            top_k=int(_env("SOUSCHEF_TOP_K", "4")),
            llm_backend=_env("SOUSCHEF_LLM_BACKEND", "stub"),
            ollama_host=_env("OLLAMA_HOST", "http://localhost:11434"),
            ollama_model=_env("OLLAMA_MODEL", "llama3.2:1b"),
            prompt_version=_env("SOUSCHEF_PROMPT_VERSION", "v2"),
            embedding_backend=_env("SOUSCHEF_EMBEDDING_BACKEND", "lsa"),
            embedding_model=_env("SOUSCHEF_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        )
