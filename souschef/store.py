"""SQLite persistence for query logs and evaluation runs."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    refused INTEGER NOT NULL,
    retriever TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    llm_backend TEXT NOT NULL,
    retrieval_ms REAL NOT NULL,
    generation_ms REAL NOT NULL,
    total_ms REAL NOT NULL,
    sources TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS eval_run (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    label TEXT NOT NULL,
    config TEXT NOT NULL,
    metrics TEXT NOT NULL
);
"""


class LogStore:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def log_query(self, r) -> int:  # r: souschef.rag.RagResult
        cur = self.conn.execute(
            "INSERT INTO query_log (ts, question, answer, refused, retriever, prompt_version, llm_backend,"
            " retrieval_ms, generation_ms, total_ms, sources) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), r.question, r.answer, int(r.refused), r.retriever, r.prompt_version, r.llm_backend,
             r.retrieval_ms, r.generation_ms, r.total_ms,
             json.dumps([s["chunk_id"] for s in r.sources])))
        self.conn.commit()
        return int(cur.lastrowid)

    def recent_queries(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM query_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{**dict(r), "refused": bool(r["refused"]), "sources": json.loads(r["sources"])} for r in rows]

    def stats(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(AVG(total_ms),0) AS avg_ms, COALESCE(SUM(refused),0) AS refused,"
            " COALESCE(MAX(total_ms),0) AS max_ms FROM query_log").fetchone()
        return {"queries": row["n"], "avg_latency_ms": round(row["avg_ms"], 2),
                "max_latency_ms": round(row["max_ms"], 2), "refusals": int(row["refused"])}

    def log_eval(self, label: str, config: dict, metrics: dict) -> int:
        cur = self.conn.execute("INSERT INTO eval_run (ts, label, config, metrics) VALUES (?,?,?,?)",
                                (time.time(), label, json.dumps(config), json.dumps(metrics)))
        self.conn.commit()
        return int(cur.lastrowid)

    def eval_runs(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM eval_run ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": r["id"], "ts": r["ts"], "label": r["label"], "config": json.loads(r["config"]),
                 "metrics": json.loads(r["metrics"])} for r in rows]
