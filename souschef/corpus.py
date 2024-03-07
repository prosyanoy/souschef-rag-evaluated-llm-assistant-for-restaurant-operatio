"""Render the synthetic corpus to markdown files and derive the gold question set."""
from __future__ import annotations

import json
from pathlib import Path

from .chunking import slugify
from .corpus_data import DOCS, OUT_OF_SCOPE_QUESTIONS, DocSpec


def render_doc(doc: DocSpec) -> str:
    lines = ["---", f"doc_id: {doc.doc_id}", f"category: {doc.category}", "---", f"# {doc.title}", ""]
    for sec in doc.sections:
        lines += [f"## {sec.heading}", sec.body, ""]
    return "\n".join(lines)


def build_gold() -> list[dict]:
    """One in-scope record per (section, question) plus the out-of-scope questions."""
    rows: list[dict] = []
    for doc in DOCS:
        for sec in doc.sections:
            section_id = f"{doc.doc_id}#{slugify(sec.heading)}"
            for i, q in enumerate(sec.questions):
                rows.append({
                    "id": f"{section_id}:{i}",
                    "question": q,
                    "relevant_sections": [section_id],
                    "answer_keywords": list(sec.keywords),
                    "in_scope": True,
                })
    for i, q in enumerate(OUT_OF_SCOPE_QUESTIONS):
        rows.append({"id": f"oos:{i}", "question": q, "relevant_sections": [],
                     "answer_keywords": [], "in_scope": False})
    return rows


def write_corpus(corpus_dir: Path, gold_path: Path) -> tuple[int, int]:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for doc in DOCS:
        (corpus_dir / f"{doc.doc_id}.md").write_text(render_doc(doc), encoding="utf-8")
    gold = build_gold()
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in gold), encoding="utf-8")
    return len(DOCS), len(gold)


def load_gold(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
