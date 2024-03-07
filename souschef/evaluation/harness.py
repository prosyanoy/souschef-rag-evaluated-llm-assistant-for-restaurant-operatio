"""Evaluation harness: retriever comparison, generation quality and baseline regression checks."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from ..rag import RagPipeline
from ..retrievers import METHODS, RetrievalIndex
from .metrics import (first_relevant_rank, groundedness, keyword_recall, mean_reciprocal_rank,
                      recall_at_k)

KS = (1, 3, 5)
# Metrics where higher is better / lower is better when comparing against the baseline.
HIGHER_IS_BETTER = ("citation_rate", "claim_support", "grounded_rate", "keyword_recall", "oos_refusal_rate",
                    "recall@3", "mrr")
LOWER_IS_BETTER = ("false_refusal_rate",)


def evaluate_retriever(index: RetrievalIndex, gold: list[dict], method: str) -> dict[str, float]:
    ranks = []
    for row in (r for r in gold if r["in_scope"]):
        hits = index.search(row["question"], method, max(KS))
        ranks.append(first_relevant_rank([h.chunk.section_id for h in hits], set(row["relevant_sections"])))
    out = {f"recall@{k}": round(recall_at_k(ranks, k), 4) for k in KS}
    out["mrr"] = round(mean_reciprocal_rank(ranks), 4)
    out["questions"] = len(ranks)
    return out


def compare_retrievers(index: RetrievalIndex, gold: list[dict]) -> dict[str, dict[str, float]]:
    return {m: evaluate_retriever(index, gold, m) for m in METHODS}


def evaluate_generation(pipeline: RagPipeline, gold: list[dict]) -> dict:
    """Run every gold question through the pipeline and score groundedness and refusal behaviour."""
    in_scope = [r for r in gold if r["in_scope"]]
    oos = [r for r in gold if not r["in_scope"]]
    per_question: list[dict] = []
    cite, support, grounded, kw, false_refusals, refusals_ok = [], [], [], [], 0, 0

    for row in in_scope:
        res = pipeline.ask(row["question"], log=False)
        entry = {"id": row["id"], "question": row["question"], "answer": res.answer, "refused": res.refused}
        if res.refused:
            false_refusals += 1
            kw.append(0.0)
            support.append(0.0)
            grounded.append(0.0)
            cite.append(0.0)
        else:
            g = groundedness(res.answer, [s["text"] for s in res.sources])
            cite.append(g.cited_sentences / g.sentences if g.sentences else 0.0)
            support.append(g.support)
            grounded.append(float(g.grounded))
            kw.append(keyword_recall(res.answer, row["answer_keywords"]))
            entry.update(grounded=g.grounded, support=round(g.support, 3), keyword_recall=round(kw[-1], 3))
        per_question.append(entry)

    for row in oos:
        res = pipeline.ask(row["question"], log=False)
        refusals_ok += res.refused
        per_question.append({"id": row["id"], "question": row["question"], "answer": res.answer,
                             "refused": res.refused})

    metrics = {
        "citation_rate": round(mean(cite), 4) if cite else 0.0,
        "claim_support": round(mean(support), 4) if support else 0.0,
        "grounded_rate": round(mean(grounded), 4) if grounded else 0.0,
        "keyword_recall": round(mean(kw), 4) if kw else 0.0,
        "false_refusal_rate": round(false_refusals / len(in_scope), 4) if in_scope else 0.0,
        "oos_refusal_rate": round(refusals_ok / len(oos), 4) if oos else 1.0,
    }
    return {"metrics": metrics, "per_question": per_question}


def run_full_eval(pipeline: RagPipeline, gold: list[dict]) -> dict:
    """Retrieval metrics for the pipeline's retriever plus generation metrics, as one flat dict."""
    retrieval = evaluate_retriever(pipeline.index, gold, pipeline.retriever)
    gen = evaluate_generation(pipeline, gold)
    metrics = {**gen["metrics"], "recall@3": retrieval["recall@3"], "mrr": retrieval["mrr"]}
    return {"metrics": metrics, "retrieval": retrieval, "per_question": gen["per_question"]}


def check_regression(metrics: dict[str, float], baseline: dict[str, float], tolerance: float) -> list[str]:
    """Return human-readable failures for metrics that got worse than baseline by more than tolerance."""
    failures = []
    for name in HIGHER_IS_BETTER:
        if name in baseline and metrics[name] < baseline[name] - tolerance:
            failures.append(f"{name} regressed: {metrics[name]:.3f} < baseline {baseline[name]:.3f} - {tolerance}")
    for name in LOWER_IS_BETTER:
        if name in baseline and metrics[name] > baseline[name] + tolerance:
            failures.append(f"{name} regressed: {metrics[name]:.3f} > baseline {baseline[name]:.3f} + {tolerance}")
    return failures


def load_baseline(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_table(rows: dict[str, dict[str, float]]) -> str:
    cols = list(next(iter(rows.values())).keys())
    lines = ["| name | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, vals in rows.items():
        lines.append(f"| {name} | " + " | ".join(str(vals[c]) for c in cols) + " |")
    return "\n".join(lines)
