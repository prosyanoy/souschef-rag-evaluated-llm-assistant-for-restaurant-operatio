"""Command line entry point: ``python -m souschef.cli <command>``."""
from __future__ import annotations

import argparse
import json
import sys

from .config import Settings
from .corpus import load_gold, write_corpus
from .evaluation.harness import (compare_retrievers, evaluate_retriever, format_table, load_baseline, run_full_eval,
                                 check_regression)
from .prompts import PROMPTS
from .rag import RagPipeline, build_index
from .retrievers import METHODS
from .store import LogStore


def _pipeline(settings: Settings, prompt_version: str | None = None) -> RagPipeline:
    p = RagPipeline.from_settings(settings, with_store=False)
    if prompt_version:
        from .prompts import get_prompt
        p.prompt = get_prompt(prompt_version)
    return p


def cmd_generate_data(s: Settings, _a) -> int:
    docs, rows = write_corpus(s.corpus_dir, s.gold_path)
    print(f"wrote {docs} documents to {s.corpus_dir} and {rows} gold questions to {s.gold_path}")
    return 0


def cmd_ingest(s: Settings, _a) -> int:
    index = build_index(s)
    print(f"indexed {len(index.chunks)} chunks ({s.embedding_backend} embeddings) into {s.index_dir}")
    return 0


def cmd_ask(s: Settings, a) -> int:
    p = RagPipeline.from_settings(s)
    r = p.ask(" ".join(a.question), retriever=a.retriever)
    print(r.answer)
    for src in r.sources:
        print(f"  [{src['n']}] {src['title']} > {src['heading']}  (score {src['score']})")
    print(f"  latency: retrieval {r.retrieval_ms:.1f} ms, generation {r.generation_ms:.1f} ms")
    return 0


def cmd_compare(s: Settings, _a) -> int:
    index = _pipeline(s).index
    table = compare_retrievers(index, load_gold(s.gold_path))
    print(format_table(table))
    return 0


def cmd_matrix(s: Settings, _a) -> int:
    """Every retriever x prompt-version combination, retrieval and generation metrics side by side."""
    base = _pipeline(s)
    gold = load_gold(s.gold_path)
    rows = {}
    for retriever in METHODS:
        for version in sorted(PROMPTS):
            p = RagPipeline(base.index, base.llm, retriever=retriever, top_k=s.top_k, prompt_version=version)
            res = run_full_eval(p, gold)
            rows[f"{retriever} / {version}"] = {
                "recall@5": evaluate_retriever(base.index, gold, retriever)["recall@5"], **res["metrics"]}
    print(format_table(rows))
    return 0


def cmd_eval(s: Settings, a) -> int:
    gold = load_gold(s.gold_path)
    versions = [a.prompt] if a.prompt else sorted(PROMPTS)
    store = LogStore(s.db_path)
    results = {}
    for v in versions:
        res = run_full_eval(_pipeline(s, v), gold)
        results[v] = res["metrics"]
        store.log_eval(f"prompt-{v}", {"retriever": s.retriever, "backend": s.llm_backend, "prompt": v},
                       res["metrics"])
    print(format_table(results))
    if a.update_baseline:
        if s.llm_backend != "stub":
            print("refusing to write a baseline from a non-deterministic backend", file=sys.stderr)
            return 2
        baseline = {"tolerance": 0.02, "retriever": s.retriever, "prompts": {
            v: {"fingerprint": PROMPTS[v].fingerprint, "metrics": results[v]} for v in versions}}
        s.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        s.baseline_path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
        print(f"baseline written to {s.baseline_path}")
        return 0
    if a.check:
        base = load_baseline(s.baseline_path)
        bad = 0
        for v in versions:
            entry = base["prompts"].get(v)
            if entry is None or entry["fingerprint"] != PROMPTS[v].fingerprint:
                print(f"{v}: prompt text changed without re-baselining (run with --update-baseline)")
                bad += 1
                continue
            for f in check_regression(results[v], entry["metrics"], base["tolerance"]):
                print(f"{v}: {f}")
                bad += 1
        print("regression check:", "FAILED" if bad else "passed")
        return 1 if bad else 0
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="souschef")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate-data", help="render the synthetic corpus and gold set").set_defaults(fn=cmd_generate_data)
    sub.add_parser("ingest", help="chunk the corpus and build the BM25/FAISS index").set_defaults(fn=cmd_ingest)
    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question", nargs="+")
    ask.add_argument("--retriever", choices=["bm25", "dense", "hybrid"])
    ask.set_defaults(fn=cmd_ask)
    sub.add_parser("compare", help="BM25 vs dense vs hybrid on the gold set").set_defaults(fn=cmd_compare)
    sub.add_parser("matrix", help="metrics for every retriever x prompt-version variant").set_defaults(fn=cmd_matrix)
    ev = sub.add_parser("eval", help="full evaluation (retrieval, groundedness, refusal)")
    ev.add_argument("--prompt", choices=sorted(PROMPTS))
    ev.add_argument("--check", action="store_true", help="fail if metrics regress against eval/baseline.json")
    ev.add_argument("--update-baseline", action="store_true")
    ev.set_defaults(fn=cmd_eval)
    args = parser.parse_args(argv)
    return args.fn(Settings.from_env(), args)


if __name__ == "__main__":
    raise SystemExit(main())
