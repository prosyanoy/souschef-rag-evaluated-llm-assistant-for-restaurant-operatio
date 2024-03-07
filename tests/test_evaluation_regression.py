"""Prompt-version regression tests: these run in CI and fail when a prompt change degrades quality."""
import json
from pathlib import Path

import pytest

from souschef.evaluation.harness import check_regression, run_full_eval
from souschef.llm import StubBackend
from souschef.prompts import PROMPTS
from souschef.rag import RagPipeline

BASELINE = json.loads((Path(__file__).resolve().parent.parent / "eval" / "baseline.json").read_text())


def test_every_prompt_version_is_baselined():
    assert set(PROMPTS) == set(BASELINE["prompts"])


@pytest.mark.parametrize("version", sorted(PROMPTS))
def test_prompt_text_is_pinned(version):
    """Editing a prompt requires bumping its version or re-running `eval --update-baseline`."""
    assert PROMPTS[version].fingerprint == BASELINE["prompts"][version]["fingerprint"], \
        f"prompt {version} changed; re-baseline with `python -m souschef.cli eval --update-baseline`"


@pytest.mark.parametrize("version", sorted(PROMPTS))
def test_no_metric_regression(index, gold, version):
    pipe = RagPipeline(index, StubBackend(), retriever=BASELINE["retriever"], prompt_version=version)
    metrics = run_full_eval(pipe, gold)["metrics"]
    failures = check_regression(metrics, BASELINE["prompts"][version]["metrics"], BASELINE["tolerance"])
    assert not failures, failures


def test_quality_floors_for_latest_prompt(index, gold):
    pipe = RagPipeline(index, StubBackend(), retriever="hybrid", prompt_version="v2")
    m = run_full_eval(pipe, gold)["metrics"]
    assert m["grounded_rate"] >= 0.9
    assert m["oos_refusal_rate"] >= 0.85
    assert m["false_refusal_rate"] <= 0.1


def test_regression_checker_detects_drop():
    base = {"grounded_rate": 0.95, "false_refusal_rate": 0.02}
    assert check_regression({"grounded_rate": 0.80, "false_refusal_rate": 0.02}, base, 0.02)
    assert check_regression({"grounded_rate": 0.95, "false_refusal_rate": 0.30}, base, 0.02)
    assert not check_regression({"grounded_rate": 0.94, "false_refusal_rate": 0.03}, base, 0.02)
