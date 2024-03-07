"""Pure metric functions (no model or index dependencies)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..text import split_sentences, tokenize

_CITE_RE = re.compile(r"\[(\d+)\]")


def first_relevant_rank(ranked_sections: list[str], relevant: set[str]) -> int | None:
    """1-based rank of the first relevant section, collapsing repeated sections from split chunks."""
    seen: list[str] = []
    for s in ranked_sections:
        if s not in seen:
            seen.append(s)
    for i, s in enumerate(seen, 1):
        if s in relevant:
            return i
    return None


def recall_at_k(ranks: list[int | None], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    return sum(1 / r for r in ranks if r) / len(ranks) if ranks else 0.0


@dataclass(frozen=True)
class Groundedness:
    sentences: int
    cited_sentences: int  # sentences whose citations all point to a retrieved passage (and exist)
    support: float  # mean fraction of a sentence's content terms found in its cited passages
    grounded: bool


def groundedness(answer: str, passages: list[str], support_threshold: float = 0.7) -> Groundedness:
    """Check citations and claim overlap for each sentence of an answer.

    ``passages[i]`` is the text of citation ``[i+1]``.  A sentence is *cited* when it contains at least
    one citation and every citation refers to an existing passage.  Its *support* is the share of its
    content terms that occur in the cited passages.  An answer is *grounded* when every sentence is
    cited and its support reaches ``support_threshold``.
    """
    sentences = split_sentences(_CITE_RE.sub(lambda m: m.group(0), answer))
    # re-attach orphan citation tokens that the splitter may have separated from their sentence
    merged: list[str] = []
    for s in sentences:
        if merged and _CITE_RE.fullmatch(s.strip()):
            merged[-1] += " " + s
        else:
            merged.append(s)
    if not merged:
        return Groundedness(0, 0, 0.0, False)
    cited, supports = 0, []
    for sent in merged:
        nums = [int(n) for n in _CITE_RE.findall(sent)]
        valid = bool(nums) and all(1 <= n <= len(passages) for n in nums)
        cited += valid
        terms = set(tokenize(_CITE_RE.sub(" ", sent)))
        if valid and terms:
            evidence = set().union(*(set(tokenize(passages[n - 1])) for n in nums))
            supports.append(len(terms & evidence) / len(terms))
        else:
            supports.append(0.0)
    mean_support = sum(supports) / len(supports)
    grounded = cited == len(merged) and min(supports) >= support_threshold
    return Groundedness(len(merged), cited, mean_support, grounded)


def keyword_recall(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    low = answer.lower()
    return sum(1 for k in keywords if k.lower() in low) / len(keywords)
