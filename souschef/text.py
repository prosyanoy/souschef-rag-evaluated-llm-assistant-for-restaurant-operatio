"""Tokenisation helpers shared by retrieval, the stub LLM and the evaluation metrics."""
from __future__ import annotations

import re

STOPWORDS = frozenset("""
a an and are as at be but by can could do does for from had has have how i if in into is it its
me my of on or our should so than that the their them then there these they this to us was we
what when where which who whom why will with would you your about any also not no yes get got
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['$.][a-z0-9]+)*")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z$\-\d])|\n+")


def stem(token: str) -> str:
    """Very light suffix stripping; enough to match plural/verb forms."""
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def tokenize(text: str, *, drop_stopwords: bool = True, do_stem: bool = True) -> list[str]:
    tokens = [t.replace("'", "") for t in _TOKEN_RE.findall(text.lower())]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [stem(t) for t in tokens] if do_stem else tokens


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s and s.strip()]
