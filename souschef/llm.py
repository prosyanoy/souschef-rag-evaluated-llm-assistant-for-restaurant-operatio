"""Pluggable LLM backends: a deterministic extractive stub (offline/tests) and Ollama (local model)."""
from __future__ import annotations

import re
from typing import Protocol

import httpx

from .config import Settings
from .prompts import REFUSAL
from .text import split_sentences, tokenize


class LLMBackend(Protocol):
    name: str

    def generate(self, system: str, user: str) -> str: ...


_PASSAGE_RE = re.compile(r"^\[(\d+)\] \(([^)]*)\) (.*?)(?=^\[\d+\] \(|\n\nQuestion:)", re.S | re.M)
_QUESTION_RE = re.compile(r"\nQuestion:\s*(.*?)\nAnswer:", re.S)
_SETTING_RE = re.compile(r"(\w+)=([\d.]+)")


class StubBackend:
    """Deterministic extractive 'LLM'.

    It follows the prompt contract the way a well-behaved model would: it parses the numbered context
    passages and the question, refuses when too few question terms are supported by the context, and
    otherwise answers with the most question-relevant context sentences, each carrying a citation.
    ``max_sentences`` and ``min_support`` are read from the ``Settings:`` line of the system prompt, so
    prompt versions change its behaviour the way they would change a real model's.
    """

    name = "stub"

    def generate(self, system: str, user: str) -> str:
        settings = {k: float(v) for k, v in _SETTING_RE.findall(system.split("Settings:")[-1])}
        max_sentences = int(settings.get("max_sentences", 2))
        min_support = settings.get("min_support", 0.5)

        qm = _QUESTION_RE.search(user)
        question = qm.group(1) if qm else ""
        passages = [(int(n), body.strip()) for n, _src, body in _PASSAGE_RE.findall(user)]
        q_terms = set(tokenize(question))
        if not q_terms or not passages:
            return REFUSAL

        context_terms = set().union(*(set(tokenize(body)) for _, body in passages))
        if len(q_terms & context_terms) / len(q_terms) < min_support:
            return REFUSAL

        scored: list[tuple[float, int, int, str]] = []
        for p_pos, (num, body) in enumerate(passages):
            for s_pos, sent in enumerate(split_sentences(body)):
                overlap = len(q_terms & set(tokenize(sent))) / len(q_terms)
                if overlap > 0:
                    scored.append((overlap, p_pos, s_pos, f"{sent.lstrip('- ').rstrip('.')}. [{num}]"))
        if not scored:
            return REFUSAL
        scored.sort(key=lambda t: (-t[0], t[1], t[2]))
        scored = [t for t in scored if t[0] >= 0.45 * scored[0][0]]  # drop weakly related sentences
        return " ".join(s[3] for s in scored[:max_sentences])


class OllamaBackend:
    name = "ollama"

    def __init__(self, host: str, model: str, timeout: float = 120.0):
        self.host, self.model, self.timeout = host.rstrip("/"), model, timeout

    def generate(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0, "seed": 7},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        resp = httpx.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


def get_backend(settings: Settings) -> LLMBackend:
    if settings.llm_backend == "stub":
        return StubBackend()
    if settings.llm_backend == "ollama":
        return OllamaBackend(settings.ollama_host, settings.ollama_model)
    raise ValueError(f"Unknown LLM backend {settings.llm_backend!r} (use 'stub' or 'ollama')")
