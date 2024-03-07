"""Versioned prompt templates.

Each version is pinned by a SHA-256 in ``eval/baseline.json`` so that editing a prompt without
bumping its version (and re-baselining) fails CI.  The ``Settings:`` line in the system prompt is
parsed by the deterministic stub backend to emulate instruction following; real LLMs simply read it.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

REFUSAL = "I don't have enough information in the SousChef knowledge base to answer that."


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    system: str
    user: str  # fields: {context}, {question}

    def render(self, question: str, passages: list[tuple[str, str]]) -> tuple[str, str]:
        """passages: [(source label, text)] -> (system, user) messages."""
        context = "\n".join(f"[{i}] ({src}) {text}" for i, (src, text) in enumerate(passages, 1))
        return self.system, self.user.format(context=context or "(no passages retrieved)", question=question)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256((self.system + "\n---\n" + self.user).encode()).hexdigest()


_USER = "Context passages:\n{context}\n\nQuestion: {question}\nAnswer:"

PROMPTS: dict[str, PromptTemplate] = {
    "v1": PromptTemplate(
        "v1",
        "You are SousChef, a support assistant for restaurant owners. Answer using the context passages "
        "and cite passages like [1]. If the context does not contain the answer, reply exactly: "
        f"{REFUSAL}\n"
        "Settings: max_sentences=3; min_support=0.34",
        _USER),
    "v2": PromptTemplate(
        "v2",
        "You are SousChef, a support assistant for restaurant owners. Use ONLY the context passages. "
        "Write at most two short sentences, every sentence must end with a citation such as [1], and never "
        "add facts that are not in the passages. If the passages do not answer the question, reply exactly: "
        f"{REFUSAL}\n"
        "Settings: max_sentences=2; min_support=0.5",
        _USER),
}


def get_prompt(version: str) -> PromptTemplate:
    try:
        return PROMPTS[version]
    except KeyError:
        raise ValueError(f"Unknown prompt version {version!r}; available: {sorted(PROMPTS)}") from None
