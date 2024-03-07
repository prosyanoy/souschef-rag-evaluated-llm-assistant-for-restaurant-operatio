"""Markdown parsing and section-aware chunking."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .text import split_sentences


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    section_id: str  # doc_id#heading-slug; the unit gold relevance is judged at
    doc_id: str
    title: str
    heading: str
    category: str
    text: str  # body only

    @property
    def full_text(self) -> str:
        """Text used for indexing: title and heading give the body context."""
        return f"{self.title}. {self.heading}. {self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def parse_markdown(raw: str) -> tuple[dict, str, list[tuple[str, str]]]:
    """Return (metadata, title, [(heading, body), ...]) for one corpus file."""
    meta: dict[str, str] = {}
    lines = raw.splitlines()
    if lines and lines[0].strip() == "---":
        end = lines.index("---", 1)
        for line in lines[1:end]:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip()
        lines = lines[end + 1:]
    title = ""
    sections: list[tuple[str, list[str]]] = []
    for line in lines:
        if line.startswith("# ") and not title:
            title = line[2:].strip()
        elif line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif sections:
            sections[-1][1].append(line)
    return meta, title, [(h, "\n".join(b).strip()) for h, b in sections if "".join(b).strip()]


def split_body(body: str, max_words: int, overlap_sentences: int) -> list[str]:
    """Pack sentences into windows of at most ``max_words`` words, with sentence overlap."""
    if len(body.split()) <= max_words:
        return [body]
    sentences = split_sentences(body)
    windows: list[list[str]] = []
    current: list[str] = []
    for sent in sentences:
        if current and len(" ".join(current + [sent]).split()) > max_words:
            windows.append(current)
            current = current[-overlap_sentences:] if overlap_sentences else []
        current.append(sent)
    if current and (not windows or current != windows[-1][-len(current):]):
        windows.append(current)
    return [" ".join(w) for w in windows]


def chunk_document(raw: str, *, max_words: int = 120, overlap_sentences: int = 1) -> list[Chunk]:
    meta, title, sections = parse_markdown(raw)
    doc_id = meta.get("doc_id") or slugify(title)
    chunks: list[Chunk] = []
    for heading, body in sections:
        section_id = f"{doc_id}#{slugify(heading)}"
        parts = split_body(body, max_words, overlap_sentences)
        for i, part in enumerate(parts):
            cid = section_id if len(parts) == 1 else f"{section_id}~{i}"
            chunks.append(Chunk(cid, section_id, doc_id, title, heading,
                                meta.get("category", "general"), part))
    return chunks


def load_corpus(corpus_dir: Path, **kwargs) -> list[Chunk]:
    files = sorted(corpus_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No markdown files in {corpus_dir}; run `python -m souschef.cli generate-data`")
    chunks: list[Chunk] = []
    for f in files:
        chunks.extend(chunk_document(f.read_text(encoding="utf-8"), **kwargs))
    return chunks
