from pathlib import Path

from souschef.chunking import chunk_document, parse_markdown, slugify, split_body
from souschef.corpus import build_gold, render_doc
from souschef.corpus_data import DOCS

ROOT = Path(__file__).resolve().parent.parent


def test_parse_markdown_extracts_metadata_and_sections():
    raw = render_doc(DOCS[0])
    meta, title, sections = parse_markdown(raw)
    assert meta["doc_id"] == DOCS[0].doc_id
    assert title == DOCS[0].title
    assert [h for h, _ in sections] == [s.heading for s in DOCS[0].sections]


def test_one_chunk_per_short_section():
    chunks = chunk_document(render_doc(DOCS[0]))
    assert len(chunks) == len(DOCS[0].sections)
    assert chunks[0].section_id == f"{DOCS[0].doc_id}#{slugify(DOCS[0].sections[0].heading)}"


def test_long_section_is_split_with_overlap():
    body = " ".join(f"Sentence number {i} talks about printers and tickets." for i in range(40))
    parts = split_body(body, max_words=30, overlap_sentences=1)
    assert len(parts) > 3
    assert all(len(p.split()) <= 30 + 8 for p in parts)
    # overlap: last sentence of a window starts the next
    assert parts[0].split(". ")[-1].rstrip(".") in parts[1]


def test_split_chunks_share_section_id():
    raw = "---\ndoc_id: d\ncategory: faq\n---\n# T\n\n## H\n" + " ".join(f"Fact {i} is here." for i in range(60))
    chunks = chunk_document(raw, max_words=20)
    assert len(chunks) > 1 and {c.section_id for c in chunks} == {"d#h"}
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_gold_set_shape():
    gold = build_gold()
    assert sum(r["in_scope"] for r in gold) >= 100
    assert any(not r["in_scope"] for r in gold)
    assert len({r["id"] for r in gold}) == len(gold)
    assert all(r["relevant_sections"] for r in gold if r["in_scope"])


def test_committed_data_matches_generator():
    """data/ is generated; fail if someone edits corpus_data.py without regenerating."""
    for doc in DOCS:
        assert (ROOT / "data" / "corpus" / f"{doc.doc_id}.md").read_text(encoding="utf-8") == render_doc(doc)
    import json
    committed = [json.loads(l) for l in (ROOT / "data" / "gold.jsonl").read_text(encoding="utf-8").splitlines()]
    assert committed == build_gold()
