from souschef.evaluation.harness import compare_retrievers, evaluate_retriever
from souschef.retrievers import BM25, RetrievalIndex, reciprocal_rank_fusion
from souschef.text import tokenize


def test_bm25_ranks_matching_doc_first():
    docs = [tokenize("kitchen printer paper"), tokenize("split check by seat"), tokenize("tip pool roles")]
    scores = BM25(docs).scores(tokenize("how to split a check"))
    assert scores.argmax() == 1 and scores[0] == 0


def test_rrf_prefers_items_ranked_high_in_both_lists():
    fused = dict(reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]]))
    assert fused[1] == fused[2] > fused[3] > 0
    assert fused[3] == fused[4]  # each appears once, at the same rank, in different lists


def test_each_method_finds_obvious_answer(index):
    for method in ("bm25", "dense", "hybrid"):
        hits = index.search("How do I split a check evenly between guests?", method, 3)
        assert hits[0].chunk.section_id == "pos-split-checks#splitting-a-check-evenly", method
        assert [h.rank for h in hits] == [1, 2, 3][: len(hits)]


def test_unknown_method_rejected(index):
    import pytest
    with pytest.raises(ValueError):
        index.search("x", "magic")


def test_index_roundtrip(index, tmp_path):
    index.save(tmp_path)
    loaded = RetrievalIndex.load(tmp_path)
    q = "What is a par level?"
    assert [h.chunk.chunk_id for h in loaded.search(q, "hybrid", 4)] == \
           [h.chunk.chunk_id for h in index.search(q, "hybrid", 4)]


def test_retrieval_quality_floor(index, gold):
    table = compare_retrievers(index, gold)
    for method, m in table.items():
        assert m["recall@5"] >= 0.95, (method, m)
        assert m["mrr"] >= 0.85, (method, m)
    assert evaluate_retriever(index, gold, "hybrid")["recall@1"] <= evaluate_retriever(index, gold, "hybrid")["recall@3"]
