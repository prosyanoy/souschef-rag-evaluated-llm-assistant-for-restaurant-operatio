from souschef.evaluation.metrics import (first_relevant_rank, groundedness, keyword_recall,
                                         mean_reciprocal_rank, recall_at_k)

PASSAGES = ["Servers can comp up to $15 per check; anything higher needs a manager PIN.",
            "Closed checks can be refunded from Check History within 30 days."]


def test_rank_collapses_duplicate_sections():
    assert first_relevant_rank(["a", "a", "b", "c"], {"c"}) == 3
    assert first_relevant_rank(["a"], {"z"}) is None


def test_recall_and_mrr():
    ranks = [1, 3, None, 2]
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 3) == 0.75
    assert abs(mean_reciprocal_rank(ranks) - (1 + 1 / 3 + 0.5) / 4) < 1e-9
    assert recall_at_k([], 3) == 0.0


def test_grounded_answer():
    g = groundedness("Servers can comp up to $15 per check. [1]", PASSAGES)
    assert g.grounded and g.cited_sentences == g.sentences == 1 and g.support > 0.9


def test_missing_citation_is_not_grounded():
    g = groundedness("Servers can comp up to $15 per check.", PASSAGES)
    assert not g.grounded and g.cited_sentences == 0


def test_invalid_citation_number_is_not_grounded():
    assert not groundedness("Servers can comp up to $15 per check. [7]", PASSAGES).grounded


def test_hallucinated_claim_has_low_support():
    g = groundedness("Comps are approved by the regional franchise director via telegram. [1]", PASSAGES)
    assert not g.grounded and g.support < 0.5


def test_keyword_recall():
    assert keyword_recall("It is $15 and needs a Manager PIN", ["$15", "manager pin"]) == 1.0
    assert keyword_recall("nothing", ["$15", "x"]) == 0.0
    assert keyword_recall("anything", []) == 1.0
