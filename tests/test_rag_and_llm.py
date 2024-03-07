import pytest

from souschef.config import Settings
from souschef.llm import OllamaBackend, StubBackend, get_backend
from souschef.prompts import PROMPTS, REFUSAL, get_prompt
from souschef.store import LogStore


def test_answer_has_citations_and_sources(pipeline):
    r = pipeline.ask("What is the maximum number of guests for an even split?", log=False)
    assert not r.refused
    assert "12" in r.answer and r.citations
    assert all(1 <= c <= len(r.sources) for c in r.citations)
    assert r.total_ms >= r.retrieval_ms >= 0


@pytest.mark.parametrize("q", ["What is the capital of France?", "Can you write me a poem about the ocean?",
                               "How do I reset my Netflix password?"])
def test_out_of_scope_is_refused(pipeline, q):
    r = pipeline.ask(q, log=False)
    assert r.refused and r.answer == REFUSAL and r.citations == []


def test_stub_is_deterministic(pipeline):
    a = pipeline.ask("How much is the French onion soup?", log=False)
    b = pipeline.ask("How much is the French onion soup?", log=False)
    assert a.answer == b.answer and "$9" in a.answer


def test_prompt_version_changes_behaviour(index):
    from souschef.rag import RagPipeline
    q = "How do I void a dish that was already sent to the kitchen?"
    v1 = RagPipeline(index, StubBackend(), prompt_version="v1").ask(q, log=False)
    v2 = RagPipeline(index, StubBackend(), prompt_version="v2").ask(q, log=False)
    assert v1.prompt_version == "v1" and v2.prompt_version == "v2"
    assert len(v1.citations) >= 1 and v1.answer.count("[") >= v2.answer.count("[")


def test_prompt_rendering_and_unknown_version():
    system, user = get_prompt("v1").render("Q?", [("Doc > H", "Body text.")])
    assert "[1] (Doc > H) Body text." in user and user.endswith("Answer:")
    with pytest.raises(ValueError):
        get_prompt("v99")


def test_backend_factory(monkeypatch):
    monkeypatch.setenv("SOUSCHEF_LLM_BACKEND", "stub")
    assert get_backend(Settings.from_env()).name == "stub"
    monkeypatch.setenv("SOUSCHEF_LLM_BACKEND", "ollama")
    assert isinstance(get_backend(Settings.from_env()), OllamaBackend)
    monkeypatch.setenv("SOUSCHEF_LLM_BACKEND", "bogus")
    with pytest.raises(ValueError):
        get_backend(Settings.from_env())


def test_ollama_backend_parses_response(monkeypatch):
    import httpx

    def fake_post(url, json, timeout):
        assert url.endswith("/api/chat") and json["options"]["temperature"] == 0
        return httpx.Response(200, json={"message": {"content": " Hello [1] "}},
                              request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "post", fake_post)
    assert OllamaBackend("http://x:1", "m").generate("s", "u") == "Hello [1]"


def test_query_log_store(pipeline):
    store = LogStore(":memory:")
    pipeline.store = store
    pipeline.ask("How do I add a new dish to the POS menu?")
    pipeline.ask("What is the capital of France?")
    rows = store.recent_queries()
    assert len(rows) == 2 and rows[0]["refused"] is True and rows[1]["sources"]
    stats = store.stats()
    assert stats["queries"] == 2 and stats["refusals"] == 1 and stats["avg_latency_ms"] > 0
    store.log_eval("x", {"a": 1}, {"m": 0.5})
    assert store.eval_runs()[0]["metrics"] == {"m": 0.5}


def test_all_prompts_have_settings_line():
    assert all("Settings:" in p.system and "{question}" in p.user for p in PROMPTS.values())
