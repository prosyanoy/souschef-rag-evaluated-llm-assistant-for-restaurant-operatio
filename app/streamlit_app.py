"""Streamlit chat UI: ``streamlit run app/streamlit_app.py``."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from souschef.config import Settings  # noqa: E402
from souschef.corpus import load_gold  # noqa: E402
from souschef.evaluation.harness import compare_retrievers  # noqa: E402
from souschef.llm import get_backend  # noqa: E402
from souschef.prompts import PROMPTS  # noqa: E402
from souschef.rag import RagPipeline, load_or_build_index  # noqa: E402
from souschef.retrievers import METHODS  # noqa: E402
from souschef.store import LogStore  # noqa: E402

st.set_page_config(page_title="SousChef RAG", page_icon="🍳", layout="wide")


@st.cache_resource
def get_resources():
    settings = Settings.from_env()
    return settings, load_or_build_index(settings), get_backend(settings), LogStore(settings.db_path)


settings, index, llm, store = get_resources()

with st.sidebar:
    st.title("🍳 SousChef RAG")
    retriever = st.selectbox("Retriever", METHODS, index=METHODS.index(settings.retriever))
    prompt_version = st.selectbox("Prompt version", sorted(PROMPTS),
                                  index=sorted(PROMPTS).index(settings.prompt_version))
    top_k = st.slider("Passages retrieved (top-k)", 1, 8, settings.top_k)
    st.caption(f"LLM backend: `{llm.name}` · {len(index.chunks)} chunks indexed")
    stats = store.stats()
    st.metric("Queries logged", stats["queries"])
    st.metric("Avg latency (ms)", stats["avg_latency_ms"])

chat_tab, eval_tab = st.tabs(["Chat", "Retriever comparison"])

with chat_tab:
    st.caption("Ask about POS features, billing, online ordering, or the Maison Basil menus.")
    if "history" not in st.session_state:
        st.session_state.history = []
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("result"):
                r = msg["result"]
                st.caption(f"⏱ total {r['total_ms']:.1f} ms · retrieval {r['retrieval_ms']:.1f} ms · "
                           f"generation {r['generation_ms']:.1f} ms · {r['retriever']} · prompt {r['prompt_version']}")
                for s in r["sources"]:
                    cited = s["n"] in r["citations"]
                    with st.expander(f"[{s['n']}] {s['title']} › {s['heading']} "
                                     f"(score {s['score']}){' ✅ cited' if cited else ''}"):
                        st.text(s["text"])

    if question := st.chat_input("e.g. How do I split a check by seat?"):
        st.session_state.history.append({"role": "user", "content": question})
        pipeline = RagPipeline(index, llm, retriever=retriever, top_k=top_k, prompt_version=prompt_version,
                               store=store)
        try:
            result = pipeline.ask(question).to_dict()
            st.session_state.history.append({"role": "assistant", "content": result["answer"], "result": result})
        except Exception as exc:  # e.g. Ollama not running
            st.session_state.history.append({"role": "assistant", "content": f"⚠️ Generation failed: {exc}"})
        st.rerun()

with eval_tab:
    st.write("Recall@K and MRR of each retriever on the generated gold set.")
    if st.button("Run comparison"):
        table = compare_retrievers(index, load_gold(settings.gold_path))
        st.dataframe(table, use_container_width=True)
