import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["SOUSCHEF_LLM_BACKEND"] = "stub"

from souschef.chunking import load_corpus  # noqa: E402
from souschef.corpus import load_gold, write_corpus  # noqa: E402
from souschef.llm import StubBackend  # noqa: E402
from souschef.rag import RagPipeline  # noqa: E402
from souschef.retrievers import RetrievalIndex  # noqa: E402


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    write_corpus(d / "corpus", d / "gold.jsonl")
    return d


@pytest.fixture(scope="session")
def chunks(data_dir):
    return load_corpus(data_dir / "corpus")


@pytest.fixture(scope="session")
def gold(data_dir):
    return load_gold(data_dir / "gold.jsonl")


@pytest.fixture(scope="session")
def index(chunks):
    return RetrievalIndex.build(chunks)


@pytest.fixture()
def pipeline(index):
    return RagPipeline(index, StubBackend(), retriever="hybrid", top_k=4, prompt_version="v2")
