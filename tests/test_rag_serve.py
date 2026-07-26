"""Tests for rag_serve.py's HTTP contract -- spins up a real
ThreadingHTTPServer on a random local port with a fake embed client and a
temp vector store, per ADR-003's testing requirements."""

import threading
from http.server import ThreadingHTTPServer

import pytest
import requests

from newshelper.rag_config import (
    ChatConfig,
    ChunkingConfig,
    EmbeddingConfig,
    PersonaConfig,
    RagConfig,
    RetentionConfig,
    RetrievalConfig,
    ServeConfig,
    StoreConfig,
)
from newshelper.rag_embed import FakeEmbedClient
from newshelper.rag_serve import make_handler
from newshelper.rag_store import Chunk, VectorStore


def make_config() -> RagConfig:
    return RagConfig(
        chunking=ChunkingConfig(chunk_size_words=600, overlap_words=80),
        retention=RetentionConfig(current_window_days=90),
        retrieval=RetrievalConfig(top_k=8, current_weight=0.7),
        embedding=EmbeddingConfig(model="fake", host="http://fake", timeout_seconds=1),
        chat=ChatConfig(model="fake", host="http://fake", timeout_seconds=1),
        persona=PersonaConfig(cadence_days=7),
        store=StoreConfig(path="unused"),
        serve=ServeConfig(host="127.0.0.1", port=0),
    )


@pytest.fixture
def running_server(tmp_path):
    store = VectorStore(tmp_path / "store")
    store.add(
        "current",
        [
            Chunk(
                chunk_id="c1", doc_id="d1", text="City council approves transit funding",
                title="Transit story", url="https://example.com/transit", build_id="b1",
                published_at="2026-07-20", vector=[1.0, 0.0, 0.0, 0.0],
            )
        ],
    )
    embed_client = FakeEmbedClient(dimension=4)
    config = make_config()
    handler = make_handler(store, embed_client, config)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_get_root_returns_ok_status(running_server):
    response = requests.get(f"{running_server}/", timeout=5)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_unknown_path_is_404(running_server):
    response = requests.get(f"{running_server}/nope", timeout=5)
    assert response.status_code == 404


def test_post_retrieve_returns_results(running_server):
    response = requests.post(f"{running_server}/retrieve", json={"query": "transit funding"}, timeout=5)
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["title"] == "Transit story"
    assert body["results"][0]["url"] == "https://example.com/transit"


def test_post_retrieve_empty_query_returns_empty_results(running_server):
    response = requests.post(f"{running_server}/retrieve", json={"query": ""}, timeout=5)
    assert response.status_code == 200
    assert response.json() == {"results": []}


def test_post_retrieve_missing_query_key_returns_empty_results(running_server):
    response = requests.post(f"{running_server}/retrieve", json={}, timeout=5)
    assert response.status_code == 200
    assert response.json() == {"results": []}


def test_post_retrieve_non_string_query_is_bad_request(running_server):
    response = requests.post(f"{running_server}/retrieve", json={"query": 123}, timeout=5)
    assert response.status_code == 400


def test_post_retrieve_invalid_json_is_bad_request(running_server):
    response = requests.post(
        f"{running_server}/retrieve",
        data="not json",
        headers={"Content-Type": "application/json"},
        timeout=5,
    )
    assert response.status_code == 400


def test_post_unknown_path_is_404(running_server):
    response = requests.post(f"{running_server}/nope", json={"query": "x"}, timeout=5)
    assert response.status_code == 404


def test_post_retrieve_surfaces_embedding_failure_as_502(tmp_path):
    class FailingEmbed:
        def embed(self, texts):
            raise RuntimeError("upstream embedding model unavailable")

    store = VectorStore(tmp_path / "store")
    config = make_config()
    handler = make_handler(store, FailingEmbed(), config)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        response = requests.post(f"http://127.0.0.1:{port}/retrieve", json={"query": "x"}, timeout=5)
        assert response.status_code == 502
    finally:
        server.shutdown()
        thread.join(timeout=5)
