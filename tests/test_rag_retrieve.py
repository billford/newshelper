"""Tests for rag_retrieve.py -- recency-weighted ranking and the
current/reference split, per ADR-003's testing requirements."""

from datetime import datetime, timezone

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
from newshelper.rag_retrieve import recency_weight, retrieve
from newshelper.rag_store import Chunk, VectorStore


def make_config(top_k: int = 8, current_weight: float = 0.7, window_days: int = 90) -> RagConfig:
    return RagConfig(
        chunking=ChunkingConfig(chunk_size_words=600, overlap_words=80),
        retention=RetentionConfig(current_window_days=window_days),
        retrieval=RetrievalConfig(top_k=top_k, current_weight=current_weight),
        embedding=EmbeddingConfig(model="fake", host="http://fake", timeout_seconds=1),
        chat=ChatConfig(model="fake", host="http://fake", timeout_seconds=1),
        persona=PersonaConfig(cadence_days=7),
        store=StoreConfig(path="unused"),
        serve=ServeConfig(host="0.0.0.0", port=8901),
    )


def make_chunk(chunk_id: str, vector, title: str, published_at: str = "2026-07-20") -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id=f"doc-{chunk_id}", text=f"text {chunk_id}", title=title,
        url=f"https://example.com/{chunk_id}", build_id="b1", published_at=published_at, vector=vector,
    )


def test_recency_weight_is_full_for_todays_date():
    now = datetime(2026, 7, 25, tzinfo=timezone.utc)
    assert recency_weight("2026-07-25", now, window_days=90) == 1.0


def test_recency_weight_decays_toward_floor_at_window_edge():
    now = datetime(2026, 7, 25, tzinfo=timezone.utc)
    weight = recency_weight("2026-04-26", now, window_days=90)  # ~90 days old
    assert weight == 0.5


def test_recency_weight_never_penalizes_undated_items():
    now = datetime(2026, 7, 25, tzinfo=timezone.utc)
    assert recency_weight("", now, window_days=90) == 1.0


def test_recency_weight_clamps_below_floor_for_very_old_dates():
    now = datetime(2026, 7, 25, tzinfo=timezone.utc)
    weight = recency_weight("2020-01-01", now, window_days=90)
    assert weight == 0.5


def test_recency_weight_handles_malformed_date_gracefully():
    now = datetime(2026, 7, 25, tzinfo=timezone.utc)
    assert recency_weight("not-a-date", now, window_days=90) == 1.0


def test_retrieve_empty_query_returns_nothing(tmp_path):
    store = VectorStore(tmp_path / "store")
    config = make_config()
    assert retrieve("", store, FakeEmbedClient(dimension=4), config) == []
    assert retrieve("   ", store, FakeEmbedClient(dimension=4), config) == []


def test_retrieve_with_empty_store_returns_nothing(tmp_path):
    store = VectorStore(tmp_path / "store")
    config = make_config()
    assert retrieve("anything", store, FakeEmbedClient(dimension=4), config) == []


def test_retrieve_prefers_nearest_vector_in_current(tmp_path):
    store = VectorStore(tmp_path / "store")
    store.add(
        "current",
        [
            make_chunk("near", [1.0, 0.0, 0.0, 0.0], "Near story"),
            make_chunk("far", [0.0, 1.0, 0.0, 0.0], "Far story"),
        ],
    )
    config = make_config(top_k=1, current_weight=1.0)

    class FixedEmbed:
        def embed(self, texts):
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    results = retrieve("a query", store, FixedEmbed(), config)
    assert len(results) == 1
    assert results[0].title == "Near story"


def test_retrieve_splits_between_current_and_reference(tmp_path):
    store = VectorStore(tmp_path / "store")
    store.add("current", [make_chunk("c1", [1.0, 0.0], "Current story")])
    store.add("reference", [make_chunk("r1", [1.0, 0.0], "A book", published_at="")])
    config = make_config(top_k=2, current_weight=0.5)

    class FixedEmbed:
        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    results = retrieve("a query", store, FixedEmbed(), config)
    collections = {r.collection for r in results}
    assert collections == {"current", "reference"}


def test_retrieve_recency_reorders_similarly_close_matches(tmp_path):
    """A slightly-further-but-fresher chunk should be able to outrank a
    slightly-closer-but-stale one -- the recency-weighted re-rank, not raw
    distance alone. (Vectors are deliberately imperfect matches for both --
    an exact distance-0 match can never be outranked, since dividing 0 by
    any recency weight is still 0; that's expected, not tested here.)"""
    store = VectorStore(tmp_path / "store")
    store.add(
        "current",
        [
            make_chunk("stale_but_close", [0.8, 0.0], "Old but on-topic", published_at="2020-01-01"),
            make_chunk("fresh_slightly_off", [0.75, 0.0], "Fresh and nearby", published_at="2026-07-25"),
        ],
    )
    config = make_config(top_k=1, current_weight=1.0, window_days=90)

    class FixedEmbed:
        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    results = retrieve(
        "a query", store, FixedEmbed(), config, now=datetime(2026, 7, 25, tzinfo=timezone.utc)
    )
    assert results[0].title == "Fresh and nearby"


def test_retrieve_returns_citation_fields(tmp_path):
    store = VectorStore(tmp_path / "store")
    store.add("current", [make_chunk("c1", [1.0, 0.0], "A story", published_at="2026-07-20")])
    config = make_config(top_k=1, current_weight=1.0)

    class FixedEmbed:
        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    results = retrieve("query", store, FixedEmbed(), config)
    assert results[0].title == "A story"
    assert results[0].url == "https://example.com/c1"
    assert results[0].published_at == "2026-07-20"
    assert results[0].text == "text c1"
