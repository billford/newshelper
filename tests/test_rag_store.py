"""Tests for the LanceDB-backed vector store (rag_store.py) -- write,
query, and prune/staleness logic, per ADR-003's testing requirements."""

from datetime import datetime, timedelta, timezone

from newshelper.rag_store import Chunk, VectorStore


def make_chunk(chunk_id: str, vector: list[float], published_at: str = "2026-07-20", stale: bool = False) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        text=f"text for {chunk_id}",
        title=f"Title {chunk_id}",
        url=f"https://example.com/{chunk_id}",
        build_id="build-1",
        published_at=published_at,
        vector=vector,
        stale=stale,
    )


def test_query_on_nonexistent_collection_returns_empty(tmp_path):
    store = VectorStore(tmp_path / "store")
    assert store.query("current", [1.0, 0.0], top_k=5) == []


def test_count_on_nonexistent_collection_is_zero(tmp_path):
    store = VectorStore(tmp_path / "store")
    assert store.count("current") == 0


def test_add_then_query_returns_nearest_first(tmp_path):
    store = VectorStore(tmp_path / "store")
    store.add(
        "current",
        [
            make_chunk("a", [1.0, 0.0, 0.0, 0.0]),
            make_chunk("b", [0.0, 1.0, 0.0, 0.0]),
        ],
    )
    results = store.query("current", [1.0, 0.0, 0.0, 0.0], top_k=2)
    assert results[0]["chunk_id"] == "a"
    assert store.count("current") == 2


def test_add_empty_list_is_a_no_op(tmp_path):
    store = VectorStore(tmp_path / "store")
    store.add("current", [])
    assert store.count("current") == 0


def test_query_excludes_stale_by_default(tmp_path):
    store = VectorStore(tmp_path / "store")
    store.add(
        "current",
        [
            make_chunk("fresh", [1.0, 0.0], stale=False),
            make_chunk("old", [1.0, 0.0], stale=True),
        ],
    )
    results = store.query("current", [1.0, 0.0], top_k=5)
    assert {r["chunk_id"] for r in results} == {"fresh"}


def test_query_includes_stale_when_requested(tmp_path):
    store = VectorStore(tmp_path / "store")
    store.add(
        "current",
        [
            make_chunk("fresh", [1.0, 0.0], stale=False),
            make_chunk("old", [1.0, 0.0], stale=True),
        ],
    )
    results = store.query("current", [1.0, 0.0], top_k=5, include_stale=True)
    assert {r["chunk_id"] for r in results} == {"fresh", "old"}


def test_mark_stale_flags_old_chunks_and_leaves_recent_ones(tmp_path):
    store = VectorStore(tmp_path / "store")
    store.add(
        "current",
        [
            make_chunk("recent", [1.0, 0.0], published_at="2026-07-20"),
            make_chunk("ancient", [1.0, 0.0], published_at="2026-01-01"),
        ],
    )
    cutoff = datetime(2026, 6, 1, tzinfo=timezone.utc)
    flagged = store.mark_stale("current", cutoff)

    assert flagged == 1
    fresh_results = store.query("current", [1.0, 0.0], top_k=5)
    assert {r["chunk_id"] for r in fresh_results} == {"recent"}
    all_results = store.query("current", [1.0, 0.0], top_k=5, include_stale=True)
    assert {r["chunk_id"] for r in all_results} == {"recent", "ancient"}


def test_mark_stale_never_deletes_rows(tmp_path):
    store = VectorStore(tmp_path / "store")
    store.add("current", [make_chunk("ancient", [1.0, 0.0], published_at="2020-01-01")])
    store.mark_stale("current", datetime.now(timezone.utc))
    assert store.count("current") == 1


def test_mark_stale_ignores_reference_items_with_no_date(tmp_path):
    """Books (published_at="") should never get flagged stale by a date
    cutoff -- ADR-003 treats undated reference items as never-expiring."""
    store = VectorStore(tmp_path / "store")
    store.add("reference", [make_chunk("book", [1.0, 0.0], published_at="")])
    flagged = store.mark_stale("reference", datetime.now(timezone.utc) + timedelta(days=1))
    assert flagged == 0
    assert store.count("reference") == 1


def test_mark_stale_on_nonexistent_collection_returns_zero(tmp_path):
    store = VectorStore(tmp_path / "store")
    assert store.mark_stale("current", datetime.now(timezone.utc)) == 0


def test_current_and_reference_collections_are_independent(tmp_path):
    store = VectorStore(tmp_path / "store")
    store.add("current", [make_chunk("news-1", [1.0, 0.0])])
    store.add("reference", [make_chunk("book-1", [1.0, 0.0], published_at="")])

    assert store.count("current") == 1
    assert store.count("reference") == 1
    assert store.query("current", [1.0, 0.0], top_k=5)[0]["chunk_id"] == "news-1"
    assert store.query("reference", [1.0, 0.0], top_k=5)[0]["chunk_id"] == "book-1"
