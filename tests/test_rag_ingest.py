"""Tests for rag_ingest.py: doc-id synthesis, document text assembly, and
one integration test (fixture build -> index -> query -> correct
citations), per ADR-003's testing requirements."""

from datetime import datetime, timezone

from newshelper.models import (
    ArticleRecommendation,
    BookRecommendation,
    EnrichedStory,
    FactCheckResult,
    HeadlineCandidate,
    Story,
)
from newshelper.rag_config import (
    ChatConfig,
    ChunkingConfig,
    EmbeddingConfig,
    PersonaConfig,
    RagConfig,
    RetentionConfig,
    RetrievalConfig,
    StoreConfig,
)
from newshelper.rag_embed import FakeEmbedClient
from newshelper.rag_ingest import (
    BuildInfo,
    build_document_text,
    ingest_build,
    prune_stale,
    synthesize_doc_id,
)
from newshelper.rag_store import CURRENT_TABLE, REFERENCE_TABLE, VectorStore


def make_config(chunk_size_words: int = 600, overlap_words: int = 80, window_days: int = 90) -> RagConfig:
    return RagConfig(
        chunking=ChunkingConfig(chunk_size_words=chunk_size_words, overlap_words=overlap_words),
        retention=RetentionConfig(current_window_days=window_days),
        retrieval=RetrievalConfig(top_k=8, current_weight=0.7),
        embedding=EmbeddingConfig(model="fake", host="http://fake", timeout_seconds=1),
        chat=ChatConfig(model="fake", host="http://fake", timeout_seconds=1),
        persona=PersonaConfig(cadence_days=7),
        store=StoreConfig(path="unused"),
    )


def make_enriched_story(title: str, link: str = "https://example.com/story") -> EnrichedStory:
    candidate = HeadlineCandidate(title=title, link=link, source="bbc", published="")
    story = Story(title=title, candidates=[candidate])
    return EnrichedStory(
        story=story,
        summary="A short AI-generated summary of the story.",
        sourced_from=[ArticleRecommendation(title=f"bbc: {title}", url=link, kind="source")],
    )


def test_synthesize_doc_id_is_stable_for_the_same_title_and_date():
    build_date = datetime(2026, 7, 25, tzinfo=timezone.utc)
    id_1 = synthesize_doc_id(build_date, "Local officials announce new transit plan")
    id_2 = synthesize_doc_id(build_date, "Local officials announce new transit plan")
    assert id_1 == id_2
    assert id_1.startswith("2026-07-25-")


def test_synthesize_doc_id_differs_by_build_date():
    title = "Same headline"
    id_day_1 = synthesize_doc_id(datetime(2026, 7, 25, tzinfo=timezone.utc), title)
    id_day_2 = synthesize_doc_id(datetime(2026, 7, 26, tzinfo=timezone.utc), title)
    assert id_day_1 != id_day_2


def test_build_document_text_includes_title_and_summary():
    enriched = make_enriched_story("Fed raises interest rates")
    text = build_document_text(enriched)
    assert "Fed raises interest rates" in text
    assert "A short AI-generated summary" in text


def test_build_document_text_includes_book_and_fact_check():
    enriched = make_enriched_story("A story")
    enriched.books.append(
        BookRecommendation(
            title="The Alchemists", author="Neil Irwin", url="https://x", verified_via="Open Library"
        )
    )
    enriched.fact_check = FactCheckResult(
        claim_text="a related claim", rating="False", publisher="Example Fact Checkers", url="https://x"
    )
    text = build_document_text(enriched)
    assert "The Alchemists" in text
    assert "Neil Irwin" in text
    assert "a related claim" in text
    assert "False" in text


def test_build_document_text_handles_a_story_with_nothing_extra():
    enriched = EnrichedStory(
        story=Story(
            title="Bare story",
            candidates=[HeadlineCandidate(title="Bare story", link="https://x", source="bbc", published="")],
        ),
        summary="A summary.",
    )
    text = build_document_text(enriched)
    assert text == "Bare story\nA summary."


def test_ingest_build_writes_current_chunks_and_returns_counts(tmp_path):
    store = VectorStore(tmp_path / "store")
    config = make_config()
    embed_client = FakeEmbedClient(dimension=4)
    stories = [make_enriched_story("Story One"), make_enriched_story("Story Two")]

    result = ingest_build(stories, BuildInfo(datetime(2026, 7, 25, tzinfo=timezone.utc), "build-1"), store, embed_client, config)

    assert result.stories_indexed == 2
    assert result.current_chunks_written == 2  # each short story is one chunk
    assert store.count(CURRENT_TABLE) == 2


def test_ingest_build_writes_books_to_reference_and_dedupes(tmp_path):
    store = VectorStore(tmp_path / "store")
    config = make_config()
    embed_client = FakeEmbedClient(dimension=4)

    book = BookRecommendation(
        title="The Alchemists", author="Neil Irwin", url="https://x", verified_via="Open Library"
    )
    story_a = make_enriched_story("Story A")
    story_a.books.append(book)
    story_b = make_enriched_story("Story B")
    story_b.books.append(book)  # same book recommended twice in one build

    result = ingest_build(
        [story_a, story_b], BuildInfo(datetime(2026, 7, 25, tzinfo=timezone.utc), "build-1"), store, embed_client, config
    )

    assert result.reference_chunks_written == 1  # deduped, not written twice
    assert store.count(REFERENCE_TABLE) == 1


def test_ingest_build_with_no_stories_writes_nothing(tmp_path):
    store = VectorStore(tmp_path / "store")
    config = make_config()
    result = ingest_build([], BuildInfo(datetime(2026, 7, 25, tzinfo=timezone.utc), "build-1"), store, FakeEmbedClient(), config)
    assert result.stories_indexed == 0
    assert result.current_chunks_written == 0
    assert store.count(CURRENT_TABLE) == 0


def test_prune_stale_flags_chunks_outside_the_retention_window(tmp_path):
    store = VectorStore(tmp_path / "store")
    config = make_config(window_days=30)
    embed_client = FakeEmbedClient(dimension=4)

    old_story = make_enriched_story("Old story")
    ingest_build([old_story], BuildInfo(datetime(2026, 1, 1, tzinfo=timezone.utc), "build-old"), store, embed_client, config)
    new_story = make_enriched_story("New story")
    ingest_build([new_story], BuildInfo(datetime(2026, 7, 25, tzinfo=timezone.utc), "build-new"), store, embed_client, config)

    flagged = prune_stale(store, config, now=datetime(2026, 7, 25, tzinfo=timezone.utc))

    assert flagged == 1
    remaining = store.query(CURRENT_TABLE, embed_client.embed(["x"])[0], top_k=10)
    assert {r["title"] for r in remaining} == {"New story"}


def test_integration_fixture_build_indexes_and_is_retrievable_with_citations(tmp_path):
    """One end-to-end pass: a small fixture build -> ingest -> query ->
    assert the right chunk comes back with correct title/url metadata for
    citation purposes (the acceptance-criteria-shaped integration test)."""
    store = VectorStore(tmp_path / "store")
    config = make_config()
    embed_client = FakeEmbedClient(dimension=6)

    fixture_stories = [
        make_enriched_story("City council approves new transit funding", link="https://bbc.example/transit"),
        make_enriched_story("Wildfires prompt evacuations in Europe", link="https://npr.example/wildfires"),
    ]
    build_date = datetime(2026, 7, 25, tzinfo=timezone.utc)

    result = ingest_build(fixture_stories, BuildInfo(build_date, "fixture-build"), store, embed_client, config)
    assert result.stories_indexed == 2

    # A query "about" the transit story should retrieve that exact chunk,
    # since FakeEmbedClient is deterministic per input text.
    transit_text = build_document_text(fixture_stories[0])
    from newshelper.rag_chunk import chunk_text as _chunk_text

    query_vector = embed_client.embed(
        _chunk_text(transit_text, config.chunking.chunk_size_words, config.chunking.overlap_words)
    )[0]

    results = store.query(CURRENT_TABLE, query_vector, top_k=1)
    assert len(results) == 1
    assert results[0]["title"] == "City council approves new transit funding"
    assert results[0]["url"] == "https://bbc.example/transit"
    assert results[0]["build_id"] == "fixture-build"
    assert results[0]["published_at"] == "2026-07-25"
