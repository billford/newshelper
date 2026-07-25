"""Turns a build's EnrichedStory list into embedded, searchable chunks
(ADR-003) -- the newshelper-rag component of the RAG chatbot spec.

Per ADR-003 Decision 2, there is no article body text anywhere in this
pipeline; what gets indexed here is exactly what newshelper already
produces (title, AI summary, source citations, book/article
recommendations, fact-check result), not fetched article HTML.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from newshelper.models import EnrichedStory
from newshelper.rag_chunk import chunk_text
from newshelper.rag_config import RagConfig, load_config
from newshelper.rag_embed import EmbedClientProtocol, OllamaEmbedClient
from newshelper.rag_store import CURRENT_TABLE, REFERENCE_TABLE, Chunk, VectorStore


def slugify(text: str) -> str:
    """Filesystem/id-safe slug, e.g. for doc ids."""
    keep = "".join(c if c.isalnum() or c == " " else "" for c in text.lower())
    return "-".join(keep.split())[:60]


def synthesize_doc_id(build_date: datetime, title: str) -> str:
    """Stories have no natural id (Story.title is fuzzy-clustered, not a
    stable key -- ADR-003 Decision 4). Scoping by build date means a
    same-day re-run overwrites rather than duplicates, which is the
    intended behavior, not an edge case to guard against."""
    return f"{build_date:%Y-%m-%d}-{slugify(title)}"


def build_document_text(enriched: EnrichedStory) -> str:
    """Everything newshelper actually knows about a story, flattened into
    one text blob: title, summary, source citations, book/article
    recommendations, and the fact-check claim if present."""
    parts = [enriched.story.title, enriched.summary]
    for citation in enriched.sourced_from:
        parts.append(citation.title)
    for article in enriched.articles:
        parts.append(article.title)
    for book in enriched.books:
        parts.append(f"{book.title} by {book.author}")
    if enriched.fact_check is not None:
        parts.append(
            f"Related fact-check: {enriched.fact_check.claim_text} "
            f"(rated {enriched.fact_check.rating} by {enriched.fact_check.publisher})"
        )
    return "\n".join(p for p in parts if p)


@dataclass
class IngestResult:
    """Summary counts from one ingest_build() call, for logging at the
    build.py hook site."""

    stories_indexed: int
    current_chunks_written: int
    reference_chunks_written: int


@dataclass
class DocMeta:
    """Metadata common to every chunk of one document (a story or a book),
    bundled so _chunks_for_text doesn't need a long positional-argument
    list for what is really one logical "which document is this" value."""

    doc_id: str
    title: str
    url: str
    build_id: str
    published_at: str  # ISO date, "" if unknown (e.g. books)


@dataclass
class BuildInfo:
    """Which build this ingestion run belongs to -- bundled for the same
    reason as DocMeta."""

    date: datetime
    build_id: str


def _chunks_for_text(
    text: str, meta: DocMeta, config: RagConfig, embed_client: EmbedClientProtocol
) -> list[Chunk]:
    pieces = chunk_text(text, config.chunking.chunk_size_words, config.chunking.overlap_words)
    if not pieces:
        return []
    vectors = embed_client.embed(pieces)
    return [
        Chunk(
            chunk_id=f"{meta.doc_id}-{i}",
            doc_id=meta.doc_id,
            text=piece,
            title=meta.title,
            url=meta.url,
            build_id=meta.build_id,
            published_at=meta.published_at,
            vector=vector,
        )
        for i, (piece, vector) in enumerate(zip(pieces, vectors))
    ]


def _ingest_stories(
    enriched_stories: list[EnrichedStory],
    build: BuildInfo,
    store: VectorStore,
    embed_client: EmbedClientProtocol,
    config: RagConfig,
) -> int:
    """Embed and write every story into the current collection. Returns
    the number of chunks written."""
    published_at = build.date.date().isoformat()
    chunks_written = 0
    for enriched in enriched_stories:
        doc_id = synthesize_doc_id(build.date, enriched.story.title)
        url = enriched.story.links[0] if enriched.story.links else ""
        text = build_document_text(enriched)
        meta = DocMeta(doc_id, enriched.story.title, url, build.build_id, published_at)
        chunks = _chunks_for_text(text, meta, config, embed_client)
        store.add(CURRENT_TABLE, chunks)
        chunks_written += len(chunks)
    return chunks_written


def _ingest_books(
    enriched_stories: list[EnrichedStory],
    build: BuildInfo,
    store: VectorStore,
    embed_client: EmbedClientProtocol,
    config: RagConfig,
) -> int:
    """Embed and write every distinct book recommendation into the
    reference collection (deduped -- the same book can be recommended for
    multiple stories in one build). Returns the number of chunks written.
    """
    chunks_written = 0
    seen_book_ids: set[str] = set()
    for enriched in enriched_stories:
        for book in enriched.books:
            doc_id = f"book-{slugify(book.title)}"
            if doc_id in seen_book_ids:
                continue
            seen_book_ids.add(doc_id)
            text = f"{book.title} by {book.author}"
            meta = DocMeta(doc_id, book.title, book.url, build.build_id, "")
            chunks = _chunks_for_text(text, meta, config, embed_client)
            store.add(REFERENCE_TABLE, chunks)
            chunks_written += len(chunks)
    return chunks_written


def ingest_build(
    enriched_stories: list[EnrichedStory],
    build: BuildInfo,
    store: VectorStore,
    embed_client: EmbedClientProtocol,
    config: RagConfig,
) -> IngestResult:
    """Embed and write one build's worth of stories (-> current) and their
    book recommendations (-> reference) into the vector store."""
    current_chunks_written = _ingest_stories(enriched_stories, build, store, embed_client, config)
    reference_chunks_written = _ingest_books(enriched_stories, build, store, embed_client, config)
    return IngestResult(
        stories_indexed=len(enriched_stories),
        current_chunks_written=current_chunks_written,
        reference_chunks_written=reference_chunks_written,
    )


def prune_stale(store: VectorStore, config: RagConfig, now: datetime | None = None) -> int:
    """Flag current-collection chunks older than the retention window as
    stale (never deletes -- ADR-003 Decision 5)."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=config.retention.current_window_days)
    return store.mark_stale(CURRENT_TABLE, cutoff)


def run_post_build_ingestion(
    enriched_stories: list[EnrichedStory], build_date: datetime, build_id: str
) -> IngestResult:
    """Convenience wrapper for build.py's post-build hook: loads config,
    opens the store, embeds via Ollama/Olla, ingests, and prunes stale
    entries -- one call so the hook site stays a single try/except line.
    """
    config = load_config()
    embed_client = OllamaEmbedClient(
        host=config.embedding.host,
        model=config.embedding.model,
        timeout_seconds=config.embedding.timeout_seconds,
    )
    store = VectorStore(config.store.path)
    build = BuildInfo(build_date, build_id)
    result = ingest_build(enriched_stories, build, store, embed_client, config)
    prune_stale(store, config)
    return result
