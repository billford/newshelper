"""Phase 2 (newshelper-chat) retrieval: given a user query, embed it and
return ranked, cited chunks from both collections (ADR-003).

Query flow per the spec: embed query with the same model used for
ingestion, retrieve top-k split across current/reference weighted toward
current, apply a relevance-tied recency decay within current (a month-old
article shouldn't outrank yesterday's on a similar topic, but never gets
hard-excluded), and return chunks with citation metadata intact.
"""

from dataclasses import dataclass
from datetime import date, datetime, timezone

from newshelper.rag_config import RagConfig
from newshelper.rag_embed import EmbedClientProtocol
from newshelper.rag_store import CURRENT_TABLE, REFERENCE_TABLE, VectorStore

# How much a chunk's effective distance grows per day of age, expressed as
# the floor recency weight reached by the end of the retention window --
# tuned by inspection, not a rigorously validated cutoff (same caveat
# ADR-002 notes for its own similarity threshold).
_RECENCY_FLOOR = 0.5
_OVERFETCH_MULTIPLIER = 3  # overfetch current before re-ranking by recency


@dataclass
class RetrievedChunk:
    """One retrieval result, with everything needed to cite it."""

    title: str
    url: str
    text: str
    collection: str
    published_at: str


def recency_weight(published_at: str, now: datetime, window_days: int) -> float:
    """1.0 for today's content, decaying linearly toward _RECENCY_FLOOR by
    the end of window_days. Undated items (reference/books) are never
    penalized -- they have no "age" to weight."""
    if not published_at:
        return 1.0
    try:
        pub_date = date.fromisoformat(published_at)
    except ValueError:
        return 1.0
    age_days = max(0, (now.date() - pub_date).days)
    if window_days <= 0:
        return _RECENCY_FLOOR
    fraction = min(1.0, age_days / window_days)
    return 1.0 - fraction * (1.0 - _RECENCY_FLOOR)


def retrieve(
    query: str,
    store: VectorStore,
    embed_client: EmbedClientProtocol,
    config: RagConfig,
    now: datetime | None = None,
) -> list[RetrievedChunk]:
    """Embed query and return up to config.retrieval.top_k chunks: mostly
    from `current` (weighted by config.retrieval.current_weight, re-ranked
    by recency), the rest from `reference`. Returns [] for an empty query
    or if nothing has been ingested yet."""
    if not query.strip():
        return []
    now = now or datetime.now(timezone.utc)

    vectors = embed_client.embed([query])
    if not vectors:
        return []
    vector = vectors[0]

    top_k = config.retrieval.top_k
    current_k = max(1, round(top_k * config.retrieval.current_weight))
    reference_k = max(0, top_k - current_k)

    current_hits = store.query(CURRENT_TABLE, vector, top_k=current_k * _OVERFETCH_MULTIPLIER)
    reference_hits = store.query(REFERENCE_TABLE, vector, top_k=reference_k) if reference_k else []

    window_days = config.retention.current_window_days
    current_ranked = sorted(
        current_hits,
        key=lambda r: r["_distance"] / recency_weight(r["published_at"], now, window_days),
    )[:current_k]

    return [
        RetrievedChunk(
            title=r["title"], url=r["url"], text=r["text"],
            collection=CURRENT_TABLE, published_at=r["published_at"],
        )
        for r in current_ranked
    ] + [
        RetrievedChunk(
            title=r["title"], url=r["url"], text=r["text"],
            collection=REFERENCE_TABLE, published_at=r["published_at"],
        )
        for r in reference_hits
    ]
