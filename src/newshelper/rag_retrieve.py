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


def _dedupe_by_url(rows: list[dict], limit: int, seen: set[str]) -> list[dict]:
    """Keep the first `limit` rows whose url isn't already in `seen`
    (which this mutates, so a second call sharing the same set won't
    re-admit a url the first call already took).

    The same underlying story can land in the store more than once: each
    build re-indexes with a fresh doc_id scoped by that build's date (see
    rag_ingest.synthesize_doc_id), so a story that persists across a 9am
    and 6pm build becomes two separate documents with identical
    title/url. Without this, retrieval could return -- and cite -- the
    same story two or three times instead of surfacing genuinely
    different sources.
    """
    result = []
    for row in rows:
        if len(result) >= limit:
            break
        if row["url"] in seen:
            continue
        seen.add(row["url"])
        result.append(row)
    return result


def _to_chunks(rows: list[dict], collection: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            title=r["title"], url=r["url"], text=r["text"],
            collection=collection, published_at=r["published_at"],
        )
        for r in rows
    ]


def _retrieve_current(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    vector: list[float], k: int, store: VectorStore, config: RagConfig,
    now: datetime, seen_urls: set[str],
) -> list[dict]:
    hits = store.query(CURRENT_TABLE, vector, top_k=k * _OVERFETCH_MULTIPLIER)
    window_days = config.retention.current_window_days

    def _score(row: dict) -> float:
        return row["_distance"] / recency_weight(row["published_at"], now, window_days)

    ranked = sorted(hits, key=_score)
    return _dedupe_by_url(ranked, k, seen_urls)


def _retrieve_reference(
    vector: list[float], k: int, store: VectorStore, seen_urls: set[str]
) -> list[dict]:
    if not k:
        return []
    hits = store.query(REFERENCE_TABLE, vector, top_k=k * _OVERFETCH_MULTIPLIER)
    ranked = sorted(hits, key=lambda r: r["_distance"])
    return _dedupe_by_url(ranked, k, seen_urls)


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
    or if nothing has been ingested yet. Never returns the same url twice
    (see _dedupe_by_url) -- the same story can be re-indexed by more than
    one build."""
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

    seen_urls: set[str] = set()
    current_ranked = _retrieve_current(vector, current_k, store, config, now, seen_urls)
    reference_ranked = _retrieve_reference(vector, reference_k, store, seen_urls)

    return _to_chunks(current_ranked, CURRENT_TABLE) + _to_chunks(reference_ranked, REFERENCE_TABLE)
