"""Local embedded vector store for the RAG pipeline (ADR-003).

LanceDB, chosen over Chroma for its much lighter dependency footprint at
this project's modest scale -- see ADR-003 Decision 3. Two named tables:
`current` (time-windowed news, from EnrichedStory) and `reference`
(append-only books/long-form material, no expiry).
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lancedb
import pyarrow as pa

CURRENT_TABLE = "current"
REFERENCE_TABLE = "reference"


def _schema(vector_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("title", pa.string()),
            pa.field("url", pa.string()),
            pa.field("build_id", pa.string()),
            pa.field("published_at", pa.string()),  # ISO date, "" if unknown
            pa.field("stale", pa.bool_()),
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
        ]
    )


@dataclass
class Chunk:  # pylint: disable=too-many-instance-attributes
    """One embedded, storable unit -- one chunk_text() output plus the
    metadata needed to cite and retire it later. This is intentionally a
    flat record matching the LanceDB row shape 1:1 (see VectorStore.add) --
    splitting it into sub-objects would hurt readability at the one call
    site that actually needs all these fields together, for no real gain.
    """

    chunk_id: str
    doc_id: str
    text: str
    title: str
    url: str
    build_id: str
    published_at: str  # ISO date string, "" if unknown (e.g. books)
    vector: list[float]
    stale: bool = False


class VectorStore:
    """Wraps a LanceDB database directory. Staleness is a metadata flag,
    not a separate archive table (ADR-003 Decision 5) -- aged-out chunks
    stay queryable via include_stale=True, they're just excluded by
    default so retrieval doesn't surface month-old news.

    The vector dimension is inferred from the first chunk ever written to
    a given collection (not a constructor argument) -- the embedding
    model's output size shouldn't need to be hand-copied into a second
    place just to satisfy this store's schema.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self.path))

    def _open_table(self, name: str):
        if name not in self._db.list_tables().tables:
            return None
        return self._db.open_table(name)

    def add(self, collection: str, chunks: list[Chunk]) -> None:
        """Write chunks into `collection`, creating it (with a schema
        matching the first chunk's vector length) on first write."""
        if not chunks:
            return
        rows = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "title": c.title,
                "url": c.url,
                "build_id": c.build_id,
                "published_at": c.published_at,
                "stale": c.stale,
                "vector": c.vector,
            }
            for c in chunks
        ]
        table = self._open_table(collection)
        if table is None:
            vector_dim = len(chunks[0].vector)
            self._db.create_table(collection, schema=_schema(vector_dim), data=rows)
        else:
            table.add(rows)

    def query(
        self, collection: str, vector: list[float], top_k: int, include_stale: bool = False
    ) -> list[dict]:
        """Top-k nearest chunks in `collection` by vector distance. Returns
        [] if the collection doesn't exist yet (nothing ingested there)."""
        table = self._open_table(collection)
        if table is None:
            return []
        search = table.search(vector).limit(top_k)
        if not include_stale:
            search = search.where("stale = false")
        return search.to_list()

    def mark_stale(self, collection: str, older_than: datetime) -> int:
        """Flag chunks in `collection` published before older_than as
        stale=true. Never deletes -- see ADR-003 Decision 5. Returns the
        count flagged (0 if the collection doesn't exist)."""
        table = self._open_table(collection)
        if table is None:
            return 0
        cutoff = older_than.date().isoformat()
        where = f"published_at < '{cutoff}' AND published_at != '' AND stale = false"
        matching = table.count_rows(filter=where)
        if matching == 0:
            return 0
        table.update(where=where, values={"stale": True})
        return matching

    def count(self, collection: str) -> int:
        """Row count for `collection`, 0 if it doesn't exist yet."""
        table = self._open_table(collection)
        return table.count_rows() if table is not None else 0
