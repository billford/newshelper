"""Client for a local embedding model served through Ollama or Olla
(ADR-003). Olla speaks the Ollama-compatible API, so this client works
unmodified against either a bare Ollama host or an Olla-fronted cluster --
same shape as ollama_client.py, just pointed at /api/embed instead of
/api/generate.
"""

from dataclasses import dataclass
from typing import Protocol

import requests


class EmbedClientProtocol(Protocol):
    """Interface rag_ingest.py depends on, so a fake can stand in during tests."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in the same order."""


@dataclass
class OllamaEmbedClient:
    """Real client that calls a local Ollama/Olla /api/embed endpoint."""

    host: str
    model: str
    timeout_seconds: int = 30

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns [] for an empty input list
        without making a network call."""
        if not texts:
            return []
        response = requests.post(
            f"{self.host}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        embeddings = payload.get("embeddings")
        if embeddings is None or len(embeddings) != len(texts):
            got = len(embeddings) if embeddings is not None else 0
            raise ValueError(f"embedding response had {got} vectors for {len(texts)} inputs")
        return embeddings


class FakeEmbedClient:
    """Deterministic stand-in for tests -- no live model required.

    Vectors are derived from each text's content (not random), so the same
    text always embeds to the same vector within a test run, which is
    enough to exercise store/retrieval logic without a real model.
    """

    def __init__(self, dimension: int = 8):
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one deterministic fake vector per input text."""
        return [
            [float((hash((t, i)) % 1000) / 1000) for i in range(self.dimension)] for t in texts
        ]
