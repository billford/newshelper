"""Loads config/rag.yaml for the RAG chatbot pipeline (ADR-003).

Kept separate from config.py (the digest pipeline's plain-constants
module) since this config is genuinely operator-tunable data -- chunk
size, retention window, top-k, model names, and the Olla endpoint should
all be adjustable without touching code.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "rag.yaml"

_REQUIRED_SECTIONS = (
    "chunking", "retention", "retrieval", "embedding", "chat", "persona", "store", "serve",
)


class RagConfigError(ValueError):
    """The config file is missing, malformed, or missing required sections."""


@dataclass(frozen=True)
class ChunkingConfig:
    """rag_chunk.chunk_text()'s size/overlap, in words (see its docstring
    for why words rather than real tokens)."""

    chunk_size_words: int
    overlap_words: int


@dataclass(frozen=True)
class RetentionConfig:
    """How long a `current`-collection chunk stays non-stale."""

    current_window_days: int


@dataclass(frozen=True)
class RetrievalConfig:
    """Phase 2 (chat serving) retrieval tuning -- not consumed by Phase 1."""

    top_k: int
    current_weight: float


@dataclass(frozen=True)
class EmbeddingConfig:
    """The embedding model/endpoint used for ingestion (and, in Phase 2,
    for embedding user queries -- must match, or similarity is meaningless)."""

    model: str
    host: str
    timeout_seconds: int


@dataclass(frozen=True)
class ChatConfig:
    """Phase 2 (chat serving) model/endpoint -- placeholder until the
    cluster is reachable and a real model choice is validated."""

    model: str
    host: str
    timeout_seconds: int


@dataclass(frozen=True)
class PersonaConfig:
    """Phase 4 (persona LoRA) cadence -- not built yet."""

    cadence_days: int


@dataclass(frozen=True)
class StoreConfig:
    """Where the LanceDB vector store lives on disk."""

    path: str


@dataclass(frozen=True)
class ServeConfig:
    """rag_serve.py's HTTP retrieval endpoint -- LAN-only, see its module
    docstring for why this is never Funneled/exposed publicly."""

    host: str
    port: int


@dataclass(frozen=True)
class RagConfig:  # pylint: disable=too-many-instance-attributes
    """The full validated config/rag.yaml, one attribute per top-level
    YAML section -- intentionally flat rather than nested further, since
    every section is already its own small dataclass."""

    chunking: ChunkingConfig
    retention: RetentionConfig
    retrieval: RetrievalConfig
    embedding: EmbeddingConfig
    chat: ChatConfig
    persona: PersonaConfig
    store: StoreConfig
    serve: ServeConfig


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> RagConfig:
    """Load and validate a RAG config YAML file. Raises RagConfigError on
    anything missing or malformed -- never returns a partially-valid
    config, so callers don't need their own None-checking."""
    path = Path(path)
    if not path.exists():
        raise RagConfigError(f"config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RagConfigError(f"{path} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise RagConfigError(f"{path} must contain a top-level mapping")

    missing = [section for section in _REQUIRED_SECTIONS if section not in raw]
    if missing:
        raise RagConfigError(f"{path} is missing required section(s): {', '.join(missing)}")

    try:
        return RagConfig(
            chunking=ChunkingConfig(**raw["chunking"]),
            retention=RetentionConfig(**raw["retention"]),
            retrieval=RetrievalConfig(**raw["retrieval"]),
            embedding=EmbeddingConfig(**raw["embedding"]),
            chat=ChatConfig(**raw["chat"]),
            persona=PersonaConfig(**raw["persona"]),
            store=StoreConfig(**raw["store"]),
            serve=ServeConfig(**raw["serve"]),
        )
    except TypeError as exc:
        raise RagConfigError(f"{path} has a malformed section: {exc}") from exc
