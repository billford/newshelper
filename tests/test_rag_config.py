"""Tests for the RAG config loader (rag_config.py) -- must reject
malformed config and never use an unsafe YAML loader, per ADR-003's
testing requirements."""

import pytest

from newshelper.rag_config import RagConfigError, load_config

VALID_CONFIG = """
chunking:
  chunk_size_words: 600
  overlap_words: 80
retention:
  current_window_days: 90
retrieval:
  top_k: 8
  current_weight: 0.7
embedding:
  model: nomic-embed-text
  host: http://localhost:11434
  timeout_seconds: 30
chat:
  model: llama3.1:8b
  host: http://localhost:11434
  timeout_seconds: 60
persona:
  cadence_days: 7
store:
  path: data/rag_store
serve:
  host: 0.0.0.0
  port: 8901
"""


def test_load_config_reads_the_real_default_file():
    config = load_config()
    assert config.chunking.chunk_size_words > 0
    assert config.retention.current_window_days > 0
    assert config.embedding.model
    assert config.store.path


def test_load_valid_config_from_a_temp_file(tmp_path):
    path = tmp_path / "rag.yaml"
    path.write_text(VALID_CONFIG, encoding="utf-8")
    config = load_config(path)

    assert config.chunking.chunk_size_words == 600
    assert config.chunking.overlap_words == 80
    assert config.retention.current_window_days == 90
    assert config.retrieval.top_k == 8
    assert config.retrieval.current_weight == 0.7
    assert config.embedding.model == "nomic-embed-text"
    assert config.chat.model == "llama3.1:8b"
    assert config.persona.cadence_days == 7
    assert config.store.path == "data/rag_store"
    assert config.serve.host == "0.0.0.0"
    assert config.serve.port == 8901


def test_missing_file_raises_rag_config_error(tmp_path):
    with pytest.raises(RagConfigError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_malformed_yaml_raises_rag_config_error(tmp_path):
    path = tmp_path / "rag.yaml"
    path.write_text("chunking: [this is not: a valid: mapping", encoding="utf-8")
    with pytest.raises(RagConfigError):
        load_config(path)


def test_non_mapping_top_level_raises(tmp_path):
    path = tmp_path / "rag.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(RagConfigError):
        load_config(path)


def test_missing_required_section_raises(tmp_path):
    path = tmp_path / "rag.yaml"
    path.write_text("chunking:\n  chunk_size_words: 600\n  overlap_words: 80\n", encoding="utf-8")
    with pytest.raises(RagConfigError):
        load_config(path)


def test_malformed_section_raises(tmp_path):
    # "embedding" missing its required "model" field.
    bad = VALID_CONFIG.replace("  model: nomic-embed-text\n", "")
    path = tmp_path / "rag.yaml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(RagConfigError):
        load_config(path)


def test_yaml_arbitrary_tag_execution_is_rejected(tmp_path):
    """A safe loader must refuse Python-object tags (e.g. !!python/object)
    rather than silently constructing arbitrary objects -- this is the
    concrete difference between yaml.safe_load and yaml.load."""
    path = tmp_path / "rag.yaml"
    path.write_text(
        "chunking: !!python/object:builtins.object {}\n"
        "retention:\n  current_window_days: 90\n"
        "retrieval:\n  top_k: 8\n  current_weight: 0.7\n"
        "embedding:\n  model: x\n  host: http://x\n  timeout_seconds: 1\n"
        "chat:\n  model: x\n  host: http://x\n  timeout_seconds: 1\n"
        "persona:\n  cadence_days: 7\n"
        "store:\n  path: x\n"
        "serve:\n  host: 0.0.0.0\n  port: 8901\n",
        encoding="utf-8",
    )
    with pytest.raises(RagConfigError):
        load_config(path)
