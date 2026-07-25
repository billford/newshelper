"""Tests for the embedding client (rag_embed.py) -- mocked Ollama/Olla
responses, timeout/error handling, per ADR-003's testing requirements."""

from unittest.mock import Mock, patch

import pytest
import requests

from newshelper.rag_embed import FakeEmbedClient, OllamaEmbedClient


def test_fake_embed_client_returns_one_vector_per_text():
    client = FakeEmbedClient(dimension=4)
    vectors = client.embed(["hello", "world"])
    assert len(vectors) == 2
    assert all(len(v) == 4 for v in vectors)


def test_fake_embed_client_is_deterministic_for_the_same_text():
    client = FakeEmbedClient(dimension=4)
    assert client.embed(["same text"]) == client.embed(["same text"])


def test_embed_empty_list_makes_no_network_call():
    client = OllamaEmbedClient(host="http://localhost:11434", model="nomic-embed-text")
    with patch("newshelper.rag_embed.requests.post") as mock_post:
        result = client.embed([])
    assert result == []
    mock_post.assert_not_called()


def test_embed_success_returns_embeddings_in_order():
    client = OllamaEmbedClient(host="http://localhost:11434", model="nomic-embed-text")
    fake_response = Mock()
    fake_response.json.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
    fake_response.raise_for_status.return_value = None

    with patch("newshelper.rag_embed.requests.post", return_value=fake_response) as mock_post:
        result = client.embed(["a", "b"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    mock_post.assert_called_once()
    called_url = mock_post.call_args.args[0]
    assert called_url == "http://localhost:11434/api/embed"
    assert mock_post.call_args.kwargs["json"]["input"] == ["a", "b"]


def test_embed_raises_on_http_error():
    client = OllamaEmbedClient(host="http://localhost:11434", model="nomic-embed-text")
    fake_response = Mock()
    fake_response.raise_for_status.side_effect = requests.HTTPError("boom")

    with patch("newshelper.rag_embed.requests.post", return_value=fake_response):
        with pytest.raises(requests.HTTPError):
            client.embed(["a"])


def test_embed_raises_on_timeout():
    client = OllamaEmbedClient(host="http://localhost:11434", model="nomic-embed-text")
    with patch("newshelper.rag_embed.requests.post", side_effect=requests.Timeout("slow")):
        with pytest.raises(requests.Timeout):
            client.embed(["a"])


def test_embed_raises_on_mismatched_response_length():
    client = OllamaEmbedClient(host="http://localhost:11434", model="nomic-embed-text")
    fake_response = Mock()
    fake_response.json.return_value = {"embeddings": [[0.1, 0.2]]}  # only 1, for 2 inputs
    fake_response.raise_for_status.return_value = None

    with patch("newshelper.rag_embed.requests.post", return_value=fake_response):
        with pytest.raises(ValueError):
            client.embed(["a", "b"])


def test_embed_raises_when_embeddings_key_missing():
    client = OllamaEmbedClient(host="http://localhost:11434", model="nomic-embed-text")
    fake_response = Mock()
    fake_response.json.return_value = {}
    fake_response.raise_for_status.return_value = None

    with patch("newshelper.rag_embed.requests.post", return_value=fake_response):
        with pytest.raises(ValueError):
            client.embed(["a"])
