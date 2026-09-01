"""Tests for the Ollama client's retry behaviour against a busy cluster."""

from unittest.mock import Mock, patch

import pytest
import requests

from newshelper.ollama_client import OllamaClient


def _response(status_code: int, payload: dict | None = None) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = status_code
    response.json.return_value = payload or {}
    if status_code >= 400:
        error = requests.HTTPError(f"{status_code} Server Error", response=response)
        response.raise_for_status.side_effect = error
    else:
        response.raise_for_status.return_value = None
    return response


def make_client(**kwargs) -> OllamaClient:
    # backoff 0 so the tests don't actually sleep
    return OllamaClient(max_attempts=3, backoff_seconds=0, **kwargs)


def test_retries_a_503_and_succeeds():
    responses = [_response(503), _response(200, {"response": "hello"})]
    with patch("newshelper.ollama_client.requests.post", side_effect=responses) as post:
        assert make_client().generate("prompt") == "hello"
    assert post.call_count == 2


def test_gives_up_after_max_attempts_and_raises():
    with patch("newshelper.ollama_client.requests.post", return_value=_response(503)) as post:
        with pytest.raises(requests.HTTPError):
            make_client().generate("prompt")
    assert post.call_count == 3


def test_does_not_retry_a_client_error():
    # A bad model name fails identically every time; retrying only wastes time.
    with patch("newshelper.ollama_client.requests.post", return_value=_response(404)) as post:
        with pytest.raises(requests.HTTPError):
            make_client().generate("prompt")
    assert post.call_count == 1


def test_retries_a_connection_error():
    outcomes = [requests.ConnectionError("reset"), _response(200, {"response": "ok"})]
    with patch("newshelper.ollama_client.requests.post", side_effect=outcomes) as post:
        assert make_client().generate("prompt") == "ok"
    assert post.call_count == 2


def test_returns_empty_string_when_payload_has_no_response_key():
    with patch("newshelper.ollama_client.requests.post", return_value=_response(200, {})):
        assert make_client().generate("prompt") == ""
