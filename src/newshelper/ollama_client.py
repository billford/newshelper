"""Thin client for a local Ollama server.

Kept separate and narrow so tests can inject a fake/mock implementation
instead of making live calls to wanderlust — the pipeline must stay
testable without that machine being on.
"""

import json
from dataclasses import dataclass
from typing import Protocol

import requests

from newshelper.config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS


class OllamaClientProtocol(Protocol):
    """Interface enrich.py depends on, so a fake can stand in during tests."""

    def generate(self, prompt: str) -> str:
        """Return the model's text response to a prompt."""


@dataclass
class OllamaClient:
    """Real client that calls a local Ollama /api/generate endpoint."""

    host: str = OLLAMA_HOST
    model: str = OLLAMA_MODEL
    timeout_seconds: int = OLLAMA_TIMEOUT_SECONDS

    def generate(self, prompt: str) -> str:
        """Send a prompt to the local model and return its raw text response."""
        response = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("response", "")


class FakeOllamaClient:
    """Deterministic stand-in for tests and offline development.

    Returns a canned JSON-ish response so enrich.py's parsing logic can be
    exercised without a live model.
    """

    def __init__(self, fixed_response: str | None = None):
        self._fixed_response = fixed_response

    def generate(self, prompt: str) -> str:
        """Return the fixed response, ignoring the prompt content."""
        del prompt
        if self._fixed_response is not None:
            return self._fixed_response
        return json.dumps(
            {
                "summary": "A placeholder summary generated without a live model.",
                "book_topics": ["a related nonfiction topic"],
                "article_topics": ["a related follow-up article"],
            }
        )
