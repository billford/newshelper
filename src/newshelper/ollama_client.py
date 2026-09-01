"""Thin client for a local Ollama server.

Kept separate and narrow so tests can inject a fake/mock implementation
instead of making live calls to wanderlust — the pipeline must stay
testable without that machine being on.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Protocol

import requests

from newshelper.config import (
    OLLAMA_HOST,
    OLLAMA_MAX_ATTEMPTS,
    OLLAMA_MODEL,
    OLLAMA_RETRY_BACKOFF_SECONDS,
    OLLAMA_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

# Worth another try: the cluster is busy or briefly unreachable, not wrong.
# A 4xx (bad model name, malformed request) will fail identically forever, so
# retrying it just delays a build that is going to fail anyway.
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


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
    max_attempts: int = OLLAMA_MAX_ATTEMPTS
    backoff_seconds: float = OLLAMA_RETRY_BACKOFF_SECONDS

    def generate(self, prompt: str) -> str:
        """Send a prompt to the model and return its raw text response.

        Retries a busy or unreachable cluster a few times before raising, so
        one transient 503 doesn't cost the caller a story.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._generate_once(prompt)
            except requests.HTTPError as error:
                status = error.response.status_code if error.response is not None else None
                if status not in _RETRYABLE_STATUS_CODES:
                    raise
                last_error = error
            except requests.RequestException as error:
                # Connection reset, DNS failure, read timeout -- all worth another go.
                last_error = error

            if attempt < self.max_attempts:
                delay = self.backoff_seconds * attempt
                logger.warning(
                    "ollama request failed (attempt %d/%d): %s; retrying in %.0fs",
                    attempt, self.max_attempts, last_error, delay,
                )
                time.sleep(delay)

        assert last_error is not None
        raise last_error

    def _generate_once(self, prompt: str) -> str:
        """Make a single /api/generate call."""
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
