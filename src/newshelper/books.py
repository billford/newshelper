"""Book-recommendation verification: Open Library primary, Google Books fallback.

Per ADR-001, the local model only ever suggests a *topic*, never a title
directly used in the output — every published book recommendation must
round-trip through one of these real APIs first, so hallucinated titles
never reach the site.
"""

import logging

import requests

from newshelper.config import (
    BOOKS_API_TIMEOUT_SECONDS,
    GOOGLE_BOOKS_SEARCH_URL,
    OPEN_LIBRARY_SEARCH_URL,
)
from newshelper.models import BookRecommendation

logger = logging.getLogger(__name__)


def search_open_library(topic: str) -> BookRecommendation | None:
    """Look up a topic in Open Library; return the first verified match."""
    try:
        response = requests.get(
            OPEN_LIBRARY_SEARCH_URL,
            params={"q": topic, "limit": 1},
            timeout=BOOKS_API_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Open Library lookup failed for %r: %s", topic, exc)
        return None

    docs = response.json().get("docs", [])
    if not docs:
        return None

    doc = docs[0]
    title = doc.get("title")
    authors = doc.get("author_name") or ["Unknown"]
    key = doc.get("key")
    if not title or not key:
        return None

    return BookRecommendation(
        title=title,
        author=authors[0],
        url=f"https://openlibrary.org{key}",
        verified_via="Open Library",
    )


def search_google_books(topic: str) -> BookRecommendation | None:
    """Fallback lookup against the Google Books API (free, keyless)."""
    try:
        response = requests.get(
            GOOGLE_BOOKS_SEARCH_URL,
            params={"q": topic, "maxResults": 1},
            timeout=BOOKS_API_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Google Books lookup failed for %r: %s", topic, exc)
        return None

    items = response.json().get("items", [])
    if not items:
        return None

    volume_info = items[0].get("volumeInfo", {})
    title = volume_info.get("title")
    authors = volume_info.get("authors") or ["Unknown"]
    info_link = volume_info.get("infoLink")
    if not title or not info_link:
        return None

    return BookRecommendation(
        title=title,
        author=authors[0],
        url=info_link,
        verified_via="Google Books",
    )


def verify_book_topic(topic: str) -> BookRecommendation | None:
    """Try Open Library, then Google Books; return None if neither has a match.

    A None result means the suggestion is dropped, not published unverified.
    """
    return search_open_library(topic) or search_google_books(topic)
