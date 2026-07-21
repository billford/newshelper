"""Tests for book-topic verification against Open Library / Google Books."""

from unittest.mock import Mock, patch

import requests

from newshelper.books import search_google_books, search_open_library, verify_book_topic


def _mock_response(json_data):
    mock = Mock()
    mock.json.return_value = json_data
    mock.raise_for_status.return_value = None
    return mock


def test_search_open_library_returns_recommendation_on_match():
    payload = {
        "docs": [{"title": "The Alchemists", "author_name": ["Neil Irwin"], "key": "/works/OL123W"}]
    }
    with patch("newshelper.books.requests.get", return_value=_mock_response(payload)):
        result = search_open_library("central banking history")
    assert result is not None
    assert result.title == "The Alchemists"
    assert result.verified_via == "Open Library"
    assert result.url == "https://openlibrary.org/works/OL123W"


def test_search_open_library_returns_none_on_no_docs():
    with patch("newshelper.books.requests.get", return_value=_mock_response({"docs": []})):
        assert search_open_library("nonexistent topic") is None


def test_search_open_library_returns_none_on_network_error():
    with patch("newshelper.books.requests.get", side_effect=requests.RequestException("boom")):
        assert search_open_library("anything") is None


def test_search_google_books_returns_recommendation_on_match():
    payload = {
        "items": [
            {
                "volumeInfo": {
                    "title": "Poor Economics",
                    "authors": ["Banerjee", "Duflo"],
                    "infoLink": "https://books.google.com/books?id=abc",
                }
            }
        ]
    }
    with patch("newshelper.books.requests.get", return_value=_mock_response(payload)):
        result = search_google_books("development economics")
    assert result is not None
    assert result.verified_via == "Google Books"


def test_verify_book_topic_falls_back_to_google_books_when_open_library_misses():
    with patch("newshelper.books.search_open_library", return_value=None):
        with patch("newshelper.books.search_google_books") as mock_google:
            mock_google.return_value = "sentinel"
            assert verify_book_topic("some topic") == "sentinel"
            mock_google.assert_called_once()


def test_verify_book_topic_returns_none_when_both_miss():
    with patch("newshelper.books.search_open_library", return_value=None):
        with patch("newshelper.books.search_google_books", return_value=None):
            assert verify_book_topic("some topic") is None
