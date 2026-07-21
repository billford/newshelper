"""Tests for the enrichment stage, using a fake Ollama client (no live model)."""

import json
from unittest.mock import patch

from newshelper.enrich import enrich_story, parse_model_response
from newshelper.models import BookRecommendation, HeadlineCandidate, Story
from newshelper.ollama_client import FakeOllamaClient


def make_story() -> Story:
    candidate = HeadlineCandidate(
        title="Fed raises interest rates",
        link="https://bbc.example/1",
        source="bbc",
        published="",
    )
    return Story(title=candidate.title, candidates=[candidate])


def test_parse_model_response_extracts_json_even_with_surrounding_text():
    raw = 'Sure, here is the JSON:\n{"summary": "ok", "book_topics": []}\nEnjoy!'
    parsed = parse_model_response(raw)
    assert parsed == {"summary": "ok", "book_topics": []}


def test_parse_model_response_returns_empty_dict_on_garbage():
    assert parse_model_response("not json at all") == {}


def test_enrich_story_uses_summary_and_drops_unverified_books():
    client = FakeOllamaClient(
        json.dumps(
            {
                "summary": "Central banks are trying to cool inflation.",
                "book_topics": ["monetary policy history"],
                "article_topics": ["what is inflation"],
            }
        )
    )
    fake_book = BookRecommendation(
        title="The Alchemists", author="Neil Irwin", url="https://openlibrary.org/x", verified_via="Open Library"
    )
    with patch("newshelper.enrich.verify_book_topic", return_value=fake_book):
        result = enrich_story(make_story(), client)

    assert result.summary == "Central banks are trying to cool inflation."
    assert len(result.books) == 1
    assert result.books[0].verified_via == "Open Library"
    assert len(result.articles) == 1


def test_enrich_story_drops_book_when_verification_fails():
    client = FakeOllamaClient(
        json.dumps({"summary": "x", "book_topics": ["a made-up topic"], "article_topics": []})
    )
    with patch("newshelper.enrich.verify_book_topic", return_value=None):
        result = enrich_story(make_story(), client)
    assert result.books == []


def test_enrich_story_falls_back_to_placeholder_summary_when_missing():
    client = FakeOllamaClient(json.dumps({"book_topics": [], "article_topics": []}))
    result = enrich_story(make_story(), client)
    assert result.summary == "Summary unavailable."
