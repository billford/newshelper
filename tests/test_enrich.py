"""Tests for the enrichment stage, using a fake Ollama client (no live model)."""

import json
from unittest.mock import patch

import pytest

from newshelper.enrich import (
    EnrichmentUnavailable,
    build_source_citations,
    enrich_all,
    enrich_story,
    parse_model_response,
)
from newshelper.models import BookRecommendation, FactCheckResult, HeadlineCandidate, Story
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


def test_enrich_story_cites_the_original_candidates_as_sources():
    client = FakeOllamaClient(json.dumps({"summary": "x", "book_topics": [], "article_topics": []}))
    result = enrich_story(make_story(), client)
    assert len(result.sourced_from) == 1
    assert result.sourced_from[0].kind == "source"
    assert result.sourced_from[0].url == "https://bbc.example/1"
    assert "bbc" in result.sourced_from[0].title


def test_build_source_citations_covers_every_candidate():
    candidates = [
        HeadlineCandidate(title="A", link="https://a.example/1", source="bbc", published=""),
        HeadlineCandidate(title="A", link="https://a.example/2", source="npr", published=""),
    ]
    citations = build_source_citations(candidates)
    assert len(citations) == 2
    assert {c.url for c in citations} == {"https://a.example/1", "https://a.example/2"}
    assert all(c.kind == "source" for c in citations)


def test_enrich_story_attaches_fact_check_result_when_present():
    client = FakeOllamaClient(json.dumps({"summary": "x", "book_topics": [], "article_topics": []}))
    fake_result = FactCheckResult(
        claim_text="Fed raises interest rates",
        rating="False",
        publisher="Example Fact Checkers",
        url="https://factcheck.example/1",
    )
    with patch("newshelper.enrich.check_headline", return_value=fake_result):
        result = enrich_story(make_story(), client)
    assert result.fact_check is fake_result


def test_enrich_story_leaves_fact_check_none_when_no_match():
    client = FakeOllamaClient(json.dumps({"summary": "x", "book_topics": [], "article_topics": []}))
    with patch("newshelper.enrich.check_headline", return_value=None):
        result = enrich_story(make_story(), client)
    assert result.fact_check is None


def test_enrich_story_uses_model_tone_when_valid():
    client = FakeOllamaClient(
        json.dumps({"summary": "x", "book_topics": [], "article_topics": [], "tone": "grave"})
    )
    result = enrich_story(make_story(), client)
    assert result.tone == "grave"


def test_enrich_story_defaults_tone_to_neutral_when_missing():
    client = FakeOllamaClient(json.dumps({"summary": "x", "book_topics": [], "article_topics": []}))
    result = enrich_story(make_story(), client)
    assert result.tone == "neutral"


def test_enrich_story_falls_back_to_neutral_on_unrecognized_tone():
    client = FakeOllamaClient(
        json.dumps({"summary": "x", "book_topics": [], "article_topics": [], "tone": "excited"})
    )
    result = enrich_story(make_story(), client)
    assert result.tone == "neutral"


# --- resilience when the GPU cluster is busy or down ----------------------


class ExplodingOllamaClient:
    """Stands in for a cluster returning 503 to every call."""

    def __init__(self, error: Exception | None = None):
        self._error = error or RuntimeError("503 Server Error: Service Unavailable")

    def generate(self, prompt: str) -> str:
        del prompt
        raise self._error


class FlakyOllamaClient:
    """Fails the first call, then behaves like a working model."""

    def __init__(self, good_response: str):
        self._good_response = good_response
        self.calls = 0

    def generate(self, prompt: str) -> str:
        del prompt
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("503 Server Error: Service Unavailable")
        return self._good_response


def test_enrich_all_keeps_going_when_one_story_fails():
    good = json.dumps({"summary": "a real summary", "book_topics": [], "article_topics": []})
    stories = [make_story(), make_story()]
    with patch("newshelper.enrich.check_headline", return_value=None):
        results = enrich_all(stories, FlakyOllamaClient(good))

    assert len(results) == 2
    summaries = sorted(r.summary for r in results)
    assert summaries == ["Summary unavailable.", "a real summary"]


def test_enrich_all_still_cites_sources_for_a_failed_story():
    stories = [make_story(), make_story()]
    good = json.dumps({"summary": "ok", "book_topics": [], "article_topics": []})
    with patch("newshelper.enrich.check_headline", return_value=None):
        results = enrich_all(stories, FlakyOllamaClient(good))

    failed = next(r for r in results if r.summary == "Summary unavailable.")
    assert len(failed.sourced_from) == 1
    assert failed.sourced_from[0].url == "https://bbc.example/1"
    assert failed.tone == "neutral"


def test_enrich_all_raises_when_every_story_fails():
    stories = [make_story(), make_story()]
    with patch("newshelper.enrich.check_headline", return_value=None):
        with pytest.raises(EnrichmentUnavailable):
            enrich_all(stories, ExplodingOllamaClient())


def test_enrich_all_handles_an_empty_story_list():
    assert enrich_all([], ExplodingOllamaClient()) == []
