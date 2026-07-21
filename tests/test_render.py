"""Tests for the HTML rendering stage."""

from datetime import datetime, timezone

import pytest

from newshelper.models import ArticleRecommendation, BookRecommendation, EnrichedStory, HeadlineCandidate, Story
from newshelper.render import render_html


def make_enriched(title: str, with_extras: bool = False) -> EnrichedStory:
    candidate = HeadlineCandidate(title=title, link="https://bbc.example/x", source="bbc", published="")
    story = Story(title=title, candidates=[candidate])
    books = (
        [BookRecommendation(title="Some Book", author="An Author", url="https://openlibrary.org/x", verified_via="Open Library")]
        if with_extras
        else []
    )
    articles = [ArticleRecommendation(title="Related piece", url="https://bbc.example/y")] if with_extras else []
    return EnrichedStory(story=story, summary="A summary.", books=books, articles=articles)


def test_render_html_includes_lead_story_and_condensed_rest():
    lead = make_enriched("Lead headline", with_extras=True)
    rest = [make_enriched("Second headline"), make_enriched("Third headline")]
    html = render_html([lead, *rest], build_date=datetime(2026, 7, 21, tzinfo=timezone.utc))

    assert "Lead headline" in html
    assert "A summary." in html
    assert "Some Book" in html
    assert "verified via Open Library" in html
    assert "Second headline" in html
    assert "Third headline" in html


def test_render_html_raises_on_empty_story_list():
    with pytest.raises(ValueError):
        render_html([])
