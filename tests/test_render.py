"""Tests for the HTML rendering stage."""

from datetime import datetime, timezone

import pytest

from newshelper.models import ArticleRecommendation, BookRecommendation, EnrichedStory, HeadlineCandidate, Story
from newshelper.render import render_html


def make_enriched(title: str, with_extras: bool = False, is_satire: bool = False) -> EnrichedStory:
    candidate = HeadlineCandidate(title=title, link="https://bbc.example/x", source="bbc", published="")
    story = Story(title=title, candidates=[candidate], is_satire=is_satire)
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


def test_render_html_shows_satire_badge_when_tagged():
    lead = make_enriched("A satirical headline", is_satire=True)
    html = render_html([lead], build_date=datetime(2026, 7, 21, tzinfo=timezone.utc))
    assert "SATIRE" in html


def test_render_html_omits_satire_badge_when_not_tagged():
    lead = make_enriched("A regular headline", is_satire=False)
    html = render_html([lead], build_date=datetime(2026, 7, 21, tzinfo=timezone.utc))
    assert "SATIRE" not in html
