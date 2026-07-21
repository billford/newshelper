"""Tests for the HTML rendering stage."""

from datetime import datetime, timezone

import pytest

from newshelper.models import (
    ArticleRecommendation,
    BookRecommendation,
    EnrichedStory,
    FactCheckResult,
    HeadlineCandidate,
    Story,
)
from newshelper.render import render_about_html, render_html, write_site


def make_enriched(
    title: str,
    with_extras: bool = False,
    is_satire: bool = False,
    fact_check: FactCheckResult | None = None,
) -> EnrichedStory:
    candidate = HeadlineCandidate(title=title, link="https://bbc.example/x", source="bbc", published="")
    story = Story(title=title, candidates=[candidate], is_satire=is_satire)
    books = (
        [BookRecommendation(title="Some Book", author="An Author", url="https://openlibrary.org/x", verified_via="Open Library")]
        if with_extras
        else []
    )
    articles = [ArticleRecommendation(title="Related piece", url="https://bbc.example/y")] if with_extras else []
    sourced_from = (
        [ArticleRecommendation(title="bbc: " + title, url="https://bbc.example/x", kind="source")]
        if with_extras
        else []
    )
    return EnrichedStory(
        story=story,
        summary="A summary.",
        books=books,
        articles=articles,
        sourced_from=sourced_from,
        fact_check=fact_check,
    )


def test_render_html_includes_lead_story_and_rest_titles():
    lead = make_enriched("Lead headline", with_extras=True)
    rest = [make_enriched("Second headline"), make_enriched("Third headline")]
    html = render_html([lead, *rest], build_date=datetime(2026, 7, 21, tzinfo=timezone.utc))

    assert "Lead headline" in html
    assert "A summary." in html
    assert "Some Book" in html
    assert "verified via Open Library" in html
    assert "Second headline" in html
    assert "Third headline" in html


def test_render_html_gives_secondary_stories_full_summary_and_go_deeper_links():
    lead = make_enriched("Lead headline")
    second = make_enriched("Second headline", with_extras=True)
    html = render_html([lead, second], build_date=datetime(2026, 7, 21, tzinfo=timezone.utc))

    assert "Related piece" in html
    assert 'href="https://bbc.example/y"' in html
    assert "Some Book" in html
    assert "verified via Open Library" in html
    assert html.count("A summary.") == 2


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
    assert '<p class="satire-badge">SATIRE</p>' not in html


def test_render_html_tags_each_go_deeper_link_with_its_kind_and_book_disclaimer():
    lead = make_enriched("Lead headline", with_extras=True)
    html = render_html([lead], build_date=datetime(2026, 7, 21, tzinfo=timezone.utc))

    assert 'class="kind-tag kind-article"' in html
    assert 'class="kind-tag kind-book"' in html
    assert 'class="kind-tag kind-source"' in html
    assert "make no money from book sales" in html


def test_render_html_shows_fact_check_notice_with_caveat_when_present():
    fact_check = FactCheckResult(
        claim_text="The moon landing was faked",
        rating="False",
        publisher="Example Fact Checkers",
        url="https://factcheck.example/1",
    )
    lead = make_enriched("A disputed headline", fact_check=fact_check)
    html = render_html([lead], build_date=datetime(2026, 7, 21, tzinfo=timezone.utc))

    assert "FACT-CHECKED CLAIM NEARBY" in html
    assert "The moon landing was faked" in html
    assert "Example Fact Checkers" in html
    assert 'href="https://factcheck.example/1"' in html
    assert "not necessarily about the exact same story" in html


def test_render_html_omits_fact_check_notice_when_absent():
    lead = make_enriched("An undisputed headline", fact_check=None)
    html = render_html([lead], build_date=datetime(2026, 7, 21, tzinfo=timezone.utc))
    assert "FACT-CHECKED CLAIM NEARBY" not in html
    assert "factcheck-notice" not in html


def test_render_html_includes_a_legend_covering_every_current_tag_kind():
    lead = make_enriched("Lead headline", with_extras=True)
    html = render_html([lead], build_date=datetime(2026, 7, 21, tzinfo=timezone.utc))

    assert 'class="legend"' in html
    # Every tag kind currently rendered elsewhere on the page must have a
    # legend entry -- this is the standing rule from the template comment,
    # exercised as a test so a future tag addition without a legend update
    # fails loudly.
    for label in ("TOPIC", "SATIRE", "FACT-CHECKED", "ARTICLE", "BOOK", "SOURCE"):
        assert label in html


def test_render_html_includes_the_source_and_purpose_disclaimer():
    lead = make_enriched("Lead headline")
    html = render_html([lead], build_date=datetime(2026, 7, 21, tzinfo=timezone.utc))

    assert 'class="disclaimer"' in html
    assert "third-party outlets we do not control" in html
    assert "entertainment and educational purposes only" in html


def test_render_html_links_to_the_about_page():
    lead = make_enriched("Lead headline")
    html = render_html([lead], build_date=datetime(2026, 7, 21, tzinfo=timezone.utc))
    assert 'href="about.html"' in html


def test_render_about_html_explains_purpose_and_links_back():
    html = render_about_html(build_date=datetime(2026, 7, 21, tzinfo=timezone.utc))
    assert "Why NewsHelper exists" in html
    assert 'href="index.html"' in html
    assert "satire" in html.lower()
    assert "fact-check" in html.lower()


def test_render_about_html_includes_a_published_timestamp():
    html = render_about_html(build_date=datetime(2026, 7, 21, 14, 30, tzinfo=timezone.utc))
    assert "Last published" in html
    assert "Tuesday, July 21, 2026" in html
    assert "2:30 PM UTC" in html


def test_write_site_writes_both_index_and_about_pages(tmp_path):
    lead = make_enriched("Lead headline")
    out = write_site([lead], output_dir=str(tmp_path / "dist"))

    assert (out / "index.html").exists()
    assert (out / "about.html").exists()
    assert "Lead headline" in (out / "index.html").read_text(encoding="utf-8")
    assert "Why NewsHelper exists" in (out / "about.html").read_text(encoding="utf-8")
