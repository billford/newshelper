"""Tests for RSS feed fetching, using canned feedparser output (no network)."""

from unittest.mock import patch

import feedparser

from newshelper.fetch import fetch_all, fetch_feed

SAMPLE_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<title>Sample Feed</title>
<item><title>First headline</title><link>https://example.com/1</link><pubDate>Mon, 01 Jan 2026 00:00:00 GMT</pubDate></item>
<item><title>Second headline</title><link>https://example.com/2</link></item>
<item><link>https://example.com/no-title</link></item>
</channel></rss>
"""


def test_fetch_feed_parses_entries_and_skips_incomplete_ones():
    with patch("newshelper.fetch.feedparser.parse", return_value=feedparser.parse(SAMPLE_RSS)):
        candidates = fetch_feed("sample", "https://example.com/rss")

    assert len(candidates) == 2
    assert candidates[0].title == "First headline"
    assert candidates[0].source == "sample"
    assert candidates[0].link == "https://example.com/1"


def test_fetch_feed_returns_empty_list_on_parse_failure():
    broken = feedparser.parse("not xml at all")
    with patch("newshelper.fetch.feedparser.parse", return_value=broken):
        candidates = fetch_feed("broken", "https://example.com/rss")
    assert candidates == []


def test_fetch_all_combines_multiple_feeds():
    with patch("newshelper.fetch.feedparser.parse", return_value=feedparser.parse(SAMPLE_RSS)):
        candidates = fetch_all({"a": "https://a.example/rss", "b": "https://b.example/rss"})
    assert len(candidates) == 4
    assert {c.source for c in candidates} == {"a", "b"}
