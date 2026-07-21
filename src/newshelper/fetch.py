"""Stage 1: pull headline candidates from configured RSS feeds."""

import logging

import feedparser

from newshelper.config import FEEDS
from newshelper.models import HeadlineCandidate

logger = logging.getLogger(__name__)


def fetch_feed(source: str, url: str) -> list[HeadlineCandidate]:
    """Parse a single RSS feed into headline candidates.

    Network/parse failures are logged and yield an empty list rather than
    raising, so one bad feed can't take down the whole daily build.
    """
    parsed = feedparser.parse(url)
    if parsed.bozo and not parsed.entries:
        logger.warning("feed %s (%s) failed to parse: %s", source, url, parsed.bozo_exception)
        return []

    candidates = []
    for entry in parsed.entries:
        title = getattr(entry, "title", "").strip()
        link = getattr(entry, "link", "").strip()
        if not title or not link:
            continue
        published = getattr(entry, "published", "") or getattr(entry, "updated", "")
        candidates.append(
            HeadlineCandidate(title=title, link=link, source=source, published=published)
        )
    return candidates


def fetch_all(feeds: dict[str, str] | None = None) -> list[HeadlineCandidate]:
    """Fetch every configured feed, returning the combined candidate pool."""
    feeds = FEEDS if feeds is None else feeds
    all_candidates: list[HeadlineCandidate] = []
    for source, url in feeds.items():
        all_candidates.extend(fetch_feed(source, url))
    return all_candidates
