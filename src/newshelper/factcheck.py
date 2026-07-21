"""Misinformation tagging v2: grounded fact-check lookups via Google's
Fact Check Claims Search API.

This is deliberately a different mechanism from satire.py's domain
allowlist. Satire matching is exact and free (a domain either is or isn't
on the list). Fact-check search is keyword-based against a real,
independently-published database of fact-checks -- broader coverage, but
imprecise: a search can return a claim that merely shares a word or two
with the headline, not the same story. To avoid mislabeling an unrelated
real story as "disputed", a match is only accepted when the returned
claim's text is similar enough to the headline (see
`rank.similarity`) -- and even then, this module never asserts its own
verdict. It surfaces the existing published rating, publisher, and link,
and lets the reader judge fit and follow through themselves.

Requires a Google Cloud API key (NEWSHELPER_FACTCHECK_API_KEY) with the
Fact Check Tools API enabled -- unlike the books APIs, this one is not
keyless. With no key configured, lookups are skipped entirely; this must
never fail a build.
"""

import logging

import requests

from newshelper.config import (
    FACT_CHECK_API_KEY,
    FACT_CHECK_API_URL,
    FACT_CHECK_SIMILARITY_THRESHOLD,
    FACT_CHECK_TIMEOUT_SECONDS,
)
from newshelper.models import FactCheckResult
from newshelper.rank import similarity

logger = logging.getLogger(__name__)


def _best_claim_review(claim: dict) -> dict | None:
    """Return the first claimReview entry that has both a rating and a URL."""
    for review in claim.get("claimReview", []):
        if review.get("textualRating") and review.get("url"):
            return review
    return None


def search_claims(query: str, api_key: str = FACT_CHECK_API_KEY) -> list[dict]:
    """Query the Fact Check Claims Search API; return the raw `claims` list.

    Returns an empty list on any error, missing key, or no results -- never
    raises, so a fact-check outage or quota limit can't take down a build.
    """
    if not api_key:
        return []

    try:
        response = requests.get(
            FACT_CHECK_API_URL,
            params={"query": query, "key": api_key},
            timeout=FACT_CHECK_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Fact Check API lookup failed for %r: %s", query, exc)
        return []

    return response.json().get("claims", [])


def check_headline(
    headline: str,
    threshold: float = FACT_CHECK_SIMILARITY_THRESHOLD,
    api_key: str = FACT_CHECK_API_KEY,
) -> FactCheckResult | None:
    """Look up a headline and return the best sufficiently-similar fact-check.

    Only the highest-similarity claim above `threshold` is returned, so a
    loosely related claim on a shared keyword never gets attached to a
    story it isn't actually about.
    """
    claims = search_claims(headline, api_key=api_key)

    best_result: FactCheckResult | None = None
    best_score = threshold
    for claim in claims:
        claim_text = claim.get("text", "")
        if not claim_text:
            continue
        score = similarity(headline, claim_text)
        if score < best_score:
            continue
        review = _best_claim_review(claim)
        if review is None:
            continue
        best_score = score
        best_result = FactCheckResult(
            claim_text=claim_text,
            rating=review["textualRating"],
            publisher=review.get("publisher", {}).get("name", "a fact-checker"),
            url=review["url"],
        )

    return best_result
