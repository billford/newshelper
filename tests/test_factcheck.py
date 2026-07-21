"""Tests for grounded fact-check lookups (misinformation tagging v2).

No live network/API key needed -- requests.get is mocked throughout, per
the same pattern as test_books.py.
"""

from unittest.mock import Mock, patch

import requests

from newshelper.factcheck import check_headline, search_claims


def _mock_response(json_data):
    mock = Mock()
    mock.json.return_value = json_data
    mock.raise_for_status.return_value = None
    return mock


def _claim(text, rating="False", publisher="Example Fact Checkers", url="https://factcheck.example/1"):
    return {
        "text": text,
        "claimReview": [
            {
                "textualRating": rating,
                "publisher": {"name": publisher},
                "url": url,
            }
        ],
    }


def test_search_claims_returns_empty_list_without_api_key():
    # No requests.get patch needed -- a missing key must short-circuit before
    # any network call is attempted.
    with patch("newshelper.factcheck.requests.get") as mock_get:
        result = search_claims("some headline", api_key="")
    assert result == []
    mock_get.assert_not_called()


def test_search_claims_returns_empty_list_on_network_error():
    with patch("newshelper.factcheck.requests.get", side_effect=requests.RequestException("boom")):
        assert search_claims("some headline", api_key="fake-key") == []


def test_check_headline_tags_a_highly_similar_claim():
    payload = {"claims": [_claim("The moon landing was faked by NASA in a studio")]}
    with patch("newshelper.factcheck.requests.get", return_value=_mock_response(payload)):
        result = check_headline(
            "The moon landing was faked by NASA in a studio", api_key="fake-key"
        )
    assert result is not None
    assert result.rating == "False"
    assert result.publisher == "Example Fact Checkers"
    assert result.url == "https://factcheck.example/1"


def test_check_headline_ignores_loosely_related_claim_below_threshold():
    # Shares only a couple of common words with the headline -- must not be
    # treated as "about this story."
    payload = {"claims": [_claim("A celebrity chef's restaurant closed last year")]}
    with patch("newshelper.factcheck.requests.get", return_value=_mock_response(payload)):
        result = check_headline("Senate passes new budget bill after weekend vote", api_key="fake-key")
    assert result is None


def test_check_headline_skips_claims_missing_a_rating_or_url():
    payload = {"claims": [{"text": "Senate passes new budget bill", "claimReview": []}]}
    with patch("newshelper.factcheck.requests.get", return_value=_mock_response(payload)):
        result = check_headline("Senate passes new budget bill", api_key="fake-key")
    assert result is None


def test_check_headline_picks_the_most_similar_of_multiple_claims():
    payload = {
        "claims": [
            _claim("A budget bill passed somewhere once", rating="Mostly False", url="https://factcheck.example/loose"),
            _claim("Senate passes new budget bill after weekend vote", rating="False", url="https://factcheck.example/tight"),
        ]
    }
    with patch("newshelper.factcheck.requests.get", return_value=_mock_response(payload)):
        result = check_headline("Senate passes new budget bill after weekend vote", api_key="fake-key")
    assert result is not None
    assert result.url == "https://factcheck.example/tight"


def test_check_headline_returns_none_when_no_claims_found():
    with patch("newshelper.factcheck.requests.get", return_value=_mock_response({"claims": []})):
        assert check_headline("Senate passes new budget bill", api_key="fake-key") is None


def test_satire_and_fact_check_are_independent_signals():
    """A satire-domain story and a fact-checked-false story are distinct
    concerns -- this module never looks at domains, and satire.py never
    looks at fact-check ratings. Confirms the two mechanisms don't get
    conflated: matching one says nothing about the other.
    """
    from newshelper.satire import is_satire_domain  # pylint: disable=import-outside-toplevel

    satire_domains = frozenset({"theonion.com"})
    # A real (non-satire) domain, headline text matching a fact-checked claim.
    assert is_satire_domain("bbc.co.uk", satire_domains) is False

    payload = {"claims": [_claim("Vaccines contain microchips for tracking")]}
    with patch("newshelper.factcheck.requests.get", return_value=_mock_response(payload)):
        result = check_headline("Vaccines contain microchips for tracking", api_key="fake-key")
    assert result is not None
    # The fact-check result carries no notion of satire at all.
    assert not hasattr(result, "is_satire")
