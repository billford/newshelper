"""Tests for satire-domain matching. No live network/file-driven cases needed
beyond the fixed test allowlist -- the matcher only ever sees domain/URL
strings, never a fetched response.
"""

from newshelper.satire import extract_domain, is_satire_domain, is_satire_url

ALLOWLIST = frozenset({"theonion.com", "babylonbee.com", "thedailymash.co.uk"})


def test_is_satire_domain_exact_match():
    assert is_satire_domain("theonion.com", ALLOWLIST) is True


def test_is_satire_domain_subdomain_match():
    assert is_satire_domain("local.theonion.com", ALLOWLIST) is True


def test_is_satire_domain_case_insensitive():
    assert is_satire_domain("TheOnion.COM", ALLOWLIST) is True
    assert is_satire_domain("Local.TheOnion.com", ALLOWLIST) is True


def test_is_satire_domain_non_match():
    assert is_satire_domain("bbc.co.uk", ALLOWLIST) is False
    assert is_satire_domain("notthedailymash.co.uk", ALLOWLIST) is False


def test_is_satire_domain_does_not_match_substring_lookalike():
    # "theonion.com.evil.example" is not a subdomain of theonion.com.
    assert is_satire_domain("theonion.com.evil.example", ALLOWLIST) is False


def test_extract_domain_strips_scheme_path_and_www():
    assert extract_domain("https://www.theonion.com/some/article/path?x=1") == "theonion.com"
    assert extract_domain("http://theonion.com/") == "theonion.com"
    assert extract_domain("https://local.theonion.com") == "local.theonion.com"


def test_is_satire_url_trailing_slash_and_path_variants():
    assert is_satire_url("https://www.theonion.com/", ALLOWLIST) is True
    assert is_satire_url("https://theonion.com/politics/some-story-1234", ALLOWLIST) is True
    assert is_satire_url("https://local.theonion.com/entertainment/x", ALLOWLIST) is True


def test_is_satire_url_non_match():
    assert is_satire_url("https://www.bbc.co.uk/news/some-story", ALLOWLIST) is False
