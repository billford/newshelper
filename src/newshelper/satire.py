"""Satire/parody outlet detection via a maintained domain allowlist.

v1 of misinformation handling. Deliberately does not ask any model (local
or Claude) to render a true/false verdict on a story -- an LLM judging
truthfulness is itself an ungrounded claim, and risks mislabeling a real
outlet's real story. Instead, known satire outlets are flagged by matching
their domain against a small, manually-curated allowlist kept in
`data/satire_domains.json`, separate from this matching logic so it's easy
to review and expand.

Flagged stories are tagged, never dropped: if something satirical is
trending enough to look like real news, surfacing that is more useful than
silently hiding it.
"""

import json
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SATIRE_DOMAINS_FILE = DATA_DIR / "satire_domains.json"


def load_satire_domains(path: Path = SATIRE_DOMAINS_FILE) -> frozenset[str]:
    """Load the satire domain allowlist from a JSON file, lowercased."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return frozenset(domain.strip().lower() for domain in data.get("domains", []) if domain.strip())


@lru_cache(maxsize=1)
def default_allowlist() -> frozenset[str]:
    """The repo's shipped satire domain list, cached for repeated lookups."""
    return load_satire_domains()


def extract_domain(url: str) -> str:
    """Extract the lowercase registrable hostname from a URL (no port, no 'www.')."""
    hostname = (urlparse(url).hostname or "").lower()
    if hostname.startswith("www."):
        hostname = hostname[len("www.") :]
    return hostname


def is_satire_domain(domain: str, allowlist: frozenset[str]) -> bool:
    """True if `domain` is, or is a subdomain of, an allowlisted satire domain."""
    domain = domain.strip().lower()
    if domain.startswith("www."):
        domain = domain[len("www.") :]
    return any(domain == entry or domain.endswith(f".{entry}") for entry in allowlist)


def is_satire_url(url: str, allowlist: frozenset[str]) -> bool:
    """True if the URL's source domain matches the satire allowlist."""
    return is_satire_domain(extract_domain(url), allowlist)
