"""Central configuration: feed list, model choice, API endpoints.

Kept as plain data (no env-var magic beyond simple overrides) so tests can
import and override values directly without patching the environment.
"""

import os

# --- RSS feed sources -------------------------------------------------
# v1 set per ADR-001: Google News topic feeds + Google Trends + BBC + NPR.
# AP/Reuters intentionally excluded until their feed status is verified.
FEEDS: dict[str, str] = {
    "google-news-top": "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
    "google-news-world": (
        "https://news.google.com/rss/headlines/section/topic/WORLD"
        "?hl=en-US&gl=US&ceid=US:en"
    ),
    "google-news-technology": (
        "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY"
        "?hl=en-US&gl=US&ceid=US:en"
    ),
    "google-news-business": (
        "https://news.google.com/rss/headlines/section/topic/BUSINESS"
        "?hl=en-US&gl=US&ceid=US:en"
    ),
    "google-trends": "https://trends.google.com/trending/rss?geo=US",
    "bbc-news": "https://feeds.bbci.co.uk/news/rss.xml",
    "npr-news": "https://feeds.npr.org/1001/rss.xml",
}

# --- Ranking ------------------------------------------------------------
TOP_STORY_COUNT = 6
# Titles are clustered as "the same story" when their fuzzy similarity
# meets this threshold (see rank.py:similarity).
TITLE_SIMILARITY_THRESHOLD = 0.55

# --- Local model (Ollama) ------------------------------------------------
OLLAMA_HOST = os.environ.get("NEWSHELPER_OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("NEWSHELPER_OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TIMEOUT_SECONDS = 120

# --- Book verification APIs ----------------------------------------------
OPEN_LIBRARY_SEARCH_URL = "https://openlibrary.org/search.json"
GOOGLE_BOOKS_SEARCH_URL = "https://www.googleapis.com/books/v1/volumes"
BOOKS_API_TIMEOUT_SECONDS = 15

# --- Fact-check verification (misinformation tagging v2) ------------------
# Requires a Google Cloud API key with the "Fact Check Tools API" enabled --
# unlike the books APIs above, this one is not keyless. Set
# NEWSHELPER_FACTCHECK_API_KEY in wanderlust's local environment (never in
# git). Left blank, fact-check lookups are skipped entirely -- never a
# build failure.
FACT_CHECK_API_KEY = os.environ.get("NEWSHELPER_FACTCHECK_API_KEY", "")
FACT_CHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
FACT_CHECK_TIMEOUT_SECONDS = 15
# A returned claim only counts as "about this story" if its text is at least
# this similar to the headline (same difflib measure as rank.py's title
# clustering) -- fact-check search is keyword-based and can otherwise
# surface an unrelated claim that merely shares a word or two.
FACT_CHECK_SIMILARITY_THRESHOLD = 0.4

# --- Output ---------------------------------------------------------------
DIST_DIR = "dist"
SITE_TITLE = "NewsHelper"
SITE_TAGLINE = "& the rest of the story"
