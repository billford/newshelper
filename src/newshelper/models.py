"""Shared data structures passed between pipeline stages."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HeadlineCandidate:
    """A single article/entry pulled from one RSS feed."""

    title: str
    link: str
    source: str
    published: str
    topic: str = "general"


@dataclass
class Story:
    """A cluster of candidates believed to be the same underlying story."""

    title: str
    candidates: list[HeadlineCandidate]
    topic: str = "general"
    is_satire: bool = False

    @property
    def score(self) -> int:
        """Cross-feed frequency: how many distinct sources reported this."""
        return len({c.source for c in self.candidates})

    @property
    def sources(self) -> list[str]:
        """Distinct source names that reported this story."""
        return sorted({c.source for c in self.candidates})

    @property
    def links(self) -> list[str]:
        """All candidate article links for this story."""
        return [c.link for c in self.candidates]


@dataclass
class BookRecommendation:
    """A book suggestion, only ever populated after API verification.

    `url` points to a Bookshop.org search for the verified title/author, not
    the verifying API itself -- plain search link, no affiliate ID, since we
    make no money from book sales. `kind` is a rendering discriminator so the
    "go deeper" template can tag every link with what type it is; a future
    "paper" recommendation kind would follow the same pattern.
    """

    title: str
    author: str
    url: str
    verified_via: str  # "Open Library" or "Google Books"
    kind: str = "book"


@dataclass
class ArticleRecommendation:
    """A follow-up article/link suggestion (not verified, just carried through)."""

    title: str
    url: str
    kind: str = "article"


@dataclass
class FactCheckResult:
    """A published fact-check of a claim judged similar enough to this story.

    Grounded misinformation signal (v2): unlike satire tagging, this never
    asserts a verdict of our own -- it surfaces an existing, real,
    independently-published fact-check with its own rating and link, and
    lets the reader judge fit and follow through themselves.
    """

    claim_text: str
    rating: str
    publisher: str
    url: str


@dataclass
class EnrichedStory:
    """A story plus AI-generated summary and verified deeper-reading list."""

    story: Story
    summary: str
    books: list[BookRecommendation] = field(default_factory=list)
    articles: list[ArticleRecommendation] = field(default_factory=list)
    sourced_from: list[ArticleRecommendation] = field(default_factory=list)
    fact_check: FactCheckResult | None = None
