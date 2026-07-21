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
    """A book suggestion, only ever populated after API verification."""

    title: str
    author: str
    url: str
    verified_via: str  # "Open Library" or "Google Books"


@dataclass
class ArticleRecommendation:
    """A follow-up article/link suggestion (not verified, just carried through)."""

    title: str
    url: str


@dataclass
class EnrichedStory:
    """A story plus AI-generated summary and verified deeper-reading list."""

    story: Story
    summary: str
    books: list[BookRecommendation] = field(default_factory=list)
    articles: list[ArticleRecommendation] = field(default_factory=list)
