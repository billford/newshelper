"""Stage 2: cluster cross-feed candidates into stories and pick the top N.

Deliberately plain, deterministic code rather than a model call — per
ADR-001, this is the hardest part of the pipeline and errors here are
harder to catch than a slightly-off AI summary, so it stays inspectable
and unit-testable.
"""

from difflib import SequenceMatcher

from newshelper.config import TITLE_SIMILARITY_THRESHOLD, TOP_STORY_COUNT
from newshelper.models import HeadlineCandidate, Story


def normalize_title(title: str) -> str:
    """Lowercase and strip punctuation-heavy noise for comparison purposes."""
    return " ".join(title.lower().split())


def similarity(title_a: str, title_b: str) -> float:
    """Fuzzy similarity ratio between two titles, in [0.0, 1.0]."""
    return SequenceMatcher(None, normalize_title(title_a), normalize_title(title_b)).ratio()


def cluster_candidates(
    candidates: list[HeadlineCandidate],
    threshold: float = TITLE_SIMILARITY_THRESHOLD,
) -> list[Story]:
    """Group candidates whose titles are similar enough into single stories.

    Greedy single-pass clustering: each candidate joins the first existing
    story whose title is similar enough, else starts a new story. Good
    enough for a same-day batch of a few hundred headlines; not meant to
    scale to a general dedup problem.
    """
    stories: list[Story] = []
    for candidate in candidates:
        matched = False
        for story in stories:
            if similarity(candidate.title, story.title) >= threshold:
                story.candidates.append(candidate)
                matched = True
                break
        if not matched:
            stories.append(Story(title=candidate.title, candidates=[candidate]))
    return stories


def top_stories(
    candidates: list[HeadlineCandidate],
    count: int = TOP_STORY_COUNT,
) -> list[Story]:
    """Cluster candidates and return the `count` highest cross-feed-scored stories."""
    stories = cluster_candidates(candidates)
    stories.sort(key=lambda story: story.score, reverse=True)
    return stories[:count]
