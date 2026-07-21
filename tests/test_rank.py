"""Tests for cross-feed clustering and top-story selection."""

from newshelper.models import HeadlineCandidate
from newshelper.rank import cluster_candidates, similarity, top_stories


def make_candidate(title: str, source: str) -> HeadlineCandidate:
    return HeadlineCandidate(title=title, link=f"https://{source}.example/x", source=source, published="")


def test_similarity_identical_titles_is_one():
    assert similarity("Fed raises rates", "Fed raises rates") == 1.0


def test_similarity_unrelated_titles_is_low():
    assert similarity("Fed raises interest rates again", "Local bakery wins award") < 0.4


def test_cluster_groups_near_duplicate_titles_across_sources():
    candidates = [
        make_candidate("Fed raises interest rates by half a point", "bbc"),
        make_candidate("Fed raises interest rates half a point", "npr"),
        make_candidate("Local bakery wins national award", "bbc"),
    ]
    stories = cluster_candidates(candidates)
    assert len(stories) == 2
    fed_story = next(s for s in stories if "Fed" in s.title)
    assert fed_story.score == 2
    assert set(fed_story.sources) == {"bbc", "npr"}


def test_top_stories_ranks_by_cross_feed_score_and_respects_count():
    candidates = [
        make_candidate("Senate passes new budget bill", "bbc"),
        make_candidate("Senate passes new budget bill", "npr"),
        make_candidate("Senate passes new budget bill", "google-news-top"),
        make_candidate("Local zoo welcomes baby giraffe", "bbc"),
        make_candidate("Championship game ends in overtime thriller", "bbc"),
    ]
    stories = top_stories(candidates, count=2)
    assert len(stories) == 2
    assert stories[0].title == "Senate passes new budget bill"
    assert stories[0].score == 3


def test_top_stories_empty_input_returns_empty_list():
    assert top_stories([]) == []


def test_cluster_tags_story_as_satire_when_source_matches_allowlist():
    satire_domains = frozenset({"theonion.com"})
    candidates = [
        HeadlineCandidate(
            title="Area man declares victory",
            link="https://www.theonion.com/area-man-declares-victory",
            source="google-news-top",
            published="",
        ),
    ]
    stories = cluster_candidates(candidates, satire_domains=satire_domains)
    assert len(stories) == 1
    assert stories[0].is_satire is True


def test_cluster_does_not_tag_non_satire_story():
    satire_domains = frozenset({"theonion.com"})
    candidates = [make_candidate("Senate passes new budget bill", "bbc")]
    stories = cluster_candidates(candidates, satire_domains=satire_domains)
    assert stories[0].is_satire is False


def test_cluster_tags_story_satire_even_if_only_one_of_several_sources_matches():
    satire_domains = frozenset({"theonion.com"})
    candidates = [
        make_candidate("City council approves new budget", "npr"),
        HeadlineCandidate(
            title="City council approves new budget",
            link="https://www.theonion.com/city-council-approves-new-budget",
            source="google-news-top",
            published="",
        ),
    ]
    stories = cluster_candidates(candidates, satire_domains=satire_domains)
    assert len(stories) == 1
    assert stories[0].is_satire is True
    assert stories[0].score == 2
