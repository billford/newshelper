"""Tests for the parts of video.py that are cheap to unit test without a
live Kokoro model / ffmpeg: the narration-pronunciation override, and the
guard that skips stories with no real summary."""

from unittest.mock import patch

from newshelper.enrich import SUMMARY_UNAVAILABLE
from newshelper.models import EnrichedStory, HeadlineCandidate, Story
from newshelper.video import apply_pronunciation_overrides, generate_all, has_narratable_summary


def test_apply_pronunciation_overrides_respells_known_words():
    text = "Oman hands Iran a Gulf-backed proposal for joint control over Hormuz."
    result = apply_pronunciation_overrides(text)
    assert "Oh-Mahn" in result
    assert "Hor-mooz" in result
    assert "Oman" not in result
    assert "Hormuz" not in result


def test_apply_pronunciation_overrides_is_case_insensitive():
    assert "Oh-Mahn" in apply_pronunciation_overrides("news from oman today")


def test_apply_pronunciation_overrides_respects_word_boundaries():
    text = "Omani officials met with Iranian counterparts."
    assert apply_pronunciation_overrides(text) == text


def test_apply_pronunciation_overrides_leaves_unrelated_text_untouched():
    text = "The Federal Reserve raised interest rates today."
    assert apply_pronunciation_overrides(text) == text


# --- skipping stories that have no real summary --------------------------


def _story(summary: str) -> EnrichedStory:
    candidate = HeadlineCandidate(
        title="Fed raises interest rates", link="https://bbc.example/1", source="bbc", published=""
    )
    story = Story(title=candidate.title, candidates=[candidate])
    return EnrichedStory(story=story, summary=summary)


def test_has_narratable_summary_accepts_a_real_summary():
    assert has_narratable_summary(_story("Central banks are trying to cool inflation."))


def test_has_narratable_summary_rejects_the_placeholder():
    assert not has_narratable_summary(_story(SUMMARY_UNAVAILABLE))


def test_has_narratable_summary_rejects_blank_and_whitespace():
    assert not has_narratable_summary(_story(""))
    assert not has_narratable_summary(_story("   \n "))


def test_generate_all_skips_a_story_with_no_summary(tmp_path):
    stories = [_story(SUMMARY_UNAVAILABLE), _story("A real summary.")]

    with patch("newshelper.video.make_story_video") as make_video:
        generate_all(stories, tmp_path / "video")

    # Only the story with a real summary is rendered at all -- the placeholder
    # one never reaches TTS, so nothing narrates "Summary unavailable." aloud.
    assert make_video.call_count == 1
    assert stories[0].video_path is None
    assert stories[1].video_path == "video/02-fed-raises-interest-rates.mp4"
