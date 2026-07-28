"""Tests for the narration-pronunciation override, the one part of video.py
that's cheap to unit test without a live Kokoro model / ffmpeg."""

from newshelper.video import apply_pronunciation_overrides


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
