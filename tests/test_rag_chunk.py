"""Tests for the RAG chunker (rag_chunk.py) -- boundary conditions and
overlap math, per ADR-003's testing requirements."""

import pytest

from newshelper.rag_chunk import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", chunk_size_words=10, overlap_words=2) == []


def test_whitespace_only_text_returns_no_chunks():
    assert chunk_text("   \n\t  ", chunk_size_words=10, overlap_words=2) == []


def test_short_text_returns_single_unsplit_chunk():
    text = "a short sentence about a story"
    assert chunk_text(text, chunk_size_words=50, overlap_words=5) == [text]


def test_text_exactly_at_chunk_size_returns_single_chunk():
    words = [f"w{i}" for i in range(10)]
    text = " ".join(words)
    assert chunk_text(text, chunk_size_words=10, overlap_words=2) == [text]


def test_text_longer_than_chunk_size_splits_with_overlap():
    words = [f"w{i}" for i in range(25)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size_words=10, overlap_words=3)

    assert len(chunks) > 1
    # First chunk is exactly the first 10 words.
    assert chunks[0] == " ".join(words[0:10])
    # Second chunk starts 7 words in (step = chunk_size - overlap = 7),
    # so it overlaps the tail of the first chunk by 3 words.
    assert chunks[1] == " ".join(words[7:17])


def test_last_chunk_reaches_the_end_of_the_text_exactly_once():
    words = [f"w{i}" for i in range(22)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size_words=10, overlap_words=2)

    assert chunks[-1].split()[-1] == "w21"
    # No chunk is emitted twice for the same tail.
    assert len(chunks) == len(set(chunks))


def test_zero_overlap_produces_contiguous_non_overlapping_chunks():
    words = [f"w{i}" for i in range(20)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size_words=10, overlap_words=0)

    assert chunks == [" ".join(words[0:10]), " ".join(words[10:20])]


def test_non_positive_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size_words=0, overlap_words=0)
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size_words=-5, overlap_words=0)


def test_overlap_equal_to_or_larger_than_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size_words=5, overlap_words=5)
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size_words=5, overlap_words=6)


def test_negative_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size_words=5, overlap_words=-1)
