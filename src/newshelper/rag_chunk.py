"""Splits text into overlapping word-count chunks for embedding (ADR-003).

Uses word count as a proxy for "tokens" -- accurate enough for sizing at
this project's scale and avoids pulling in a real tokenizer (tiktoken etc)
as a new dependency. Per ADR-003 Decision 2, the text being chunked here
is short (title + 2-4 sentence summary + citations), so in practice most
items produce a single chunk; the overlap/multi-chunk path exists for
correctness and so this doesn't need a redesign if longer text (e.g.
fetched article bodies) is ever added later.
"""


def chunk_text(text: str, chunk_size_words: int, overlap_words: int) -> list[str]:
    """Split text into chunks of at most chunk_size_words words, each
    overlapping the previous chunk by overlap_words words.

    Empty/whitespace-only text returns []. Text with chunk_size_words or
    fewer words returns a single chunk (the whole text, unsplit).
    """
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if overlap_words < 0 or overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be >= 0 and less than chunk_size_words")

    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size_words:
        return [" ".join(words)]

    chunks = []
    step = chunk_size_words - overlap_words
    start = 0
    while True:
        chunk_words = words[start : start + chunk_size_words]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size_words >= len(words):
            break
        start += step
    return chunks
