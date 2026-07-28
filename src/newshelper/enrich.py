"""Stage 3: ask the local model for a summary + follow-up topics, then verify."""

import json
import logging

from newshelper.books import verify_book_topic
from newshelper.factcheck import check_headline
from newshelper.models import ArticleRecommendation, EnrichedStory, HeadlineCandidate, Story
from newshelper.ollama_client import OllamaClientProtocol

logger = logging.getLogger(__name__)

# Narration mood for video.py -- keep this set small and the definitions
# concrete, since a model asked for open-ended "tone" tends to drift toward
# whatever adjective it saw most recently in the headline.
VALID_TONES = ("grave", "somber", "neutral", "upbeat")
DEFAULT_TONE = "neutral"

PROMPT_TEMPLATE = """You are helping build a daily news digest that explains \
the story behind a headline, not just the headline itself.

Headline: {title}
Reported by: {sources}

Respond with ONLY a JSON object with these keys:
- "summary": a 2-4 sentence plain-language explanation of the broader story \
(context a reader would need beyond the headline itself).
- "book_topics": a list of 1-2 short topic phrases (not titles) describing a \
nonfiction book subject that would help a reader understand this story in depth.
- "article_topics": a list of 1-2 short topic phrases for a follow-up article.
- "tone": the single best fit from exactly these four words for how a \
narrator reading this story aloud should sound -- "grave" (deaths, \
casualties, disasters, atrocities), "somber" (serious conflict, hardship, \
or controversy but no confirmed deaths), "neutral" (routine politics, \
business, science, procedural news), or "upbeat" (positive, lighthearted, \
or human-interest news). Pick exactly one of those four words.
"""


def build_prompt(story: Story) -> str:
    """Render the enrichment prompt for a single story."""
    return PROMPT_TEMPLATE.format(title=story.title, sources=", ".join(story.sources))


def parse_model_response(raw_response: str) -> dict:
    """Parse the model's JSON reply, tolerating minor formatting noise."""
    text = raw_response.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        logger.warning("could not parse model response as JSON: %r", raw_response)
        return {}


def build_source_citations(candidates: list[HeadlineCandidate]) -> list[ArticleRecommendation]:
    """List the actual RSS entries a story was built from -- the summary's sources.

    Purely structural: every candidate was already fetched and clustered, so
    no extra network/model call is needed to produce this list, and nothing
    here can be hallucinated.
    """
    return [
        ArticleRecommendation(
            title=f"{candidate.source}: {candidate.title}",
            url=candidate.link,
            kind="source",
        )
        for candidate in candidates
    ]


def enrich_story(story: Story, client: OllamaClientProtocol) -> EnrichedStory:
    """Summarize a story and attach verified book/article recommendations."""
    raw_response = client.generate(build_prompt(story))
    parsed = parse_model_response(raw_response)

    summary = parsed.get("summary", "").strip() or "Summary unavailable."

    tone = parsed.get("tone", DEFAULT_TONE)
    if tone not in VALID_TONES:
        if tone:
            logger.warning("model returned unrecognized tone %r; defaulting to %r", tone, DEFAULT_TONE)
        tone = DEFAULT_TONE

    books = []
    for topic in parsed.get("book_topics", []):
        recommendation = verify_book_topic(topic)
        if recommendation is not None:
            books.append(recommendation)

    articles = [
        ArticleRecommendation(title=topic, url=story.links[0] if story.links else "")
        for topic in parsed.get("article_topics", [])
    ]

    sourced_from = build_source_citations(story.candidates)
    fact_check = check_headline(story.title)

    return EnrichedStory(
        story=story,
        summary=summary,
        books=books,
        articles=articles,
        sourced_from=sourced_from,
        fact_check=fact_check,
        tone=tone,
    )


def enrich_all(stories: list[Story], client: OllamaClientProtocol) -> list[EnrichedStory]:
    """Enrich every story in the day's top-N list."""
    return [enrich_story(story, client) for story in stories]
