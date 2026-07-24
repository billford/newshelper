"""Orchestrates the full daily pipeline: fetch -> rank -> enrich -> render.

Intended entrypoint for the wanderlust cron job:
    python -m newshelper.build
"""

import logging
import sys
from pathlib import Path

from newshelper.config import DIST_DIR
from newshelper.enrich import enrich_all
from newshelper.fetch import fetch_all
from newshelper.ollama_client import OllamaClient
from newshelper.rank import top_stories
from newshelper.render import write_site
from newshelper.video import generate_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run() -> int:
    """Run the full pipeline once. Returns a process exit code."""
    candidates = fetch_all()
    if not candidates:
        logger.error("no headline candidates fetched from any feed; aborting build")
        return 1
    logger.info("fetched %d candidates", len(candidates))

    stories = top_stories(candidates)
    if not stories:
        logger.error("no stories survived ranking; aborting build")
        return 1
    logger.info("selected %d top stories", len(stories))

    enriched = enrich_all(stories, OllamaClient())

    try:
        generate_all(enriched, Path(DIST_DIR) / "video")
    except Exception:
        logger.exception("video generation step failed; continuing without videos")

    output_dir = write_site(enriched)
    logger.info("site written to %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(run())
