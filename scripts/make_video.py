"""Manual test harness for the local video-summary feature (src/newshelper/video.py).

Runs fetch -> rank -> enrich on today's real feeds (same as the daily build),
then renders one video per enriched story via video.generate_all -- the same
function build.py calls. Useful for iterating on the video pipeline without
running (and republishing) the full daily build:

    NEWSHELPER_OLLAMA_MODEL=qwen2.5:32b .venv/bin/python scripts/make_video.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from newshelper.enrich import enrich_all
from newshelper.fetch import fetch_all
from newshelper.ollama_client import OllamaClient
from newshelper.rank import top_stories
from newshelper.video import generate_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    candidates = fetch_all()
    stories = top_stories(candidates)
    enriched = enrich_all(stories, OllamaClient())
    if not enriched:
        logger.error("no enriched stories to render")
        return 1

    out_dir = Path("dist/video")
    generate_all(enriched, out_dir)

    rendered = [e for e in enriched if e.video_path]
    logger.info("done: %d/%d videos in %s", len(rendered), len(enriched), out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
