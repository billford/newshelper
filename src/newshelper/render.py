"""Stage 4: render enriched stories to a static HTML page."""

import shutil
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from newshelper.config import DIST_DIR, SITE_TAGLINE, SITE_TITLE
from newshelper.models import EnrichedStory

TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


def get_environment() -> Environment:
    """Build the Jinja2 environment used to render the digest page."""
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )


def render_html(enriched_stories: list[EnrichedStory], build_date: datetime | None = None) -> str:
    """Render the day's enriched stories into the digest page's HTML."""
    build_date = build_date or datetime.now(timezone.utc)
    if not enriched_stories:
        raise ValueError("cannot render a digest with zero stories")

    env = get_environment()
    template = env.get_template("index.html.j2")
    return template.render(
        site_title=SITE_TITLE,
        site_tagline=SITE_TAGLINE,
        build_date=build_date,
        lead=enriched_stories[0],
        rest=enriched_stories[1:],
    )


def write_site(enriched_stories: list[EnrichedStory], output_dir: str = DIST_DIR) -> Path:
    """Render the page and write it, plus static assets, to the output directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    html = render_html(enriched_stories)
    (out / "index.html").write_text(html, encoding="utf-8")

    static_out = out / "static"
    if STATIC_DIR.exists():
        shutil.copytree(STATIC_DIR, static_out, dirs_exist_ok=True)

    return out
