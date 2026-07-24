"""Local test pipeline: turn an enriched story into a short (10-20s) summary
video -- narrated title + summary card, rendered entirely on-machine.

Three steps, each independently testable:
    1. narrate()   -- Piper TTS, text -> wav
    2. render_card() -- PIL, title/summary -> portrait PNG
    3. assemble()  -- ffmpeg, image + audio -> mp4 (Ken Burns zoom)

make_story_video() chains all three. This is a standalone experiment, not
yet wired into build.py -- run via scripts/make_video.py.
"""

import logging
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from newshelper.config import (
    PIPER_MODEL_PATH,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_MAX_SECONDS,
    VIDEO_WIDTH,
)
from newshelper.models import EnrichedStory

logger = logging.getLogger(__name__)

# Google News RSS titles are formatted "Headline - Publisher"; strip that
# suffix so it doesn't get read aloud or printed on the card. Conservative on
# purpose (short, title-cased tail only) so it doesn't eat a headline that
# legitimately ends in " - something".
_SOURCE_SUFFIX_RE = re.compile(r"\s[-–—]\s([A-Z][\w.&'’]*(?:\s[A-Z][\w.&'’]*){0,4})$")


def strip_source_suffix(title: str) -> str:
    """Remove a trailing " - Publisher Name" suffix, if present."""
    match = _SOURCE_SUFFIX_RE.search(title)
    if match and len(match.group(1)) <= 40:
        return title[: match.start()]
    return title


def slugify(title: str) -> str:
    """Filesystem-safe slug for video filenames, e.g. dist/video/01-<slug>.mp4."""
    keep = "".join(c if c.isalnum() or c == " " else "" for c in title.lower())
    return "-".join(keep.split())[:60]

SYSTEM_FONT_DIR = Path("/System/Library/Fonts")
SUPPLEMENTAL_FONT_DIR = Path("/System/Library/Fonts/Supplemental")
# Site branding (static/css/style.css, static/brand/logo.svg) uses Lora
# (serif headings) and Inter (sans body) loaded as webfonts -- neither is
# installed locally, so we use the same fallback families the site's own
# CSS declares next in its font stack: Georgia for serif, San Francisco
# (system default) for sans.
TITLE_FONT_PATH = SUPPLEMENTAL_FONT_DIR / "Georgia Bold.ttf"
TAGLINE_FONT_PATH = SUPPLEMENTAL_FONT_DIR / "Georgia Italic.ttf"
BODY_FONT_PATH = SYSTEM_FONT_DIR / "SFNS.ttf"

# Brand palette -- must match static/css/style.css's :root custom properties
# and static/brand/logo.svg exactly, not an approximation.
PAPER = (251, 249, 244)  # --paper
INK = (38, 36, 32)  # --ink
ACCENT = (138, 106, 74)  # --accent
RULE = (216, 205, 184)  # --rule
MUTED = (122, 117, 104)  # tagline gray, logo.svg's "#7A7568"

# VIDEO_MAX_SECONDS/MIN_SECONDS describe the *typical* case, not a hard
# limit -- title + full summary is always narrated in full (never cut
# mid-sentence, never dropped), so a story with a long summary simply
# produces a longer video. See ADR pending for the tradeoff.


def build_narration_script(enriched: EnrichedStory) -> str:
    """Title + full summary, verbatim."""
    title = strip_source_suffix(enriched.story.title).rstrip(".")
    return f"{title}. {enriched.summary.strip()}"


def narrate(text: str, out_wav: Path) -> Path:
    """Synthesize `text` to a wav file with Piper. Requires PIPER_MODEL_PATH
    to exist locally (see data/piper_voices/ -- not committed, download
    on-demand)."""
    from piper import PiperVoice  # local import: heavy, optional dependency

    model_path = Path(PIPER_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Piper voice model not found at {model_path}. Download it first, e.g.:\n"
            "  curl -sL -o data/piper_voices/en_US-lessac-medium.onnx "
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx\n"
            "  curl -sL -o data/piper_voices/en_US-lessac-medium.onnx.json "
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
        )

    voice = PiperVoice.load(str(model_path))
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_wav), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    return out_wav


def wav_duration_seconds(wav_path: Path) -> float:
    """Duration of a wav file in seconds, via the stdlib wave module."""
    with wave.open(str(wav_path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Greedy word-wrap text to fit max_width pixels for the given font."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if draw.textlength(candidate, font=font) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def _draw_masthead(draw: ImageDraw.ImageDraw, center_x: int, top: int, width: int) -> int:
    """Draw the NewsHelper wordmark (mirrors static/brand/logo.svg's double
    rule + two-tone title + italic tagline). Returns the y-coordinate just
    below the masthead."""
    wordmark_font = ImageFont.truetype(str(TITLE_FONT_PATH), 54)
    tagline_font = ImageFont.truetype(str(TAGLINE_FONT_PATH), 22)

    rule_inset = width // 2 - 40
    draw.rectangle([center_x - rule_inset, top, center_x + rule_inset, top + 2], fill=INK)
    draw.rectangle(
        [center_x - rule_inset, top + 6, center_x + rule_inset, top + 7], fill=INK
    )

    y = top + 30
    news_w = draw.textlength("News", font=wordmark_font)
    helper_w = draw.textlength("Helper", font=wordmark_font)
    start_x = center_x - (news_w + helper_w) / 2
    draw.text((start_x, y), "News", font=wordmark_font, fill=INK)
    draw.text((start_x + news_w, y), "Helper", font=wordmark_font, fill=ACCENT)
    y += 70

    tagline = "& the rest of the story"
    tagline_w = draw.textlength(tagline, font=tagline_font)
    draw.text((center_x - tagline_w / 2, y), tagline, font=tagline_font, fill=MUTED)
    y += 36

    draw.rectangle([center_x - rule_inset, y, center_x + rule_inset, y + 2], fill=INK)
    draw.rectangle([center_x - rule_inset, y + 6, center_x + rule_inset, y + 7], fill=INK)
    return y + 30


def render_card(enriched: EnrichedStory, out_png: Path) -> Path:
    """Render a portrait title/summary card, branded to match the site
    (static/css/style.css palette + static/brand/logo.svg masthead)."""
    img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), color=PAPER)
    draw = ImageDraw.Draw(img)
    center_x = VIDEO_WIDTH // 2

    margin = 90
    max_text_width = VIDEO_WIDTH - 2 * margin

    title_font = ImageFont.truetype(str(TITLE_FONT_PATH), 72)
    body_font = ImageFont.truetype(str(BODY_FONT_PATH), 46)

    title_lines = _wrap(draw, strip_source_suffix(enriched.story.title), title_font, max_text_width)
    summary_lines = _wrap(draw, enriched.summary, body_font, max_text_width)

    title_line_height = int(title_font.size * 1.25)
    body_line_height = int(body_font.size * 1.45)
    block_height = (
        len(title_lines) * title_line_height
        + 50
        + len(summary_lines) * body_line_height
    )

    masthead_bottom = _draw_masthead(draw, center_x, 120, VIDEO_WIDTH - 2 * margin)
    y = masthead_bottom + max(60, (VIDEO_HEIGHT - masthead_bottom - 200 - block_height) // 2)

    draw.rectangle([margin, y - 30, margin + 120, y - 22], fill=ACCENT)

    for line in title_lines:
        draw.text((margin, y), line, font=title_font, fill=INK)
        y += title_line_height
    y += 50
    for line in summary_lines:
        draw.text((margin, y), line, font=body_font, fill=MUTED)
        y += body_line_height

    footer_y = VIDEO_HEIGHT - 90
    draw.rectangle([margin, footer_y, VIDEO_WIDTH - margin, footer_y + 2], fill=RULE)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return out_png


def assemble(image_png: Path, audio_wav: Path, out_mp4: Path, duration_seconds: float) -> Path:
    """Combine a static card and narration into an mp4, with a slow Ken Burns
    zoom so the frame isn't perfectly static. Requires ffmpeg on PATH."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    total_frames = max(1, int(duration_seconds * VIDEO_FPS))
    zoom_expr = f"if(lte(zoom,1.0),1.05,zoom-0.0006)"

    cmd = [
        "ffmpeg",
        "-y",
        "-loop", "1",
        "-i", str(image_png),
        "-i", str(audio_wav),
        "-filter_complex",
        (
            f"[0:v]scale={VIDEO_WIDTH * 2}:{VIDEO_HEIGHT * 2},"
            f"zoompan=z='{zoom_expr}':d={total_frames}:"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:fps={VIDEO_FPS}"
        ),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_mp4


def make_story_video(enriched: EnrichedStory, work_dir: Path, out_mp4: Path) -> Path:
    """Full pipeline for one story: script -> narration -> card -> mp4."""
    script = build_narration_script(enriched)
    wav_path = narrate(script, work_dir / "narration.wav")
    duration = wav_duration_seconds(wav_path)
    png_path = render_card(enriched, work_dir / "card.png")
    return assemble(png_path, wav_path, out_mp4, duration)


def generate_all(enriched_stories: list[EnrichedStory], output_dir: Path) -> None:
    """Render one video per story into output_dir, setting each story's
    `video_path` (relative to the site root, e.g. "video/01-slug.mp4") on
    success. Wipes output_dir first so yesterday's videos never linger.

    A single story's video failing (missing model, a bad ffmpeg run, etc.)
    is logged and skipped rather than raised -- this is a best-effort
    enhancement, never a reason to fail the whole daily build.
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        for i, enriched in enumerate(enriched_stories, start=1):
            slug = slugify(strip_source_suffix(enriched.story.title)) or f"story-{i}"
            filename = f"{i:02d}-{slug}.mp4"
            out_path = output_dir / filename
            try:
                make_story_video(enriched, Path(tmp), out_path)
                enriched.video_path = f"video/{filename}"
                logger.info("generated video for %r", enriched.story.title)
            except Exception:
                logger.exception("video generation failed for %r; skipping", enriched.story.title)
