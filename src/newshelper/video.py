"""Turns an enriched story into a short narrated summary video: a branded
title/summary card with each word highlighted karaoke-style as it's spoken
(accessibility -- lets a viewer follow along without sound), plus the
NewsHelper owl mascot animated in a bottom-left "newscaster inset" (see
mascot.py). No Ken Burns zoom -- an earlier version had one, but it was
distracting alongside the mascot and the word highlighting.

Pipeline: narrate() (Kokoro TTS) -> render_caption_timeline() (one frame
per word, matched to estimated per-word timing) composited with
mascot.write_pip_timeline() (the mouth-flap animation) via
assemble_with_avatar_pip().
"""

import logging
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from newshelper.config import (
    KOKORO_MODEL_PATH,
    KOKORO_VOICE,
    KOKORO_VOICES_PATH,
    TONE_VOICE,
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


def estimate_word_timings(script: str, total_duration: float) -> list[tuple[str, float, float]]:
    """(word, start_seconds, end_seconds) for each word in script, spread
    across total_duration proportional to word length.

    Kokoro doesn't expose real per-word timestamps, so this is an estimate,
    not a forced alignment -- consistent with mascot.py's amplitude-driven
    mouth-flap rather than true lip-sync: simple, no extra dependencies,
    good enough for a karaoke-style highlight rather than frame-perfect sync.
    """
    words = script.split()
    if not words:
        return []
    weights = [len(w) + 1 for w in words]
    total_weight = sum(weights)
    timings = []
    t = 0.0
    for word, wt in zip(words, weights):
        dur = total_duration * wt / total_weight
        timings.append((word, t, t + dur))
        t += dur
    return timings


_kokoro_instance = None


def _get_kokoro():
    """Lazily load Kokoro (a ~325MB model) once per process and reuse it
    across stories in the same build run."""
    global _kokoro_instance
    if _kokoro_instance is None:
        from kokoro_onnx import Kokoro  # local import: heavy, optional dependency

        model_path = Path(KOKORO_MODEL_PATH)
        voices_path = Path(KOKORO_VOICES_PATH)
        if not model_path.exists() or not voices_path.exists():
            raise FileNotFoundError(
                f"Kokoro model files not found at {model_path} / {voices_path}. "
                "Download them first, e.g.:\n"
                "  mkdir -p data/kokoro_voices\n"
                "  curl -sL -o data/kokoro_voices/kokoro-v1.0.onnx "
                "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx\n"
                "  curl -sL -o data/kokoro_voices/voices-v1.0.bin "
                "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
            )
        _kokoro_instance = Kokoro(str(model_path), str(voices_path))
    return _kokoro_instance


def narrate(text: str, out_wav: Path, voice: str = KOKORO_VOICE, speed: float = 1.0) -> Path:
    """Synthesize `text` to a wav file with Kokoro (v2.5 -- replaced Piper,
    whose voice quality wasn't good enough). Requires KOKORO_MODEL_PATH /
    KOKORO_VOICES_PATH to exist locally (see data/kokoro_voices/ -- not
    committed, download on-demand). `voice`/`speed` default to the
    standard narrator at normal pace; make_story_video overrides both per
    TONE_VOICE to moderate mood -- see config.TONE_VOICE."""
    kokoro = _get_kokoro()
    samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang="en-us")

    if np.issubdtype(samples.dtype, np.floating):
        pcm = np.clip(samples, -1.0, 1.0)
        pcm = (pcm * 32767).astype(np.int16)
    else:
        pcm = samples.astype(np.int16)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_wav), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
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


HIGHLIGHT_FILL = (240, 214, 168)  # soft on-brand "highlighter" tone


def _build_card_chrome(enriched: EnrichedStory) -> tuple[Image.Image, dict]:
    """The card's static chrome (masthead, accent bar, footer rule, paper
    background) plus the text layout info needed to draw words onto it --
    split out from the actual word-drawing so render_caption_timeline can
    reuse the same chrome across every word-highlight frame instead of
    redrawing the masthead hundreds of times."""
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
    summary_gap = 50
    block_height = (
        len(title_lines) * title_line_height + summary_gap + len(summary_lines) * body_line_height
    )

    masthead_bottom = _draw_masthead(draw, center_x, 120, VIDEO_WIDTH - 2 * margin)
    title_start_y = masthead_bottom + max(60, (VIDEO_HEIGHT - masthead_bottom - 200 - block_height) // 2)

    draw.rectangle([margin, title_start_y - 30, margin + 120, title_start_y - 22], fill=ACCENT)

    footer_y = VIDEO_HEIGHT - 90
    draw.rectangle([margin, footer_y, VIDEO_WIDTH - margin, footer_y + 2], fill=RULE)

    layout = {
        "margin": margin,
        "title_lines": title_lines,
        "summary_lines": summary_lines,
        "title_font": title_font,
        "body_font": body_font,
        "title_line_height": title_line_height,
        "body_line_height": body_line_height,
        "title_start_y": title_start_y,
        "summary_gap": summary_gap,
    }
    return img, layout


def _word_positions(
    draw: ImageDraw.ImageDraw, lines: list[str], font: ImageFont.FreeTypeFont,
    margin: int, start_y: int, line_height: int,
) -> tuple[list[dict], int]:
    """Per-word pixel positions within already-wrapped lines, so a highlight
    rectangle can be drawn behind exactly one word without re-wrapping."""
    positions = []
    y = start_y
    for line in lines:
        x = margin
        for word in line.split():
            w = draw.textlength(word, font=font)
            positions.append({"word": word, "x": x, "y": y, "w": w, "h": font.size})
            x += draw.textlength(word + " ", font=font)
        y += line_height
    return positions, y


def render_card_frame(base_img: Image.Image, layout: dict, highlight_index: int | None = None) -> Image.Image:
    """One frame: base_img's chrome plus title+summary text, with the word
    at highlight_index (if any) given a highlighter-style background --
    the karaoke effect, one word lit up at a time as it's narrated."""
    img = base_img.copy()
    draw = ImageDraw.Draw(img)

    title_positions, y_after_title = _word_positions(
        draw, layout["title_lines"], layout["title_font"], layout["margin"],
        layout["title_start_y"], layout["title_line_height"],
    )
    summary_start_y = y_after_title + layout["summary_gap"]
    summary_positions, _ = _word_positions(
        draw, layout["summary_lines"], layout["body_font"], layout["margin"],
        summary_start_y, layout["body_line_height"],
    )
    all_positions = title_positions + summary_positions

    if highlight_index is not None and 0 <= highlight_index < len(all_positions):
        p = all_positions[highlight_index]
        pad = 5
        draw.rounded_rectangle(
            [p["x"] - pad, p["y"] - pad, p["x"] + p["w"] + pad, p["y"] + p["h"] + pad],
            radius=6, fill=HIGHLIGHT_FILL,
        )

    for p in title_positions:
        draw.text((p["x"], p["y"]), p["word"], font=layout["title_font"], fill=INK)
    for p in summary_positions:
        draw.text((p["x"], p["y"]), p["word"], font=layout["body_font"], fill=MUTED)

    return img


def render_card(enriched: EnrichedStory, out_png: Path) -> Path:
    """A single static card with no word highlighted -- used for a poster
    frame, not by the video pipeline itself (see render_caption_timeline)."""
    base_img, layout = _build_card_chrome(enriched)
    img = render_card_frame(base_img, layout, highlight_index=None)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return out_png


def render_caption_timeline(enriched: EnrichedStory, wav_path: Path, work_dir: Path) -> Path:
    """Render one card frame per narrated word (that word highlighted) and
    write an ffmpeg concat-demuxer file timed to each word's estimated
    duration. Returns the concat file's path."""
    duration = wav_duration_seconds(wav_path)
    script = build_narration_script(enriched)
    timings = estimate_word_timings(script, duration)

    base_img, layout = _build_card_chrome(enriched)

    work_dir.mkdir(parents=True, exist_ok=True)
    concat_path = work_dir / "card_concat.txt"
    with open(concat_path, "w") as f:
        last_frame_path = None
        for i, (_word, start, end) in enumerate(timings):
            frame = render_card_frame(base_img, layout, highlight_index=i)
            frame_path = work_dir / f"card_{i:04d}.png"
            frame.save(frame_path)
            f.write(f"file '{frame_path}'\n")
            f.write(f"duration {max(end - start, 1 / VIDEO_FPS)}\n")
            last_frame_path = frame_path
        if last_frame_path:
            f.write(f"file '{last_frame_path}'\n")
    return concat_path


def assemble_with_avatar_pip(
    enriched: EnrichedStory, mascot_path: Path, audio_wav: Path, out_mp4: Path, work_dir: Path
) -> Path:
    """Composite the word-highlighted caption timeline with the mascot's
    amplitude-driven mouth-flap animation as a bottom-left "newscaster
    inset" -- see mascot.py for why it's a mouth-flap and not a real
    lip-sync model. No Ken Burns zoom: static card frames only, the word
    highlighting is the motion now."""
    from newshelper import mascot

    card_concat_path = render_caption_timeline(enriched, audio_wav, work_dir / "card_frames")
    avatar_concat_path, mask_path = mascot.write_pip_timeline(
        mascot_path, audio_wav, work_dir / "avatar_frames"
    )

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    pip_margin_x = 40
    pip_margin_y = 60
    overlay_x = pip_margin_x
    overlay_y = VIDEO_HEIGHT - mascot.PIP_SIZE - pip_margin_y

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(card_concat_path),
        "-f", "concat", "-safe", "0", "-i", str(avatar_concat_path),
        "-loop", "1", "-i", str(mask_path),
        "-i", str(audio_wav),
        "-filter_complex",
        (
            f"[0:v]fps={VIDEO_FPS}[bg];"
            f"[1:v]fps={VIDEO_FPS}[avatar_v];"
            f"[2:v]fps={VIDEO_FPS},format=gray[mask];"
            f"[avatar_v][mask]alphamerge[avatar];"
            f"[bg][avatar]overlay=x={overlay_x}:y={overlay_y}:shortest=1[outv]"
        ),
        "-map", "[outv]", "-map", "3:a",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_mp4


def make_story_video(enriched: EnrichedStory, work_dir: Path, out_mp4: Path) -> Path:
    """Full pipeline for one story: script -> narration -> captioned card
    frames + mascot PIP -> mp4."""
    from newshelper.config import AVATAR_IMAGE_PATH

    script = build_narration_script(enriched)
    voice, speed = TONE_VOICE.get(enriched.tone, TONE_VOICE["neutral"])
    wav_path = narrate(script, work_dir / "narration.wav", voice=voice, speed=speed)
    return assemble_with_avatar_pip(enriched, Path(AVATAR_IMAGE_PATH), wav_path, out_mp4, work_dir)


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
