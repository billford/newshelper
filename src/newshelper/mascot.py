"""Animates the NewsHelper mascot's mouth along with narration audio.

Deliberately not a lip-sync model: the mouth just opens/closes/mids based on
how loud the narration is at each moment (amplitude-driven "mouth flap"),
not which sound is being spoken. This is the fallback chosen over a real
lip-sync model (Wav2Lip/MuseTalk/SadTalker) for two reasons -- those are
trained on real human faces and don't generalize well to a simplified
cartoon mouth, and this needs no GPU cluster at all, running entirely on
wanderlust like the rest of the pipeline.

Three mouth states (closed/mid/open) are painted onto copies of the base
portrait, then stitched to the narration's timing via ffmpeg's concat
demuxer -- cheap because only 3 frames ever get rendered, no matter how
long the clip is.
"""

import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Region of static/brand/mascot.png (1024x1024) containing the owl's beak,
# picked by visual inspection -- kept tight and well clear of the eyes,
# since anything stretched/sampled near them risks smearing their edge in.
BEAK_BBOX = (475, 425, 570, 545)
FACE_FILL = (251, 246, 227)  # sampled from a clean patch of face plumage
INK = (38, 36, 32)  # matches video.py's brand palette
BEAK_TOP_FILL = (214, 144, 50)  # sampled from the beak's upper half
BEAK_BOTTOM_FILL = (161, 90, 24)  # sampled from the beak's lower half
MOUTH_FILL = (140, 46, 40)  # dark interior, visible in the open gap

# Tight square crop around just the head (ear tufts + wings + bow tie,
# excludes feet), used for the picture-in-picture "newscaster inset" --
# picked by visual inspection, see the crop checks that led here.
FACE_CROP_BBOX = (55, 5, 995, 945)
PIP_SIZE = 360  # final inset diameter in the 1080x1920 video


def _erase_beak(img: Image.Image) -> Image.Image:
    """Return a copy of img with the original beak painted over in flat
    face color. Solid fill rather than sampling/stretching a nearby patch --
    stretching risks smearing in a sliver of a nearby feature (learned the
    hard way on the previous mascot: a strip grazing the eye's edge became
    a vertical stripe once stretched across the whole erase height)."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle(BEAK_BBOX, fill=FACE_FILL)
    return out


def _draw_beak(draw: ImageDraw.ImageDraw, cx: int, cy: int, width: float, gap: float) -> None:
    """Upper and lower beak halves separated by `gap` pixels -- an owl
    "talking" by opening its beak, the same trick Twitter's bird and
    Duolingo's owl use instead of a real mouth."""
    half_h = 55
    if gap > 4:
        draw.ellipse(
            [cx - width / 2, cy - gap / 2 - half_h, cx + width / 2, cy - gap / 2 + 10],
            fill=MOUTH_FILL,
        )
    draw.pieslice(
        [cx - width / 2, cy - gap / 2 - half_h, cx + width / 2, cy - gap / 2 + half_h],
        180, 360, fill=BEAK_TOP_FILL, outline=INK, width=5,
    )
    draw.pieslice(
        [cx - width / 2, cy + gap / 2 - half_h, cx + width / 2, cy + gap / 2 + half_h],
        0, 180, fill=BEAK_BOTTOM_FILL, outline=INK, width=5,
    )


def build_mouth_states(mascot_path: Path) -> dict[str, Image.Image]:
    """Three full-frame images: closed (the original), mid, and open beak."""
    base = Image.open(mascot_path).convert("RGB")
    x0, y0, x1, y1 = BEAK_BBOX
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    width = (x1 - x0) * 0.95

    closed = base

    mid = _erase_beak(base)
    _draw_beak(ImageDraw.Draw(mid), cx, cy, width, gap=14)

    open_ = _erase_beak(base)
    _draw_beak(ImageDraw.Draw(open_), cx, cy, width, gap=48)

    return {"closed": closed, "mid": mid, "open": open_}


def build_pip_states(mascot_path: Path) -> dict[str, Image.Image]:
    """Same three mouth states as build_mouth_states, but cropped tight to
    just the face and sized for the picture-in-picture newscaster inset.
    No border ring drawn here -- the circular alpha mask (see circular_mask
    and video.py's overlay) is what gives it a clean circular edge; an inked
    ring baked into the frame on top of that just reads as an ugly dark
    halo, not a "camera bug" outline."""
    states = {}
    for name, img in build_mouth_states(mascot_path).items():
        states[name] = img.crop(FACE_CROP_BBOX).resize((PIP_SIZE, PIP_SIZE))
    return states


def circular_mask(size: int) -> Image.Image:
    """A white-filled circle on black, used as an alpha channel (via
    ffmpeg's alphamerge) so the PIP inset's square corners don't show."""
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([0, 0, size, size], fill=255)
    return mask


def _write_concat_file(concat_path: Path, frame_paths: dict[str, Path], states: list[str], chunk_seconds: float) -> None:
    with open(concat_path, "w") as f:
        for state in states:
            f.write(f"file '{frame_paths[state]}'\n")
            f.write(f"duration {chunk_seconds}\n")
        # concat demuxer quirk: the last "file" line needs no trailing
        # duration to take effect, so repeat it once more.
        f.write(f"file '{frame_paths[states[-1]]}'\n")


def write_pip_timeline(
    mascot_path: Path, wav_path: Path, work_dir: Path, chunk_seconds: float = 0.12
) -> tuple[Path, Path]:
    """Write the PIP mouth-state frames, a concat-demuxer timeline matching
    the narration's amplitude, and a circular alpha mask into work_dir.
    Returns (concat_path, mask_path) for the caller's ffmpeg command --
    see video.py's assemble_with_avatar_pip, which owns the actual overlay
    compositing since that's where the branded card background lives."""
    work_dir.mkdir(parents=True, exist_ok=True)
    states = amplitude_states(wav_path, chunk_seconds)
    frames = build_pip_states(mascot_path)

    frame_paths = {}
    for name, img in frames.items():
        path = work_dir / f"pip_{name}.png"
        img.save(path)
        frame_paths[name] = path

    concat_path = work_dir / "pip_concat.txt"
    _write_concat_file(concat_path, frame_paths, states, chunk_seconds)

    mask_path = work_dir / "pip_mask.png"
    circular_mask(PIP_SIZE).save(mask_path)

    return concat_path, mask_path


def amplitude_states(wav_path: Path, chunk_seconds: float = 0.12) -> list[str]:
    """Bucket the wav into "closed"/"mid"/"open" per chunk_seconds, based on
    each chunk's RMS relative to the clip's own loud/quiet range -- relative
    rather than an absolute threshold so it adapts to each narration's
    volume instead of assuming a fixed loudness."""
    with wave.open(str(wav_path), "rb") as wav_file:
        n_frames = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        raw = wav_file.readframes(n_frames)

    if sample_width != 2:
        raise ValueError(f"expected 16-bit PCM, got sample width {sample_width}")
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)

    chunk_frames = max(1, int(frame_rate * chunk_seconds))
    rms_values = [
        np.sqrt(np.mean(np.square(chunk)))
        for i in range(0, len(samples), chunk_frames)
        if len(chunk := samples[i : i + chunk_frames]) == chunk_frames
    ]
    if not rms_values:
        return ["closed"]

    quiet, loud = np.percentile(rms_values, [20, 85])
    mid_cut = quiet + (loud - quiet) * 0.35
    open_cut = quiet + (loud - quiet) * 0.75

    states = []
    for rms in rms_values:
        if rms <= mid_cut:
            states.append("closed")
        elif rms <= open_cut:
            states.append("mid")
        else:
            states.append("open")
    return states


def assemble_mouth_flap_video(
    mascot_path: Path, wav_path: Path, out_mp4: Path, work_dir: Path, chunk_seconds: float = 0.12
) -> Path:
    """Render the mascot with a mouth-flap animation timed to wav_path,
    muxed with that same audio as narration."""
    import subprocess

    states = amplitude_states(wav_path, chunk_seconds)
    frames = build_mouth_states(mascot_path)

    work_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = {}
    for name, img in frames.items():
        path = work_dir / f"mouth_{name}.png"
        img.save(path)
        frame_paths[name] = path

    concat_path = work_dir / "concat.txt"
    _write_concat_file(concat_path, frame_paths, states, chunk_seconds)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_path),
            "-i", str(wav_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(out_mp4),
        ],
        check=True, capture_output=True,
    )
    return out_mp4
