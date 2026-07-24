"""Client for the (planned) GPU-cluster talking-head inference service.

v3 idea: an illustrated NewsHelper mascot reads each story's narration,
lip-synced by a model (MuseTalk, most likely) running on GPU hardware on
the local network -- not wanderlust, which has no GPU worth the name.

This module only talks to that service; it doesn't run any model itself.
Wanderlust's side of the contract:

    POST {AVATAR_SERVICE_URL}/jobs
        multipart form: image=<mascot portrait>, audio=<narration wav>
        -> 202 {"job_id": "..."}

    GET {AVATAR_SERVICE_URL}/jobs/{job_id}
        -> 200 {"status": "queued"|"running"|"done"|"error",
                 "video_url": "..."}   # present when status == "done"
                 "error": "..."}       # present when status == "error"

Async on purpose: lip-sync inference can take a while even on a real GPU,
and a home network + a single blocking HTTP request is a bad combination.

The feature is entirely opt-in: AVATAR_SERVICE_URL unset (the default)
means is_avatar_enabled() is False and callers should fall back to the
static branded card -- there is no cluster yet, and there may not be one
for a while, so nothing here should ever block the regular video pipeline.
"""

import logging
import time
from pathlib import Path

import requests

from newshelper.config import (
    AVATAR_IMAGE_PATH,
    AVATAR_POLL_INTERVAL_SECONDS,
    AVATAR_SERVICE_TIMEOUT_SECONDS,
    AVATAR_SERVICE_URL,
)

logger = logging.getLogger(__name__)


def is_avatar_enabled() -> bool:
    """Whether NEWSHELPER_AVATAR_SERVICE_URL is configured. False means the
    GPU cluster / inference service isn't set up (yet), which is the
    expected default -- callers should treat that as "skip, not an error"."""
    return bool(AVATAR_SERVICE_URL)


class AvatarRenderError(Exception):
    """The inference service reported a failure, or never finished in time."""


def render_talking_head(audio_wav: Path, out_mp4: Path) -> Path:
    """Submit a narration wav to the avatar inference service and poll until
    the lip-synced clip is ready, then download it to out_mp4.

    Raises AvatarRenderError (never falls back silently) so callers decide
    for themselves whether a failure here should just skip the avatar for
    this story -- see video.py's per-story error isolation for the pattern
    this is meant to slot into.
    """
    if not is_avatar_enabled():
        raise AvatarRenderError("NEWSHELPER_AVATAR_SERVICE_URL is not configured")

    image_path = Path(AVATAR_IMAGE_PATH)
    if not image_path.exists():
        raise AvatarRenderError(f"mascot portrait not found at {image_path}")

    with open(image_path, "rb") as image_file, open(audio_wav, "rb") as audio_file:
        response = requests.post(
            f"{AVATAR_SERVICE_URL}/jobs",
            files={"image": image_file, "audio": audio_file},
            timeout=30,
        )
    response.raise_for_status()
    job_id = response.json()["job_id"]
    logger.info("submitted avatar render job %s for %s", job_id, audio_wav.name)

    deadline = time.monotonic() + AVATAR_SERVICE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        status_response = requests.get(f"{AVATAR_SERVICE_URL}/jobs/{job_id}", timeout=30)
        status_response.raise_for_status()
        payload = status_response.json()
        status = payload.get("status")

        if status == "done":
            video_response = requests.get(payload["video_url"], timeout=60)
            video_response.raise_for_status()
            out_mp4.parent.mkdir(parents=True, exist_ok=True)
            out_mp4.write_bytes(video_response.content)
            return out_mp4
        if status == "error":
            raise AvatarRenderError(f"job {job_id} failed: {payload.get('error')}")

        time.sleep(AVATAR_POLL_INTERVAL_SECONDS)

    raise AvatarRenderError(f"job {job_id} did not finish within {AVATAR_SERVICE_TIMEOUT_SECONDS}s")
