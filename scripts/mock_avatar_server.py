"""Throwaway stand-in for the real GPU-cluster avatar inference service, so
the client contract in src/newshelper/avatar.py can be tested end-to-end
today without waiting on the actual cluster/MuseTalk setup.

Implements the same HTTP contract the real service will (see avatar.py's
docstring): POST /jobs (multipart image+audio) -> job_id, GET /jobs/<id>
polls status, GET /videos/<id>.mp4 serves the result. Instead of running a
lip-sync model, it just composites the image + audio with ffmpeg (same
technique as video.py's assemble()) after a short artificial delay, so the
async submit/poll/download flow gets genuinely exercised.

Run:
    .venv/bin/python scripts/mock_avatar_server.py
Then in another shell, with NEWSHELPER_AVATAR_SERVICE_URL=http://localhost:8935:
    .venv/bin/python -c "from pathlib import Path; from newshelper.avatar import render_talking_head; render_talking_head(Path('/tmp/some_narration.wav'), Path('/tmp/avatar_test.mp4'))"
"""

import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 8935
JOBS: dict[str, dict] = {}
WORK_DIR = Path(tempfile.mkdtemp(prefix="mock_avatar_"))


def _parse_multipart(handler: BaseHTTPRequestHandler) -> dict[str, bytes]:
    content_type = handler.headers["Content-Type"]
    length = int(handler.headers["Content-Length"])
    body = handler.rfile.read(length)

    # cgi.parse_multipart is deprecated/removed in newer Pythons in some
    # builds; do the minimal boundary split ourselves to avoid the dependency.
    boundary = content_type.split("boundary=")[1].encode()
    parts = body.split(b"--" + boundary)
    fields = {}
    for part in parts:
        if b'name="' not in part:
            continue
        header, _, content = part.partition(b"\r\n\r\n")
        name = header.split(b'name="')[1].split(b'"')[0].decode()
        fields[name] = content.rstrip(b"\r\n--\r\n").rstrip(b"\r\n")
    return fields


def _render_job(job_id: str, image_bytes: bytes, audio_bytes: bytes) -> None:
    time.sleep(2)  # pretend this is real GPU inference time, not instant
    image_path = WORK_DIR / f"{job_id}.png"
    audio_path = WORK_DIR / f"{job_id}.wav"
    out_path = WORK_DIR / f"{job_id}.mp4"
    image_path.write_bytes(image_bytes)
    audio_path.write_bytes(audio_bytes)

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-loop", "1", "-i", str(image_path),
                "-i", str(audio_path),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                str(out_path),
            ],
            check=True, capture_output=True,
        )
        JOBS[job_id] = {"status": "done", "video_path": str(out_path)}
    except subprocess.CalledProcessError as exc:
        JOBS[job_id] = {"status": "error", "error": exc.stderr.decode(errors="replace")[-500:]}


class Handler(BaseHTTPRequestHandler):
    def _json(self, payload: dict, code: int = 200) -> None:
        import json
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/jobs":
            self._json({"error": "not found"}, 404)
            return
        fields = _parse_multipart(self)
        job_id = uuid.uuid4().hex[:12]
        JOBS[job_id] = {"status": "running"}
        threading.Thread(
            target=_render_job, args=(job_id, fields["image"], fields["audio"]), daemon=True
        ).start()
        self._json({"job_id": job_id}, 202)

    def do_GET(self):
        if self.path.startswith("/jobs/"):
            job_id = self.path.removeprefix("/jobs/")
            job = JOBS.get(job_id)
            if job is None:
                self._json({"error": "unknown job"}, 404)
                return
            if job["status"] == "done":
                self._json({"status": "done", "video_url": f"http://localhost:{PORT}/videos/{job_id}.mp4"})
            else:
                self._json(job)
            return

        if self.path.startswith("/videos/"):
            job_id = self.path.removeprefix("/videos/").removesuffix(".mp4")
            job = JOBS.get(job_id)
            if not job or job.get("status") != "done":
                self._json({"error": "not ready"}, 404)
                return
            video_bytes = Path(job["video_path"]).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(video_bytes)))
            self.end_headers()
            self.wfile.write(video_bytes)
            return

        self._json({"error": "not found"}, 404)

    def log_message(self, fmt, *args):
        print(f"[mock-avatar] {fmt % args}", file=sys.stderr)


if __name__ == "__main__":
    print(f"Mock avatar service on http://localhost:{PORT} (work dir: {WORK_DIR})")
    ThreadingHTTPServer(("localhost", PORT), Handler).serve_forever()
