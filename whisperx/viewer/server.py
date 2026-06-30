"""Zero-dependency HTTP server for the WhisperX stage debug viewer.

Serves three things off the standard library only:
  - ``GET /``            the single-page frontend (see frontend.py)
  - ``GET /api/stages``  the four ``--debug_dir`` JSON artifacts, merged
  - ``GET /media``       the original media file, with HTTP Range support
                         (required for <video> seeking)

Security: only these fixed routes are served. ``/media`` streams exactly one
whitelisted file (the ``--media`` path resolved at startup); the stage JSONs
are read from the resolved ``--debug_dir`` by fixed suffix. No request path is
ever mapped onto the filesystem, so directory traversal is not possible. The
server binds to localhost by default.
"""

import glob
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from whisperx.log_utils import get_logger

logger = get_logger(__name__)

# Stage suffix -> key in the merged /api/stages payload.
_STAGE_SUFFIXES = {
    "01_vad_chunks": "vad",
    "02_asr": "asr",
    "03_align": "align",
    "04_diarize": "diarize",
}

_READ_CHUNK = 1024 * 256  # 256 KiB streaming chunk for /media


def _collect_stages(debug_dir: str) -> dict:
    """Load whichever stage JSON files exist in ``debug_dir``.

    Returns a dict with keys vad/asr/align/diarize (None when that stage's
    file is absent or unreadable) plus ``meta`` describing what was found.
    Files are matched by suffix so any audio basename works.
    """
    stages: dict = {key: None for key in _STAGE_SUFFIXES.values()}
    found = {}
    for suffix, key in _STAGE_SUFFIXES.items():
        matches = sorted(glob.glob(os.path.join(debug_dir, f"*.{suffix}.json")))
        if not matches:
            continue
        path = matches[0]  # one media per debug_dir in the common case
        try:
            with open(path, "r", encoding="utf-8") as f:
                stages[key] = json.load(f)
            found[key] = os.path.basename(path)
        except (OSError, ValueError) as e:
            logger.warning(f"Could not read stage file {path}: {e}")
    stages["meta"] = {"debug_dir": debug_dir, "found": found}
    return stages


def _parse_range(range_header: str, file_size: int) -> Optional[tuple]:
    """Parse a single ``bytes=start-end`` Range header.

    Returns (start, end) inclusive byte offsets clamped to the file, or None
    if the header is absent/malformed (caller then serves the full file).
    Only the common single-range form is supported — enough for media seeking.
    """
    if not range_header or not range_header.startswith("bytes="):
        return None
    spec = range_header[len("bytes="):].split(",")[0].strip()
    if "-" not in spec:
        return None
    start_s, _, end_s = spec.partition("-")
    try:
        if start_s == "":
            # suffix range: bytes=-N  -> last N bytes
            length = int(end_s)
            if length <= 0:
                return None
            start = max(0, file_size - length)
            end = file_size - 1
        else:
            start = int(start_s)
            end = int(end_s) if end_s else file_size - 1
    except ValueError:
        return None
    end = min(end, file_size - 1)
    if start > end or start >= file_size:
        return None
    return start, end


class _ViewerHandler(BaseHTTPRequestHandler):
    # Injected by make_server via class attributes on a subclass.
    debug_dir: str = ""
    media_path: str = ""
    html: bytes = b""

    def log_message(self, fmt, *args):  # quieter: route through our logger
        logger.debug("%s - %s" % (self.address_string(), fmt % args))

    def _send_json(self, obj: dict, status: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        # Strip any query string; we route only on the path.
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path == "/api/stages":
            self._send_json(_collect_stages(self.debug_dir))
        elif path == "/media":
            self._serve_media()
        else:
            self._send_json({"error": "not found"}, status=404)

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(self.html)))
        self.end_headers()
        self.wfile.write(self.html)

    def _serve_media(self):
        # Only ever serve the one whitelisted file resolved at startup.
        try:
            file_size = os.path.getsize(self.media_path)
        except OSError:
            self._send_json({"error": "media not found"}, status=404)
            return

        ctype = mimetypes.guess_type(self.media_path)[0] or "application/octet-stream"
        rng = _parse_range(self.headers.get("Range", ""), file_size)

        if rng is None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            start, end = 0, file_size - 1
        else:
            start, end = rng
            self.send_response(206)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        remaining = end - start + 1
        try:
            with open(self.media_path, "rb") as f:
                f.seek(start)
                while remaining > 0:
                    chunk = f.read(min(_READ_CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # Client seeked/closed mid-stream; normal for <video>.
            pass


def make_server(host: str, port: int, debug_dir: str, media_path: str, html: bytes):
    """Build a ThreadingHTTPServer bound to (host, port).

    debug_dir and media_path are resolved to absolute paths and frozen onto the
    handler class, so requests can never widen the served file set.
    """
    handler = type(
        "BoundViewerHandler",
        (_ViewerHandler,),
        {
            "debug_dir": os.path.abspath(debug_dir),
            "media_path": os.path.abspath(media_path),
            "html": html,
        },
    )
    return ThreadingHTTPServer((host, port), handler)
