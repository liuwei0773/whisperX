"""CLI entry point for the WhisperX stage debug viewer.

Usage:
    python -m whisperx.viewer --debug_dir ./dbg --media input.mp4

Reads the per-stage JSON artifacts written by ``whisperx ... --debug_dir`` and
serves a local web UI that plays the media alongside every pipeline stage.
"""

import argparse
import os
import sys
import threading
import webbrowser

from whisperx.log_utils import get_logger, setup_logging
from whisperx.viewer.frontend import HTML
from whisperx.viewer.server import make_server

logger = get_logger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="whisperx.viewer",
        description="Visual step-by-step debugger for WhisperX pipeline stages.",
    )
    parser.add_argument("--debug_dir", required=True,
                        help="directory containing the *.NN_*.json stage artifacts "
                             "(produced by `whisperx ... --debug_dir`)")
    parser.add_argument("--media", required=True,
                        help="original audio/video file to play alongside the stages")
    parser.add_argument("--host", default="127.0.0.1",
                        help="host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=0,
                        help="port to bind (default: 0 = pick a free port)")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open a browser automatically")
    args = parser.parse_args(argv)

    setup_logging(level="info")

    if not os.path.isdir(args.debug_dir):
        parser.error(f"--debug_dir not found or not a directory: {args.debug_dir}")
    if not os.path.isfile(args.media):
        parser.error(f"--media file not found: {args.media}")

    httpd = make_server(args.host, args.port, args.debug_dir, args.media, HTML)
    host, port = httpd.server_address[0], httpd.server_address[1]
    url = f"http://{host}:{port}/"
    logger.info(f"Debug viewer serving at {url}")
    logger.info(f"  stages: {os.path.abspath(args.debug_dir)}")
    logger.info(f"  media:  {os.path.abspath(args.media)}")
    logger.info("Press Ctrl+C to stop.")

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down viewer.")
    finally:
        httpd.shutdown()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
