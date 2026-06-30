"""WhisperX stage debug viewer.

A zero-dependency local web UI for inspecting the per-stage JSON artifacts
emitted by ``whisperx ... --debug_dir`` alongside the original media.

Run with: ``python -m whisperx.viewer --debug_dir <dir> --media <file>``
"""

from whisperx.viewer.server import make_server

__all__ = ["make_server"]
