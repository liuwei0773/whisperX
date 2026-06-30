import json
import logging
import os
import sys
from typing import Any, Optional

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "info",
    log_file: Optional[str] = None,
) -> None:
    """
    Configure logging for WhisperX.

    Args:
        level: Logging level (debug, info, warning, error, critical). Default: info
        log_file: Optional path to log file. If None, logs only to console.
    """
    logger = logging.getLogger("whisperx")

    logger.handlers.clear()

    try:
        log_level = getattr(logging, level.upper())
    except AttributeError:
        log_level = logging.WARNING
    logger.setLevel(log_level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (OSError) as e:
            logger.warning(f"Failed to create log file '{log_file}': {e}")
            logger.warning("Continuing with console logging only")

    # Don't propagate to root logger to avoid duplicate messages
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module.

    Args:
        name: Logger name (typically __name__ from calling module)

    Returns:
        Logger instance configured with WhisperX settings
    """
    whisperx_logger = logging.getLogger("whisperx")
    if not whisperx_logger.handlers:
        setup_logging()

    logger_name = "whisperx" if name == "__main__" else name
    return logging.getLogger(logger_name)


def _json_default(obj: Any) -> Any:
    """Fallback serializer for objects json can't handle natively.

    Mainly numpy scalars/arrays produced along the ASR pipeline; anything
    else degrades to its repr so a debug dump never crashes the run.
    """
    # numpy scalar (np.float32 avg_logprob, etc.) -> python scalar
    if hasattr(obj, "item") and not hasattr(obj, "__len__"):
        try:
            return obj.item()
        except (ValueError, TypeError):
            pass
    # numpy array / tensor -> list
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except (ValueError, TypeError):
            pass
    return repr(obj)


def dump_debug_artifact(debug_dir: Optional[str], filename: str, data: Any) -> None:
    """Write one pipeline-stage artifact to ``debug_dir`` as pretty JSON.

    No-op when ``debug_dir`` is None (debugging disabled). Failures are logged
    but never raised — a broken dump must not abort transcription.

    Args:
        debug_dir: Directory to write into, or None to disable.
        filename: File name within ``debug_dir`` (e.g. "clip.02_asr.json").
        data: Any JSON-serializable structure; numpy values are handled.
    """
    if debug_dir is None:
        return

    logger = logging.getLogger("whisperx")
    try:
        os.makedirs(debug_dir, exist_ok=True)
        path = os.path.join(debug_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
        logger.debug(f"Wrote debug artifact: {path}")
    except (OSError, TypeError, ValueError) as e:
        logger.warning(f"Failed to write debug artifact '{filename}': {e}")
