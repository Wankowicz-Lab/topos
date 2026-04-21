"""Attach a file transcript for pipeline runs (see Runner.save_results)."""

from __future__ import annotations

import logging
from pathlib import Path

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def attach_run_file_log(path: Path, *, level: int = logging.INFO) -> logging.FileHandler:
    """Append a FileHandler to the root logger and route warnings through logging.

    Uses the same format as typical basicConfig setups so console and file match when both are configured.
    Temporarily lowers the root level to INFO so INFO records reach the file (default root is WARNING).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root = logging.getLogger()
    old = root.level
    handler._biogenesis_prev_root_level = old  # type: ignore[attr-defined]
    if old == logging.NOTSET:
        root.setLevel(logging.INFO)
    else:
        root.setLevel(min(old, logging.INFO))
    root.addHandler(handler)
    logging.captureWarnings(True)
    return handler


def detach_run_file_log(handler: logging.FileHandler) -> None:
    """Remove the run FileHandler and stop mapping warnings to logging."""
    root = logging.getLogger()
    root.removeHandler(handler)
    prev = getattr(handler, "_biogenesis_prev_root_level", logging.NOTSET)
    root.setLevel(prev)
    handler.close()
    logging.captureWarnings(False)
