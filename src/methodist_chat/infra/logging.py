"""Единая настройка логирования через rich."""

from __future__ import annotations

import logging
from logging import Logger

from rich.logging import RichHandler

from .config import get_settings

_configured = False


def setup_logging() -> None:
    global _configured
    if _configured:
        return
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=True)],
    )
    _configured = True


def get_logger(name: str) -> Logger:
    setup_logging()
    return logging.getLogger(name)
