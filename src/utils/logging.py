"""Logging configuration for project workflows."""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "pore_pressure", log_file: Optional[Path] = None) -> logging.Logger:
    """Configure and return a structured logger."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
