"""
SentinelWeb Logger

Centralized logging module for the SentinelWeb project.

Features
--------
- Singleton logger configuration
- Colored console logs
- Daily rotating log files
- Automatic log directory creation
- UTF-8 file encoding
- No duplicate handlers
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from threading import Lock


# ==========================================================
# Log Directory
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOGS_DIR = PROJECT_ROOT / "backend" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Console Formatter
# ==========================================================

class ColoredFormatter(logging.Formatter):
    """ANSI colored formatter for console logging."""

    COLORS = {
        logging.DEBUG: "\033[94m",      # Blue
        logging.INFO: "\033[92m",       # Green
        logging.WARNING: "\033[93m",    # Yellow
        logging.ERROR: "\033[91m",      # Red
        logging.CRITICAL: "\033[1;31m", # Bold Red
    }

    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)

        cloned = logging.makeLogRecord(record.__dict__)
        cloned.levelname = f"{color}{cloned.levelname}{self.RESET}"

        return super().format(cloned)


# ==========================================================
# Logger Singleton
# ==========================================================

class _LoggerManager:
    """Singleton logger configuration."""

    _initialized = False
    _lock = Lock()

    @classmethod
    def initialize(cls) -> None:

        with cls._lock:

            if cls._initialized:
                return

            log_format = (
                "%(asctime)s | "
                "%(levelname)s | "
                "%(name)s | "
                "%(funcName)s | "
                "Line %(lineno)d | "
                "%(message)s"
            )

            date_format = "%Y-%m-%d %H:%M:%S"

            root_logger = logging.getLogger()

            root_logger.setLevel(logging.DEBUG)

            if root_logger.handlers:
                root_logger.handlers.clear()

            # -------------------------------
            # Console Handler
            # -------------------------------

            console_handler = logging.StreamHandler(sys.stdout)

            console_handler.setLevel(logging.INFO)

            console_handler.setFormatter(
                ColoredFormatter(
                    fmt=log_format,
                    datefmt=date_format,
                )
            )

            # -------------------------------
            # File Handler
            # -------------------------------

            file_handler = TimedRotatingFileHandler(
                filename=LOGS_DIR / "sentinelweb.log",
                when="midnight",
                interval=1,
                backupCount=30,
                encoding="utf-8",
            )

            file_handler.setLevel(logging.DEBUG)

            file_handler.setFormatter(
                logging.Formatter(
                    fmt=log_format,
                    datefmt=date_format,
                )
            )

            root_logger.addHandler(console_handler)
            root_logger.addHandler(file_handler)

            cls._initialized = True


# ==========================================================
# Public Function
# ==========================================================

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger.

    Example
    -------
    logger = get_logger(__name__)
    """

    _LoggerManager.initialize()

    return logging.getLogger(name)