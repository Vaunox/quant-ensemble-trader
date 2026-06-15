"""
Central logging configuration for algo_v2.

Import and call configure_logging() once at process startup (live_executor.py,
train_ensemble.py, etc.) before any other module-level logging occurs.

Log level is controlled by the LOG_LEVEL environment variable (default: INFO).
A file handler is added automatically when LOG_FILE is set in the environment.
"""

import logging
import os
import sys


def configure_logging(name: str = "algo_v2") -> logging.Logger:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler (stdout so cron captures it in phase6.log)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    # Optional file handler
    log_file = os.environ.get("LOG_FILE")
    if log_file and not any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(log_file)
        for h in root.handlers
    ):
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    return logging.getLogger(name)
