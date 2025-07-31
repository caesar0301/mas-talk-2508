"""
Simplified logging configuration for agentdemo with color support.

Usage:
    from agentdemo.common.logging_config import setup_logging, get_logger

    # Basic usage - automatically uses environment variables if available
    setup_logging()
    logger = get_logger(__name__)

    # Explicit configuration
    setup_logging(level="DEBUG", enable_colors=True)

    # Disable environment configuration
    setup_logging(use_env_config=False, level="INFO")

Environment Variables:
    AGENTDEMO_LOG_LEVEL: Log level (default: INFO)
    AGENTDEMO_LOG_COLORS: Enable colors (default: true)
    AGENTDEMO_LOG_FILE: Log file path (optional)
    AGENTDEMO_LOG_FILE_LEVEL: File log level (default: DEBUG)
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

try:
    import colorlog
except ImportError:
    colorlog = None  # fallback if colorlog is not installed


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_file_level: Optional[str] = None,
    enable_colors: Optional[bool] = None,
    log_format: Optional[str] = None,
    custom_colors: Optional[Dict[str, str]] = None,
    clear_existing: bool = True,
    use_env_config: bool = True,
) -> None:
    """Initialize logging for agentdemo and third-party libs.

    Args:
        level: Log level (INFO, DEBUG, etc.)
        log_file: Path to log file
        log_file_level: Log level for file handler
        enable_colors: Whether to enable colored output
        log_format: Custom log format string
        custom_colors: Custom color mapping
        clear_existing: Whether to clear existing handlers
        use_env_config: Whether to use environment variables for missing parameters
    """
    # If use_env_config is True and any parameter is None, configure from environment
    if use_env_config and any(param is None for param in [level, log_file, log_file_level, enable_colors]):
        env_config = _get_env_config()
        level = level or env_config["level"]
        log_file = log_file or env_config["log_file"]
        log_file_level = log_file_level or env_config["log_file_level"]
        enable_colors = enable_colors if enable_colors is not None else env_config["enable_colors"]

    # Set defaults for any remaining None values
    level = level or "INFO"
    log_file_level = log_file_level or "DEBUG"
    enable_colors = enable_colors if enable_colors is not None else True

    root_logger = logging.getLogger()
    if clear_existing:
        root_logger.handlers.clear()

    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    fmt = log_format or "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    if enable_colors and colorlog:
        formatter = colorlog.ColoredFormatter(
            fmt=fmt,
            datefmt=datefmt,
            log_colors=custom_colors
            or {
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
            style="%",
        )
    else:
        formatter = logging.Formatter(fmt.replace("%(log_color)s", ""), datefmt=datefmt)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, log_file_level.upper(), logging.DEBUG))
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def _get_env_config() -> Dict[str, any]:
    """Get logging configuration from environment variables."""
    return {
        "level": os.getenv("AGENTDEMO_LOG_LEVEL", "INFO"),
        "enable_colors": os.getenv("AGENTDEMO_LOG_COLORS", "true").lower() == "true",
        "log_file": os.getenv("AGENTDEMO_LOG_FILE"),
        "log_file_level": os.getenv("AGENTDEMO_LOG_FILE_LEVEL", "DEBUG"),
    }
