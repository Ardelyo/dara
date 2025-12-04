"""
DARA Utilities - Logging Configuration
Provides structured logging with configurable levels and formatting.
"""

import logging
import sys
from typing import Optional
from pathlib import Path

# Default format for log messages
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Module-level logger cache
_loggers: dict = {}


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Configure the root logger for DARA.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to write logs to file
        format_string: Optional custom format string
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    fmt = format_string or LOG_FORMAT
    
    # Configure root logger
    root_logger = logging.getLogger("dara")
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(fmt, datefmt=DATE_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Optional file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    root_logger.info(f"DARA logging initialized at {level} level")


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the given name.
    
    Args:
        name: Logger name (will be prefixed with 'dara.')
    
    Returns:
        Configured logger instance
    """
    full_name = f"dara.{name}" if not name.startswith("dara.") else name
    
    if full_name not in _loggers:
        logger = logging.getLogger(full_name)
        _loggers[full_name] = logger
    
    return _loggers[full_name]


class LoggerMixin:
    """Mixin class to add logging capability to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger
