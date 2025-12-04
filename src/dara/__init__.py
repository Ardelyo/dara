"""
DARA - Detect & Assist Recognition AI
=====================================

A lightweight Vision Language Model for assistive technology.

"Mata untuk semua" (Eyes for everyone)

Quick Start:
    >>> from dara import DARA
    >>> dara = DARA()
    >>> result = dara.detect("photo.jpg", mode="scene")
    >>> print(result["result"])

Modes:
    - scene: Describes environment and objects
    - emotion: Reads facial expressions
    - medicine: Reads medicine labels and dosages
    - currency: Identifies currency (Indonesian Rupiah focus)
    - text: OCR for any text

For more information, see: https://github.com/ardelyo/dara
"""

__version__ = "0.2.0"
__author__ = "DARA Team"

# Core exports
from .core.model import DARA
from .config import Config, get_config, set_config

# Mode exports
from .modes import (
    BaseMode,
    ModeResult,
    SceneMode,
    EmotionMode,
    MedicineMode,
    CurrencyMode,
    TextMode,
)

# Service exports
from .services import TTSService, TranslationService, InferenceCache

# Utility exports
from .utils import setup_logging, get_logger

# Dataset export
from .dataset import DARADataset

# All public exports
__all__ = [
    # Core
    "DARA",
    "Config",
    "get_config",
    "set_config",
    # Modes
    "BaseMode",
    "ModeResult",
    "SceneMode",
    "EmotionMode",
    "MedicineMode",
    "CurrencyMode",
    "TextMode",
    # Services
    "TTSService",
    "TranslationService",
    "InferenceCache",
    # Utils
    "setup_logging",
    "get_logger",
    # Dataset
    "DARADataset",
]


def get_version() -> str:
    """Get DARA version string."""
    return __version__
