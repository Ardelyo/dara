"""
DARA Model - Backward Compatibility Wrapper

This module provides backward compatibility for code that imports
from the old structure (from dara.model import DARA).

The actual implementation is now in dara.core.model.
"""

# Re-export from new location
from .core.model import DARA

__all__ = ["DARA"]
