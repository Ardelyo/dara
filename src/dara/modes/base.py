"""
DARA Modes - Base Mode Handler
Abstract base class for all intelligent mode handlers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any

from ..utils.text import TextUtils


@dataclass
class ModeResult:
    """
    Structured result from mode processing.
    
    Attributes:
        text: Final processed text output
        confidence: Confidence score (0.0 - 1.0)
        raw_output: Original model output
        metadata: Mode-specific metadata
        suggestions: Helpful suggestions for user
    """
    text: str
    confidence: float
    raw_output: str
    metadata: dict = field(default_factory=dict)
    suggestions: list = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "raw_output": self.raw_output,
            "metadata": self.metadata,
            "suggestions": self.suggestions
        }


class BaseMode(ABC):
    """
    Abstract base class for intelligent mode handlers.
    
    Each mode handler is responsible for:
    1. Defining the appropriate prompt for the model
    2. Post-processing raw model output
    3. Calculating confidence scores
    4. Generating user-friendly output
    """
    
    # Mode identifiers
    MODE_SCENE = "scene"
    MODE_EMOTION = "emotion"
    MODE_MEDICINE = "medicine"
    MODE_CURRENCY = "currency"
    MODE_TEXT = "text"
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Mode identifier string."""
        pass
    
    @property
    @abstractmethod
    def prompt(self) -> str:
        """Florence-2 task prompt for this mode."""
        pass
    
    @property
    def description(self) -> str:
        """Human-readable description of this mode."""
        return f"{self.name.title()} detection mode"
    
    @abstractmethod
    def process(self, raw_output: str, language: str = "en") -> ModeResult:
        """
        Process raw model output into structured result.
        
        Args:
            raw_output: Raw text from model
            language: Output language code ('en' or 'id')
            
        Returns:
            Structured ModeResult
        """
        pass
    
    def preprocess(self, raw_output: Any) -> str:
        """
        Preprocess raw output to clean string.
        
        Args:
            raw_output: Raw model output (may be dict or string)
            
        Returns:
            Cleaned string
        """
        if isinstance(raw_output, dict):
            # Extract text from dict output
            raw_output = str(raw_output.get(self.prompt, raw_output))
        
        return TextUtils.clean(str(raw_output))
    
    def calculate_confidence(self, text: str, patterns_matched: int = 0) -> float:
        """
        Calculate confidence score based on output quality.
        
        Args:
            text: Processed text
            patterns_matched: Number of expected patterns found
            
        Returns:
            Confidence score (0.0 - 1.0)
        """
        base_confidence = 0.5
        
        # Boost for text coherence
        if TextUtils.is_coherent(text, min_words=3):
            base_confidence += 0.2
        
        # Boost for pattern matches
        if patterns_matched > 0:
            base_confidence += min(0.3, patterns_matched * 0.1)
        
        # Penalty for very short or very long output
        if len(text) < 10:
            base_confidence -= 0.2
        elif len(text) > 500:
            base_confidence -= 0.1
        
        return max(0.0, min(1.0, base_confidence))
    
    def translate_if_needed(self, text: str, language: str) -> str:
        """
        Translate text if language is not English.
        
        Args:
            text: Text to potentially translate
            language: Target language code
            
        Returns:
            Translated or original text
        """
        if language == "en":
            return text
        
        try:
            from ..services.translation import TranslationService
            translator = TranslationService()
            return translator.translate(text, target=language)
        except Exception:
            return text
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, prompt={self.prompt})>"
