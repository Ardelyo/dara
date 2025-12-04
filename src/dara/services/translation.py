"""
DARA Services - Translation
Provides text translation with caching and fallback mechanisms.
"""

from typing import Optional
from functools import lru_cache

from ..utils.logging import get_logger

logger = get_logger("translation")


class TranslationService:
    """
    Translation service with caching and fallback.
    
    Supports translating between English and Indonesian using
    deep_translator with automatic fallback.
    """
    
    SUPPORTED_LANGUAGES = {"en", "id"}
    
    def __init__(self, cache_size: int = 500):
        """
        Initialize translation service.
        
        Args:
            cache_size: Maximum cached translations
        """
        self.cache_size = cache_size
        self._translator = None
        self._init_translator()
    
    def _init_translator(self) -> None:
        """Initialize the translator backend."""
        try:
            from deep_translator import GoogleTranslator
            self._translator_class = GoogleTranslator
            logger.info("Translation service initialized with GoogleTranslator")
        except ImportError:
            logger.warning("deep_translator not available, translation disabled")
            self._translator_class = None
    
    @lru_cache(maxsize=500)
    def translate(
        self,
        text: str,
        source: str = "auto",
        target: str = "id"
    ) -> str:
        """
        Translate text between languages.
        
        Args:
            text: Text to translate
            source: Source language code ('auto' for detection)
            target: Target language code
            
        Returns:
            Translated text (or original if translation fails)
        """
        if not text or not self._translator_class:
            return text
        
        if target not in self.SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported target language: {target}")
            return text
        
        try:
            translator = self._translator_class(source=source, target=target)
            result = translator.translate(text)
            logger.debug(f"Translated to {target}: {text[:50]}... -> {result[:50]}...")
            return result
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return text
    
    def to_indonesian(self, text: str) -> str:
        """Shortcut to translate to Indonesian."""
        return self.translate(text, source="auto", target="id")
    
    def to_english(self, text: str) -> str:
        """Shortcut to translate to English."""
        return self.translate(text, source="auto", target="en")
    
    def clear_cache(self) -> None:
        """Clear translation cache."""
        self.translate.cache_clear()
        logger.info("Translation cache cleared")
    
    @property
    def cache_info(self) -> dict:
        """Get cache statistics."""
        info = self.translate.cache_info()
        return {
            "hits": info.hits,
            "misses": info.misses,
            "size": info.currsize,
            "maxsize": info.maxsize
        }
