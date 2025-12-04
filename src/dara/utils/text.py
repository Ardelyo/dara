"""
DARA Utilities - Text Processing
Provides text cleaning, formatting, and manipulation utilities.
"""

import re
from typing import Optional


class TextUtils:
    """Utility class for text processing operations."""
    
    # Special tokens to remove
    SPECIAL_TOKENS = ["</s>", "<s>", "<pad>", "[PAD]", "[CLS]", "[SEP]"]
    
    @staticmethod
    def clean(text: str) -> str:
        """
        Remove special tokens and clean whitespace.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text string
        """
        text = str(text)
        
        # Remove special tokens
        for token in TextUtils.SPECIAL_TOKENS:
            text = text.replace(token, "")
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def extract_numbers(text: str) -> list[str]:
        """
        Extract all numbers from text.
        
        Args:
            text: Text to extract numbers from
            
        Returns:
            List of number strings found
        """
        pattern = r'\d+(?:[.,]\d+)*'
        return re.findall(pattern, text)
    
    @staticmethod
    def extract_dosage(text: str) -> list[str]:
        """
        Extract dosage patterns (e.g., 500mg, 10ml).
        
        Args:
            text: Text to search for dosage patterns
            
        Returns:
            List of dosage strings found
        """
        pattern = r'(\d+(?:[.,]\d+)?\s*(?:mg|ml|g|mcg|IU|tablet|capsule|cap|tab)s?)'
        return re.findall(pattern, text, re.IGNORECASE)
    
    @staticmethod
    def normalize_currency(text: str) -> str:
        """
        Normalize currency text by removing separators.
        
        Args:
            text: Currency text to normalize
            
        Returns:
            Normalized text without thousand separators
        """
        # Remove dots/commas used as thousand separators
        text = re.sub(r'(\d)\.(\d{3})', r'\1\2', text)
        text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
        return text
    
    @staticmethod
    def is_coherent(text: str, min_words: int = 3) -> bool:
        """
        Check if text appears to be coherent (not gibberish).
        
        Args:
            text: Text to check
            min_words: Minimum number of words for coherence
            
        Returns:
            True if text appears coherent
        """
        if not text or len(text.strip()) < 5:
            return False
        
        words = text.split()
        if len(words) < min_words:
            return False
        
        # Check for excessive special characters
        special_ratio = sum(1 for c in text if not c.isalnum() and c != ' ') / len(text)
        if special_ratio > 0.3:
            return False
        
        return True
    
    @staticmethod
    def truncate(text: str, max_length: int = 200, suffix: str = "...") -> str:
        """
        Truncate text to maximum length with suffix.
        
        Args:
            text: Text to truncate
            max_length: Maximum length including suffix
            suffix: Suffix to add if truncated
            
        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)].rsplit(' ', 1)[0] + suffix
