"""
DARA Modes - Text Extraction Handler
Provides OCR text extraction and formatting.
"""

from .base import BaseMode, ModeResult
from ..utils.text import TextUtils


class TextMode(BaseMode):
    """
    Text extraction mode for general OCR.
    
    Reads and formats text from images including
    signs, documents, and printed materials.
    """
    
    @property
    def name(self) -> str:
        return self.MODE_TEXT
    
    @property
    def prompt(self) -> str:
        return "<OCR>"
    
    @property
    def description(self) -> str:
        return "Extracts and reads text from images"
    
    def process(self, raw_output: str, language: str = "en") -> ModeResult:
        """
        Process OCR text extraction output.
        
        Cleans and formats extracted text for
        clear audio presentation.
        """
        text = self.preprocess(raw_output)
        
        if not text or len(text.strip()) < 2:
            if language == "id":
                output_text = "Tidak ada teks yang terdeteksi."
            else:
                output_text = "No text detected."
            confidence = 0.1
        else:
            # Format text for clear presentation
            output_text = self._format_for_speech(text)
            
            # Translate if needed
            output_text = self.translate_if_needed(output_text, language)
            
            # Calculate confidence based on text quality
            confidence = self._calculate_text_confidence(text)
        
        # Analyze text type
        text_type = self._detect_text_type(text)
        
        return ModeResult(
            text=output_text,
            confidence=confidence,
            raw_output=raw_output,
            metadata={
                "character_count": len(text),
                "word_count": len(text.split()),
                "text_type": text_type,
                "has_numbers": any(c.isdigit() for c in text)
            },
            suggestions=self._get_suggestions(text_type, language)
        )
    
    def _format_for_speech(self, text: str) -> str:
        """Format text for clear speech output."""
        # Break long text into readable chunks
        if len(text) > 200:
            text = TextUtils.truncate(text, 200)
        
        # Improve readability
        text = text.replace('\n', '. ')
        text = text.replace('  ', ' ')
        
        # Handle common abbreviations
        abbreviations = {
            'Jl.': 'Jalan',
            'Jl': 'Jalan',
            'No.': 'Nomor',
            'Tlp': 'Telepon',
            'Hp': 'Handphone',
        }
        for abbr, full in abbreviations.items():
            text = text.replace(abbr, full)
        
        return text.strip()
    
    def _calculate_text_confidence(self, text: str) -> float:
        """Calculate confidence based on text characteristics."""
        confidence = 0.5
        
        # Coherent text gets boost
        if TextUtils.is_coherent(text, min_words=2):
            confidence += 0.2
        
        # Long text with words is reliable
        word_count = len(text.split())
        if word_count >= 3:
            confidence += 0.1
        if word_count >= 5:
            confidence += 0.1
        
        # Penalize gibberish-looking text
        alpha_ratio = sum(1 for c in text if c.isalnum()) / max(len(text), 1)
        if alpha_ratio < 0.5:
            confidence -= 0.2
        
        return max(0.1, min(1.0, confidence))
    
    def _detect_text_type(self, text: str) -> str:
        """Detect the type of text content."""
        text_lower = text.lower()
        
        # Check for sign patterns
        if any(word in text_lower for word in ['exit', 'entrance', 'warning', 'danger', 'keluar', 'masuk', 'awas', 'bahaya']):
            return "sign"
        
        # Check for address patterns
        if any(word in text_lower for word in ['jalan', 'jl', 'street', 'no.', 'blok', 'lantai']):
            return "address"
        
        # Check for phone/contact
        if any(pattern in text for pattern in ['08', '+62', 'telepon', 'phone', 'email', '@']):
            return "contact"
        
        # Check for menu/price
        if any(word in text_lower for word in ['rp', 'harga', 'price', 'menu', '$', '€']):
            return "menu_price"
        
        return "general"
    
    def _get_suggestions(self, text_type: str, language: str) -> list:
        """Get contextual suggestions based on text type."""
        suggestions_map = {
            "sign": {
                "en": ["Sign detected - follow directions"],
                "id": ["Tanda terdeteksi - ikuti petunjuk"]
            },
            "address": {
                "en": ["Address detected - useful for navigation"],
                "id": ["Alamat terdeteksi - berguna untuk navigasi"]
            },
            "contact": {
                "en": ["Contact information detected"],
                "id": ["Informasi kontak terdeteksi"]
            },
            "menu_price": {
                "en": ["Menu or price list detected"],
                "id": ["Menu atau daftar harga terdeteksi"]
            },
            "general": {
                "en": ["General text extracted"],
                "id": ["Teks umum diekstrak"]
            }
        }
        
        lang_key = "id" if language == "id" else "en"
        return suggestions_map.get(text_type, suggestions_map["general"]).get(lang_key, [])
