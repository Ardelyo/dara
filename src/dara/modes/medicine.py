"""
DARA Modes - Medicine Reading Handler
Provides intelligent medicine label and dosage reading.
"""

import re
from .base import BaseMode, ModeResult
from ..utils.text import TextUtils


class MedicineMode(BaseMode):
    """
    Medicine reading mode for medication safety.
    
    Reads medicine labels, extracts dosage information,
    and provides safety reminders.
    """
    
    # Common medicine keywords
    MEDICINE_KEYWORDS = [
        "tablet", "capsule", "syrup", "drops", "cream", "ointment",
        "injection", "inhaler", "patch", "suspension", "solution",
        "obat", "tablet", "kapsul", "sirup", "tetes", "krim", "salep"
    ]
    
    # Dosage units
    DOSAGE_UNITS = ["mg", "ml", "g", "mcg", "IU", "unit", "tablet", "capsule", "cap", "tab"]
    
    @property
    def name(self) -> str:
        return self.MODE_MEDICINE
    
    @property
    def prompt(self) -> str:
        return "<OCR>"
    
    @property
    def description(self) -> str:
        return "Reads medicine labels and extracts dosage information"
    
    def process(self, raw_output: str, language: str = "en") -> ModeResult:
        """
        Process medicine OCR output.
        
        Extracts:
        - Medicine name (if identifiable)
        - Dosage information
        - Usage instructions
        - Expiry date (if visible)
        """
        text = self.preprocess(raw_output)
        
        # Extract structured information
        dosages = self._extract_dosages(text)
        instructions = self._extract_instructions(text)
        expiry = self._extract_expiry(text)
        
        # Build output
        output_parts = []
        patterns_matched = 0
        
        if dosages:
            dosage_text = ", ".join(dosages)
            output_parts.append(f"Dosage: {dosage_text}")
            patterns_matched += len(dosages)
        
        if instructions:
            output_parts.append(f"Instructions: {instructions}")
            patterns_matched += 1
        
        if expiry:
            output_parts.append(f"Expiry: {expiry}")
            patterns_matched += 1
        
        if not output_parts:
            output_parts.append(f"Text found: {TextUtils.truncate(text, 150)}")
        
        output_text = ". ".join(output_parts)
        
        # Translate if needed
        output_text = self.translate_if_needed(output_text, language)
        
        # Calculate confidence
        confidence = self.calculate_confidence(text, patterns_matched)
        
        # Safety suggestions
        suggestions = self._get_safety_suggestions(language)
        
        return ModeResult(
            text=output_text,
            confidence=confidence,
            raw_output=raw_output,
            metadata={
                "dosages": dosages,
                "instructions": instructions,
                "expiry": expiry,
                "is_medicine": self._is_likely_medicine(text)
            },
            suggestions=suggestions
        )
    
    def _extract_dosages(self, text: str) -> list:
        """Extract dosage patterns from text."""
        # Pattern: number followed by unit
        pattern = r'(\d+(?:[.,]\d+)?\s*(?:' + '|'.join(self.DOSAGE_UNITS) + r')s?)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        # Clean and deduplicate
        cleaned = []
        seen = set()
        for match in matches:
            normalized = match.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                cleaned.append(match.strip())
        
        return cleaned[:5]  # Limit to 5 dosages
    
    def _extract_instructions(self, text: str) -> str:
        """Extract usage instructions."""
        instruction_patterns = [
            r'(take\s+\d+\s+(?:time|tablet|capsule|cap|tab)s?\s*(?:a\s+day|daily|per\s+day)?)',
            r'(minum\s+\d+\s+(?:kali|tablet|kapsul)\s*(?:sehari)?)',
            r'((?:before|after|with)\s+(?:meal|food|breakfast|lunch|dinner)s?)',
            r'((?:sebelum|sesudah|bersama)\s+makan)',
            r'(every\s+\d+\s+hours?)',
            r'(setiap\s+\d+\s+jam)'
        ]
        
        for pattern in instruction_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_expiry(self, text: str) -> str:
        """Extract expiry date."""
        # Common expiry patterns
        patterns = [
            r'(?:exp(?:iry)?|ed|best\s+before)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(?:exp(?:iry)?|ed|best\s+before)[:\s]*(\w+\s+\d{4})',
            r'(\d{2}/\d{4})'  # MM/YYYY format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _is_likely_medicine(self, text: str) -> bool:
        """Check if text is likely from medicine packaging."""
        text_lower = text.lower()
        
        # Check for medicine keywords
        keyword_count = sum(1 for kw in self.MEDICINE_KEYWORDS if kw in text_lower)
        
        # Check for dosage patterns
        has_dosage = bool(self._extract_dosages(text))
        
        return keyword_count >= 1 or has_dosage
    
    def _get_safety_suggestions(self, language: str) -> list:
        """Get medicine safety suggestions."""
        if language == "id":
            return [
                "Selalu konsultasikan dengan dokter",
                "Periksa tanggal kadaluarsa",
                "Baca petunjuk penggunaan lengkap"
            ]
        return [
            "Always consult a doctor for exact dosage",
            "Check expiry date before use",
            "Read full instructions on packaging"
        ]
