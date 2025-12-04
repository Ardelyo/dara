"""
DARA Modes - Scene Description Handler
Provides detailed scene description with contextual understanding.
"""

from .base import BaseMode, ModeResult


class SceneMode(BaseMode):
    """
    Scene description mode for environmental awareness.
    
    Provides detailed descriptions of the scene, objects,
    spatial relationships, and potential hazards.
    """
    
    @property
    def name(self) -> str:
        return self.MODE_SCENE
    
    @property
    def prompt(self) -> str:
        return "<MORE_DETAILED_CAPTION>"
    
    @property
    def description(self) -> str:
        return "Describes the scene with objects, people, and spatial context"
    
    def process(self, raw_output: str, language: str = "en") -> ModeResult:
        """
        Process scene description output.
        
        Enhances the raw caption with:
        - Better formatting
        - Safety hints if hazards detected
        - Navigation suggestions
        """
        text = self.preprocess(raw_output)
        
        # Analyze for safety concerns
        hazards = self._detect_hazards(text)
        suggestions = []
        
        if hazards:
            suggestions.extend([
                f"Caution: {hazard} detected" for hazard in hazards
            ])
        
        # Add navigation context
        nav_hints = self._extract_navigation(text)
        suggestions.extend(nav_hints)
        
        # Translate if needed
        text = self.translate_if_needed(text, language)
        
        # Calculate confidence
        confidence = self.calculate_confidence(text)
        
        return ModeResult(
            text=text,
            confidence=confidence,
            raw_output=raw_output,
            metadata={
                "hazards_detected": hazards,
                "has_people": self._has_people(raw_output)
            },
            suggestions=suggestions
        )
    
    def _detect_hazards(self, text: str) -> list:
        """Detect potential hazards mentioned in scene."""
        hazard_keywords = [
            "stairs", "step", "fire", "flame", "stove", "water", "pool",
            "edge", "cliff", "hole", "wet", "slippery", "sharp", "hot",
            "tangga", "api", "air", "tepi", "basah", "licin", "tajam", "panas"
        ]
        
        text_lower = text.lower()
        found = []
        
        for keyword in hazard_keywords:
            if keyword in text_lower:
                found.append(keyword)
        
        return found[:3]  # Limit to top 3
    
    def _extract_navigation(self, text: str) -> list:
        """Extract navigation-relevant hints."""
        nav_keywords = {
            "door": "Door detected",
            "exit": "Exit sign visible",
            "left": "Object on the left",
            "right": "Object on the right",
            "table": "Table nearby",
            "chair": "Chair in scene",
            "pintu": "Pintu terdeteksi",
            "kiri": "Objek di kiri",
            "kanan": "Objek di kanan"
        }
        
        text_lower = text.lower()
        hints = []
        
        for keyword, hint in nav_keywords.items():
            if keyword in text_lower:
                hints.append(hint)
        
        return hints[:3]  # Limit hints
    
    def _has_people(self, text: str) -> bool:
        """Check if people are mentioned in scene."""
        people_keywords = [
            "person", "people", "man", "woman", "child", "group",
            "orang", "pria", "wanita", "anak", "kelompok"
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in people_keywords)
