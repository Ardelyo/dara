"""
DARA Modes - Emotion Detection Handler
Provides emotion analysis from facial expressions.
"""

from .base import BaseMode, ModeResult


class EmotionMode(BaseMode):
    """
    Emotion detection mode for social awareness.
    
    Analyzes facial expressions and body language to
    infer emotional state and provide social guidance.
    """
    
    # Emotion mappings with keywords and advice
    EMOTIONS = {
        "happy": {
            "keywords": ["smile", "smiling", "happy", "laugh", "joy", "cheerful", "grinning"],
            "advice_en": "They seem in good spirits!",
            "advice_id": "Mereka terlihat senang!"
        },
        "sad": {
            "keywords": ["sad", "cry", "crying", "tear", "upset", "frown", "depressed", "down"],
            "advice_en": "Offer comfort or support.",
            "advice_id": "Tawarkan dukungan atau hibur mereka."
        },
        "angry": {
            "keywords": ["angry", "mad", "furious", "shout", "yelling", "aggressive", "frustrated"],
            "advice_en": "Give them space or ask calmly.",
            "advice_id": "Beri mereka ruang atau tanya dengan tenang."
        },
        "fearful": {
            "keywords": ["fear", "scared", "afraid", "terror", "frightened", "anxious", "worried"],
            "advice_en": "Reassure them that they are safe.",
            "advice_id": "Yakinkan mereka bahwa mereka aman."
        },
        "surprised": {
            "keywords": ["surprise", "surprised", "shocked", "amazed", "astonished"],
            "advice_en": "Something unexpected happened.",
            "advice_id": "Sesuatu yang tidak terduga terjadi."
        },
        "neutral": {
            "keywords": ["neutral", "calm", "serious", "focused"],
            "advice_en": "Ask how they are doing.",
            "advice_id": "Tanyakan kabar mereka."
        }
    }
    
    @property
    def name(self) -> str:
        return self.MODE_EMOTION
    
    @property
    def prompt(self) -> str:
        return "<CAPTION>"
    
    @property
    def description(self) -> str:
        return "Detects emotions from facial expressions and provides social guidance"
    
    def process(self, raw_output: str, language: str = "en") -> ModeResult:
        """
        Process emotion detection output.
        
        Analyzes the scene description for emotional cues
        and provides appropriate social guidance.
        """
        text = self.preprocess(raw_output)
        text_lower = text.lower()
        
        # Detect emotion and get advice
        detected_emotion, confidence_boost = self._detect_emotion(text_lower)
        emotion_data = self.EMOTIONS.get(detected_emotion, self.EMOTIONS["neutral"])
        
        # Format output
        advice = emotion_data[f"advice_{language}"] if language == "id" else emotion_data["advice_en"]
        
        if language == "id":
            emotion_names = {
                "happy": "Senang",
                "sad": "Sedih",
                "angry": "Marah",
                "fearful": "Takut",
                "surprised": "Terkejut",
                "neutral": "Netral"
            }
            emotion_display = emotion_names.get(detected_emotion, "Netral")
            output_text = f"{emotion_display}. {advice}"
        else:
            output_text = f"{detected_emotion.title()}. {advice}"
        
        # Calculate confidence
        base_confidence = self.calculate_confidence(text)
        confidence = min(1.0, base_confidence + confidence_boost)
        
        return ModeResult(
            text=output_text,
            confidence=confidence,
            raw_output=raw_output,
            metadata={
                "detected_emotion": detected_emotion,
                "all_detected": self._get_all_emotions(text_lower)
            },
            suggestions=self._get_suggestions(detected_emotion, language)
        )
    
    def _detect_emotion(self, text: str) -> tuple:
        """
        Detect primary emotion from text.
        
        Returns:
            Tuple of (emotion_name, confidence_boost)
        """
        best_match = "neutral"
        max_matches = 0
        
        for emotion, data in self.EMOTIONS.items():
            matches = sum(1 for kw in data["keywords"] if kw in text)
            if matches > max_matches:
                max_matches = matches
                best_match = emotion
        
        # Confidence boost based on matches
        confidence_boost = min(0.3, max_matches * 0.1)
        
        return best_match, confidence_boost
    
    def _get_all_emotions(self, text: str) -> list:
        """Get all detected emotions with counts."""
        results = []
        for emotion, data in self.EMOTIONS.items():
            matches = sum(1 for kw in data["keywords"] if kw in text)
            if matches > 0:
                results.append({"emotion": emotion, "strength": matches})
        return sorted(results, key=lambda x: x["strength"], reverse=True)
    
    def _get_suggestions(self, emotion: str, language: str) -> list:
        """Get contextual suggestions based on emotion."""
        suggestions = {
            "happy": {
                "en": ["Good time for conversation", "They may be receptive to requests"],
                "id": ["Waktu yang baik untuk berbicara", "Mereka mungkin terbuka untuk permintaan"]
            },
            "sad": {
                "en": ["Speak gently", "Ask if they need anything"],
                "id": ["Bicara dengan lembut", "Tanyakan apakah mereka butuh sesuatu"]
            },
            "angry": {
                "en": ["Keep calm", "Avoid confrontation"],
                "id": ["Tetap tenang", "Hindari konfrontasi"]
            },
            "fearful": {
                "en": ["Speak softly", "Explain what's happening"],
                "id": ["Bicara dengan lembut", "Jelaskan apa yang terjadi"]
            }
        }
        
        lang_key = "id" if language == "id" else "en"
        return suggestions.get(emotion, {}).get(lang_key, [])
