"""
DARA Services - Text-to-Speech
Provides TTS generation with caching and multiple engine support.
"""

from typing import Optional
from pathlib import Path
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor

from ..utils.logging import get_logger

logger = get_logger("tts")


class TTSService:
    """
    Text-to-speech service with caching.
    
    Supports pyttsx3 (offline) as primary engine with
    optional output caching to avoid regenerating audio.
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        rate: int = 150,
        enable_cache: bool = True
    ):
        """
        Initialize TTS service.
        
        Args:
            cache_dir: Directory for cached audio files
            rate: Speech rate (words per minute)
            enable_cache: Whether to cache generated audio
        """
        self.rate = rate
        self.enable_cache = enable_cache
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache/tts")
        
        self._engine = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._init_engine()
        
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_engine(self) -> None:
        """Initialize the TTS engine."""
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', self.rate)
            logger.info(f"TTS engine initialized with rate={self.rate}")
        except Exception as e:
            logger.warning(f"Failed to initialize TTS engine: {e}")
            self._engine = None
    
    def _get_cache_key(self, text: str, language: str) -> str:
        """Generate cache key for text+language."""
        content = f"{text}:{language}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path for cached audio file."""
        return self.cache_dir / f"{cache_key}.mp3"
    
    def _select_voice(self, language: str) -> Optional[str]:
        """Select appropriate voice for language."""
        if not self._engine:
            return None
        
        try:
            voices = self._engine.getProperty('voices')
            
            for voice in voices:
                voice_name = voice.name.lower()
                if language == "id" and "indonesia" in voice_name:
                    return voice.id
                elif language == "en" and "english" in voice_name:
                    return voice.id
            
            # Default to first available voice
            return voices[0].id if voices else None
        except Exception as e:
            logger.warning(f"Voice selection failed: {e}")
            return None
    
    def generate(
        self,
        text: str,
        language: str = "en",
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate TTS audio for text.
        
        Args:
            text: Text to convert to speech
            language: Language code ('en' or 'id')
            output_path: Optional custom output path
            
        Returns:
            Path to generated audio file, or None on failure
        """
        if not self._engine or not text:
            return None
        
        # Check cache first
        if self.enable_cache:
            cache_key = self._get_cache_key(text, language)
            cache_path = self._get_cache_path(cache_key)
            
            if cache_path.exists():
                logger.debug(f"TTS cache hit: {cache_key[:8]}...")
                return str(cache_path)
        
        # Generate new audio
        try:
            # Select voice for language
            voice_id = self._select_voice(language)
            if voice_id:
                self._engine.setProperty('voice', voice_id)
            
            # Determine output path
            if output_path:
                save_path = Path(output_path)
            elif self.enable_cache:
                save_path = cache_path
            else:
                save_path = Path(f"output_{uuid.uuid4().hex}.mp3")
            
            # Generate audio
            self._engine.save_to_file(text, str(save_path))
            self._engine.runAndWait()
            
            logger.debug(f"Generated TTS audio: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return None
    
    def generate_async(self, text: str, language: str = "en") -> "Future":
        """
        Generate TTS audio asynchronously.
        
        Args:
            text: Text to convert
            language: Language code
            
        Returns:
            Future that resolves to audio path
        """
        return self._executor.submit(self.generate, text, language)
    
    def clear_cache(self) -> int:
        """
        Clear all cached audio files.
        
        Returns:
            Number of files deleted
        """
        if not self.cache_dir.exists():
            return 0
        
        count = 0
        for audio_file in self.cache_dir.glob("*.mp3"):
            try:
                audio_file.unlink()
                count += 1
            except Exception:
                pass
        
        logger.info(f"Cleared {count} cached audio files")
        return count
    
    @property
    def is_available(self) -> bool:
        """Check if TTS engine is available."""
        return self._engine is not None
    
    def __del__(self):
        """Cleanup executor on destruction."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
