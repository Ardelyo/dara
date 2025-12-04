"""
DARA Core - Main Model Class
Refactored DARA model with modular architecture.
"""

import torch
from PIL import Image
from typing import Union, Optional, Dict, Any
from pathlib import Path

from transformers import AutoProcessor, AutoModelForCausalLM

from ..config import Config, get_config
from ..modes import (
    BaseMode, ModeResult,
    SceneMode, EmotionMode, MedicineMode, CurrencyMode, TextMode
)
from ..services.tts import TTSService
from ..services.cache import InferenceCache
from ..utils.image import ImageUtils
from ..utils.logging import get_logger, setup_logging

logger = get_logger("model")


class DARA:
    """
    DARA - Detect & Assist Recognition AI
    
    A lightweight Vision Language Model for assistive technology.
    
    Features:
    - 5 intelligent detection modes (scene, emotion, medicine, currency, text)
    - Integrated text-to-speech
    - Inference caching for performance
    - Bilingual support (English/Indonesian)
    
    Example:
        >>> dara = DARA()
        >>> result = dara.detect("photo.jpg", mode="scene")
        >>> print(result["result"])
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        config: Optional[Config] = None,
        enable_tts: bool = True,
        enable_cache: bool = True,
        log_level: str = "INFO"
    ):
        """
        Initialize DARA model.
        
        Args:
            model_id: Hugging Face model ID (overrides config)
            config: Configuration object (uses default if None)
            enable_tts: Enable text-to-speech output
            enable_cache: Enable inference caching
            log_level: Logging level
        """
        # Setup logging
        setup_logging(level=log_level)
        
        # Get configuration
        self.config = config or get_config()
        self.model_id = model_id or self.config.model.model_id
        self.device = self.config.device
        self.torch_dtype = self.config.torch_dtype
        
        logger.info(f"Initializing DARA ({self.model_id})...")
        logger.info(f"Device: {self.device}, Dtype: {self.torch_dtype}")
        
        # Load model and processor
        self._load_model()
        
        # Initialize mode handlers
        self._init_modes()
        
        # Initialize services
        self.tts = TTSService(
            cache_dir=self.config.tts.cache_dir,
            rate=self.config.tts.rate,
            enable_cache=self.config.tts.cache_audio
        ) if enable_tts else None
        
        # Initialize inference cache
        self.cache_enabled = enable_cache
        self.cache = InferenceCache(
            maxsize=self.config.inference.cache_size
        ) if enable_cache else None
        
        logger.info("DARA initialized successfully!")
    
    def _load_model(self) -> None:
        """Load the model and processor."""
        logger.info("Loading model...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.config.model.trust_remote_code,
            attn_implementation=self.config.model.attn_implementation
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        logger.info("Model loaded successfully")
    
    def _init_modes(self) -> None:
        """Initialize all mode handlers."""
        self.modes: Dict[str, BaseMode] = {
            self.config.MODE_SCENE: SceneMode(),
            self.config.MODE_EMOTION: EmotionMode(),
            self.config.MODE_MEDICINE: MedicineMode(),
            self.config.MODE_CURRENCY: CurrencyMode(),
            self.config.MODE_TEXT: TextMode(),
        }
        logger.debug(f"Initialized {len(self.modes)} mode handlers")
    
    @torch.inference_mode()
    def detect(
        self,
        image_input: Union[str, Path, Image.Image],
        mode: str = "scene",
        language: str = "en",
        generate_audio: bool = True
    ) -> Dict[str, Any]:
        """
        Detect and assist based on the selected mode.
        
        Args:
            image_input: Path to image or PIL Image object
            mode: Detection mode (scene, emotion, medicine, currency, text)
            language: Output language code ('en' or 'id')
            generate_audio: Whether to generate TTS audio
            
        Returns:
            Dictionary with:
                - mode: Selected mode
                - result: Text output
                - confidence: Confidence score
                - audio: Path to audio file (if generated)
                - language: Output language
                - metadata: Additional mode-specific data
        """
        # Validate mode
        if mode not in self.modes:
            available = ", ".join(self.modes.keys())
            raise ValueError(f"Invalid mode '{mode}'. Available: {available}")
        
        mode_handler = self.modes[mode]
        
        # Load image
        image = ImageUtils.load(image_input, convert_rgb=True)
        
        # Check cache
        if self.cache_enabled:
            image_hash = ImageUtils.compute_hash(image)
            cache_key = f"{mode}:{language}"
            cached = self.cache.get(image_hash, cache_key)
            if cached:
                logger.debug(f"Cache hit for {mode}")
                return cached
        
        # Get prompt for this mode
        prompt = mode_handler.prompt
        
        # Prepare inputs
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device, self.torch_dtype)
        
        # Generate
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=self.config.inference.max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=False  # Disable to prevent errors
        )
        
        # Decode
        generated_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=False
        )[0]
        
        # Post-process through HF processor
        try:
            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task=prompt,
                image_size=(image.width, image.height)
            )
            raw_output = parsed_answer.get(prompt, generated_text)
        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")
            raw_output = generated_text
        
        # Process through mode handler
        if isinstance(raw_output, dict):
            raw_output = str(raw_output)
        
        mode_result: ModeResult = mode_handler.process(raw_output, language)
        
        # Generate audio
        audio_path = None
        if generate_audio and self.tts and self.tts.is_available:
            audio_path = self.tts.generate(mode_result.text, language)
        
        # Build result
        result = {
            "mode": mode,
            "result": mode_result.text,
            "confidence": mode_result.confidence,
            "audio": audio_path,
            "language": language,
            "metadata": mode_result.metadata,
            "suggestions": mode_result.suggestions
        }
        
        # Cache result
        if self.cache_enabled:
            self.cache.set(image_hash, cache_key, result)
        
        return result
    
    def detect_all(
        self,
        image_input: Union[str, Path, Image.Image],
        language: str = "en"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all detection modes on an image.
        
        Args:
            image_input: Path to image or PIL Image
            language: Output language
            
        Returns:
            Dictionary with results for each mode
        """
        results = {}
        for mode in self.modes:
            try:
                results[mode] = self.detect(
                    image_input, 
                    mode=mode, 
                    language=language,
                    generate_audio=False  # Skip audio for batch
                )
            except Exception as e:
                logger.error(f"Error in {mode} mode: {e}")
                results[mode] = {"error": str(e)}
        
        return results
    
    def get_available_modes(self) -> list:
        """Get list of available detection modes."""
        return list(self.modes.keys())
    
    def clear_cache(self) -> int:
        """Clear inference cache. Returns count of cleared entries."""
        if self.cache:
            return self.cache.clear()
        return 0
    
    @property
    def cache_stats(self) -> Optional[dict]:
        """Get cache statistics."""
        if self.cache:
            return self.cache.stats
        return None
    
    def __repr__(self) -> str:
        return f"<DARA(model={self.model_id}, device={self.device})>"
