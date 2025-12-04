"""
DARA Core - Inference Engine
High-performance inference engine with optimization and caching.
"""

import torch
from typing import Optional, Dict, Any
from PIL import Image

from ..utils.logging import get_logger
from ..services.cache import InferenceCache

logger = get_logger("inference")


class InferenceEngine:
    """
    Optimized inference engine for DARA model.
    
    Features:
    - FP16/INT8 quantization support
    - LRU inference caching
    - Configurable generation parameters
    - Batch inference support
    """
    
    DEFAULT_GEN_CONFIG = {
        "max_new_tokens": 256,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": False,  # Disable KV cache to prevent errors
    }
    
    def __init__(
        self,
        model,
        processor: "ImageProcessor",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        enable_cache: bool = True,
        cache_size: int = 100,
        quantization: str = "none"
    ):
        """
        Initialize inference engine.
        
        Args:
            model: The loaded model
            processor: ImageProcessor instance
            device: Target device
            dtype: Model dtype
            enable_cache: Enable inference caching
            cache_size: Maximum cache entries
            quantization: Quantization mode ("none", "fp16", "int8")
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.dtype = dtype
        self.quantization = quantization
        
        # Setup caching
        self.cache_enabled = enable_cache
        self.cache = InferenceCache(maxsize=cache_size) if enable_cache else None
        
        # Apply quantization
        self._apply_quantization()
        
        logger.info(
            f"InferenceEngine initialized "
            f"(device={device}, quantization={quantization}, cache={enable_cache})"
        )
    
    def _apply_quantization(self) -> None:
        """Apply quantization based on configuration."""
        if self.quantization == "fp16" and self.device != "cpu":
            self.model = self.model.half()
            logger.info("Applied FP16 quantization")
        elif self.quantization == "int8":
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                logger.info("Applied INT8 dynamic quantization")
            except Exception as e:
                logger.warning(f"INT8 quantization failed, using default: {e}")
    
    @torch.inference_mode()
    def generate(
        self,
        image_input,
        prompt: str,
        **gen_kwargs
    ) -> str:
        """
        Generate text from image with caching.
        
        Args:
            image_input: Image path or PIL Image
            prompt: Task prompt
            **gen_kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Check cache first
        if self.cache_enabled:
            image_hash = self.processor.get_image_hash(image_input)
            cached = self.cache.get(image_hash, prompt)
            if cached:
                logger.debug("Using cached inference result")
                return cached
        
        # Get image size for post-processing
        image = self.processor.hf_processor.image_processor
        if isinstance(image_input, Image.Image):
            image_size = image_input.size
        else:
            from PIL import Image as PILImage
            with PILImage.open(image_input) as img:
                image_size = img.size
        
        # Prepare inputs
        inputs = self.processor.prepare(image_input, prompt)
        
        # Merge generation config
        gen_config = {**self.DEFAULT_GEN_CONFIG, **gen_kwargs}
        
        # Generate
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            **gen_config
        )
        
        # Decode
        generated_text = self.processor.decode(generated_ids)[0]
        
        # Post-process
        parsed = self.processor.post_process(generated_text, prompt, image_size)
        result = parsed.get(prompt, generated_text)
        
        # Convert dict to string if needed
        if isinstance(result, dict):
            result = str(result)
        
        # Cache result
        if self.cache_enabled:
            self.cache.set(image_hash, prompt, result)
        
        return result
    
    @torch.inference_mode()
    def generate_batch(
        self,
        images: list,
        prompts: list,
        **gen_kwargs
    ) -> list:
        """
        Generate text for multiple images.
        
        Args:
            images: List of image inputs
            prompts: List of prompts
            **gen_kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        # Prepare batch inputs
        inputs = self.processor.prepare_batch(images, prompts)
        
        # Merge generation config
        gen_config = {**self.DEFAULT_GEN_CONFIG, **gen_kwargs}
        
        # Generate
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            **gen_config
        )
        
        # Decode all
        return self.processor.decode(generated_ids)
    
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
