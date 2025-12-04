"""
DARA Core - Image Processor
Optimized image preprocessing pipeline for model inference.
"""

import torch
from PIL import Image
from typing import Union, Optional
from pathlib import Path

from ..utils.image import ImageUtils
from ..utils.logging import get_logger

logger = get_logger("processor")


class ImageProcessor:
    """
    Optimized image preprocessing for DARA inference.
    
    Handles image loading, resizing, and tensor conversion
    with caching of preprocessed inputs.
    """
    
    def __init__(
        self,
        hf_processor,
        max_size: int = 1024,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize image processor.
        
        Args:
            hf_processor: Hugging Face processor (AutoProcessor)
            max_size: Maximum image dimension
            device: Target device for tensors
            dtype: Target dtype for tensors
        """
        self.hf_processor = hf_processor
        self.max_size = max_size
        self.device = device
        self.dtype = dtype
        
        logger.info(f"ImageProcessor initialized (device={device}, dtype={dtype})")
    
    def prepare(
        self,
        image_input: Union[str, Path, Image.Image],
        prompt: str
    ) -> dict:
        """
        Prepare image and prompt for model inference.
        
        Args:
            image_input: Image path or PIL Image
            prompt: Task prompt for the model
            
        Returns:
            Dictionary with input_ids, pixel_values, etc.
        """
        # Load and preprocess image
        image = ImageUtils.load(image_input, convert_rgb=True)
        
        # Resize if needed
        if max(image.size) > self.max_size:
            image = ImageUtils.resize_smart(image, self.max_size)
            logger.debug(f"Resized image to {image.size}")
        
        # Process through HF processor
        inputs = self.hf_processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {
            key: value.to(self.device, self.dtype) 
            if value.dtype in [torch.float16, torch.float32, torch.float64]
            else value.to(self.device)
            for key, value in inputs.items()
        }
        
        return inputs
    
    def prepare_batch(
        self,
        images: list,
        prompts: list
    ) -> dict:
        """
        Prepare batch of images for inference.
        
        Args:
            images: List of image inputs
            prompts: List of prompts (same length as images)
            
        Returns:
            Batched inputs dictionary
        """
        if len(images) != len(prompts):
            raise ValueError("Images and prompts must have same length")
        
        # Process all images
        processed_images = []
        for img in images:
            image = ImageUtils.load(img, convert_rgb=True)
            if max(image.size) > self.max_size:
                image = ImageUtils.resize_smart(image, self.max_size)
            processed_images.append(image)
        
        # Batch process
        inputs = self.hf_processor(
            text=prompts,
            images=processed_images,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {
            key: value.to(self.device, self.dtype)
            if value.dtype in [torch.float16, torch.float32, torch.float64]
            else value.to(self.device)
            for key, value in inputs.items()
        }
        
        return inputs
    
    def get_image_hash(self, image_input: Union[str, Path, Image.Image]) -> str:
        """
        Compute hash for image (for caching).
        
        Args:
            image_input: Image to hash
            
        Returns:
            Hash string
        """
        image = ImageUtils.load(image_input, convert_rgb=True)
        return ImageUtils.compute_hash(image)
    
    def decode(self, generated_ids: torch.Tensor) -> list:
        """
        Decode generated token IDs to text.
        
        Args:
            generated_ids: Generated token IDs
            
        Returns:
            List of decoded strings
        """
        return self.hf_processor.batch_decode(
            generated_ids, 
            skip_special_tokens=False
        )
    
    def post_process(
        self,
        generated_text: str,
        task_prompt: str,
        image_size: tuple
    ) -> dict:
        """
        Post-process generated text.
        
        Args:
            generated_text: Raw generated text
            task_prompt: Task prompt used
            image_size: Original image (width, height)
            
        Returns:
            Parsed answer dictionary
        """
        try:
            return self.hf_processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=image_size
            )
        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")
            return {task_prompt: generated_text}
