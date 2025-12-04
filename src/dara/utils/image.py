"""
DARA Utilities - Image Processing
Provides image loading, preprocessing, and optimization utilities.
"""

from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional
import hashlib
import io


class ImageUtils:
    """Utility class for image processing operations."""
    
    # Supported image formats
    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    
    # Default processing settings
    DEFAULT_MAX_SIZE = 1024
    DEFAULT_QUALITY = 85
    
    @staticmethod
    def load(
        image_input: Union[str, Path, Image.Image],
        convert_rgb: bool = True
    ) -> Image.Image:
        """
        Load image from various input types.
        
        Args:
            image_input: Path string, Path object, or PIL Image
            convert_rgb: Whether to convert to RGB mode
            
        Returns:
            PIL Image object
            
        Raises:
            ValueError: If input type is not supported
        """
        if isinstance(image_input, Image.Image):
            image = image_input
        elif isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            image = Image.open(path)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        if convert_rgb and image.mode != "RGB":
            image = image.convert("RGB")
        
        return image
    
    @staticmethod
    def resize_smart(
        image: Image.Image,
        max_size: int = DEFAULT_MAX_SIZE,
        maintain_aspect: bool = True
    ) -> Image.Image:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: PIL Image to resize
            max_size: Maximum dimension (width or height)
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Resized PIL Image
        """
        if max(image.size) <= max_size:
            return image
        
        if maintain_aspect:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
        else:
            new_size = (max_size, max_size)
        
        return image.resize(new_size, Image.Resampling.LANCZOS)
    
    @staticmethod
    def compute_hash(image: Image.Image, size: int = 8) -> str:
        """
        Compute perceptual hash for image (for caching).
        
        Args:
            image: PIL Image to hash
            size: Hash grid size
            
        Returns:
            Hexadecimal hash string
        """
        # Resize to small size for hashing
        img_small = image.resize((size, size), Image.Resampling.LANCZOS)
        img_gray = img_small.convert("L")
        
        # Convert to bytes and hash
        img_bytes = img_gray.tobytes()
        return hashlib.md5(img_bytes).hexdigest()
    
    @staticmethod
    def to_bytes(
        image: Image.Image,
        format: str = "JPEG",
        quality: int = DEFAULT_QUALITY
    ) -> bytes:
        """
        Convert PIL Image to bytes.
        
        Args:
            image: PIL Image to convert
            format: Output format (JPEG, PNG, etc.)
            quality: Compression quality (for JPEG)
            
        Returns:
            Image as bytes
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format, quality=quality)
        return buffer.getvalue()
    
    @staticmethod
    def get_info(image: Image.Image) -> dict:
        """
        Get image metadata/info.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with image info
        """
        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format,
            "size_pixels": image.width * image.height,
            "aspect_ratio": round(image.width / image.height, 2)
        }
    
    @staticmethod
    def validate(image_input: Union[str, Path]) -> Tuple[bool, Optional[str]]:
        """
        Validate if file is a valid image.
        
        Args:
            image_input: Path to image file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        path = Path(image_input)
        
        if not path.exists():
            return False, f"File not found: {path}"
        
        if path.suffix.lower() not in ImageUtils.SUPPORTED_FORMATS:
            return False, f"Unsupported format: {path.suffix}"
        
        try:
            with Image.open(path) as img:
                img.verify()
            return True, None
        except Exception as e:
            return False, f"Invalid image: {e}"
