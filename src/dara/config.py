"""
DARA Configuration
Centralized configuration management with dataclasses.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model-specific configuration."""
    model_id: str = "microsoft/Florence-2-base"
    use_flash_attention: bool = False
    trust_remote_code: bool = True
    attn_implementation: str = "eager"


@dataclass
class InferenceConfig:
    """Inference optimization settings."""
    enable_cache: bool = True
    cache_size: int = 100
    max_new_tokens: int = 256
    quantization: str = "none"  # "none", "fp16", "int8"
    max_image_size: int = 1024


@dataclass
class TTSConfig:
    """Text-to-speech settings."""
    engine: str = "pyttsx3"  # "pyttsx3", "gtts"
    rate: int = 150
    cache_audio: bool = True
    cache_dir: str = ".cache/tts"


@dataclass
class Config:
    """
    Main DARA configuration.
    
    Usage:
        config = Config()
        config = Config.from_env()
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    
    # Device auto-detection
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Dtype based on device
    torch_dtype: torch.dtype = field(
        default_factory=lambda: torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Mode constants
    MODE_SCENE: str = "scene"
    MODE_EMOTION: str = "emotion"
    MODE_MEDICINE: str = "medicine"
    MODE_CURRENCY: str = "currency"
    MODE_TEXT: str = "text"
    
    # Legacy prompts mapping (for backward compatibility)
    PROMPTS: dict = field(default_factory=lambda: {
        "scene": "<MORE_DETAILED_CAPTION>",
        "emotion": "<CAPTION>",
        "medicine": "<OCR>",
        "currency": "<OCR>",
        "text": "<OCR>"
    })
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            model=ModelConfig(
                model_id=os.getenv("DARA_MODEL_ID", "microsoft/Florence-2-base"),
            ),
            inference=InferenceConfig(
                enable_cache=os.getenv("DARA_ENABLE_CACHE", "true").lower() == "true",
                cache_size=int(os.getenv("DARA_CACHE_SIZE", "100")),
                quantization=os.getenv("DARA_QUANTIZATION", "none"),
            ),
            tts=TTSConfig(
                engine=os.getenv("DARA_TTS_ENGINE", "pyttsx3"),
                rate=int(os.getenv("DARA_TTS_RATE", "150")),
            ),
        )
    
    @property  
    def MODEL_ID(self) -> str:
        """Legacy property for backward compatibility."""
        return self.model.model_id
    
    @property
    def DEVICE(self) -> str:
        """Legacy property for backward compatibility."""
        return self.device


# Global default config instance
_default_config: Optional[Config] = None


def get_config() -> Config:
    """Get the default config instance."""
    global _default_config
    if _default_config is None:
        _default_config = Config()
    return _default_config


def set_config(config: Config) -> None:
    """Set the default config instance."""
    global _default_config
    _default_config = config
