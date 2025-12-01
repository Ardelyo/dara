import torch

class Config:
    # Model Settings
    MODEL_ID = "microsoft/Florence-2-base"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Modes
    MODE_SCENE = "scene"
    MODE_EMOTION = "emotion"
    MODE_MEDICINE = "medicine"
    MODE_CURRENCY = "currency"
    MODE_TEXT = "text"

    # Prompts mapping
    PROMPTS = {
        MODE_SCENE: "<MORE_DETAILED_CAPTION>", # Detailed description instead of bounding boxes
        MODE_EMOTION: "<CAPTION>", # Captioning -> will infer emotion
        MODE_MEDICINE: "<OCR>", # OCR -> will parse for medicine info
        MODE_CURRENCY: "<CAPTION>", # Captioning -> will parse for currency info
        MODE_TEXT: "<OCR>" # OCR -> raw text
    }
