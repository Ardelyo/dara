import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from .config import Config

class DARADataset(Dataset):
    def __init__(self, data_root, processor, split="train"):
        self.data_root = data_root
        self.processor = processor
        self.split = split
        self.data = []
        
        # Mock loading logic - in real scenario, load from JSON/CSV
        # Structure: {"image": "path/to/img.jpg", "mode": "scene", "text": "A kitchen..."}
        if os.path.exists(os.path.join(data_root, "dataset.json")):
            with open(os.path.join(data_root, "dataset.json"), "r") as f:
                self.data = json.load(f)
        else:
            print("Warning: dataset.json not found. Using empty dataset.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.data_root, item["image"])
        mode = item["mode"]
        ground_truth = item["text"]
        
        image = Image.open(image_path).convert("RGB")
        prompt = Config.PROMPTS.get(mode, "<OD>")
        
        # 1. Process Image
        # Note: We don't pass text here to avoid the OverflowError in the combined call
        image_inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"]
        
        # 2. Process Text (Prompt)
        # We explicitly call the tokenizer
        text_inputs = self.processor.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=64
        )
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]
        
        # 3. Process Labels
        labels = self.processor.tokenizer(
            ground_truth, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=64
        ).input_ids
        
        return {
            "input_ids": input_ids.squeeze(),
            "attention_mask": attention_mask.squeeze(),
            "pixel_values": pixel_values.squeeze(),
            "labels": labels.squeeze()
        }
