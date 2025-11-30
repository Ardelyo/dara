import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from config import Config

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
        
        # Get prompt for the mode
        prompt = Config.PROMPTS.get(mode, "<OD>")
        
        # Prepare inputs for Florence-2
        # Note: Florence-2 training format usually involves input_ids and labels
        inputs = self.processor(
            text=prompt, 
            images=image, 
            return_tensors="pt",
            padding="max_length",
            max_length=128
        )
        
        # Process labels (simplified for demo)
        labels = self.processor.tokenizer(
            ground_truth, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=128
        ).input_ids
        
        inputs["labels"] = labels
        
        return {k: v.squeeze() for k, v in inputs.items()}
