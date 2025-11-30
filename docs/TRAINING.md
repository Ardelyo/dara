# DARA Training Guide

## Overview

This guide covers fine-tuning DARA on custom datasets using LoRA (Low-Rank Adaptation) for efficient training.

## Prerequisites

- Python 3.8+
- GPU with 8GB+ VRAM (recommended) or CPU
- ~100k training images across 5 modes

## Dataset Preparation

### 1. Data Collection

Recommended datasets:

| Mode | Dataset | Source | Size |
|------|---------|--------|------|
| **Scene** | COCO Captions | [COCO](https://cocodataset.org) | 50k |
| **Emotion** | FER2013 | [Kaggle](https://kaggle.com) | 30k |
| **Medicine** | Custom pharma labels | Manual collection | 5k |
| **Currency** | Custom bills | Manual collection | 3k |
| **Text** | ICDAR, TextOCR | [TextOCR](https://textvqa.org) | 20k |

### 2. Data Format

Create `dataset.json` in your data directory:

```json
[
  {
    "image": "images/scene_001.jpg",
    "mode": "scene",
    "text": "A modern kitchen with stainless steel appliances and marble countertops."
  },
  {
    "image": "images/emotion_001.jpg",
    "mode": "emotion",
    "text": "Person smiling happily"
  },
  {
    "image": "images/medicine_001.jpg",
    "mode": "medicine",
    "text": "Paracetamol 500mg, take 1-2 tablets every 4-6 hours"
  }
]
```

### 3. File Structure

```
data/
├── train/
│   ├── images/
│   │   ├── scene_001.jpg
│   │   ├── emotion_001.jpg
│   │   └── ...
│   └── dataset.json
└── eval/
    ├── images/
    └── dataset.json
```

## Training Configuration

### LoRA Settings

```python
peft_config = LoraConfig(
    r=16,                           # Rank (higher = more capacity)
    lora_alpha=32,                  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.05,              # Regularization
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Training Arguments

```python
training_args = TrainingArguments(
    output_dir="./dara_checkpoints",
    per_device_train_batch_size=4,   # Adjust based on VRAM
    gradient_accumulation_steps=4,   # Effective batch size = 16
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    fp16=True,                       # Mixed precision for speed
    evaluation_strategy="steps",
    eval_steps=100
)
```

## Running Training

### Basic Training

```bash
cd dara_project
python train.py
```

### Custom Configuration

Edit `train.py`:

```python
def train():
    # Modify batch size for different GPU memory
    training_args = TrainingArguments(
        per_device_train_batch_size=2,  # Lower for 4GB VRAM
        # ... other args
    )
```

### Monitor Training

```bash
# Install tensorboard
pip install tensorboard

# View logs
tensorboard --logdir ./dara_checkpoints/runs
```

## Memory Optimization

### For Limited VRAM

```python
# 1. Reduce batch size
per_device_train_batch_size=1

# 2. Increase gradient accumulation
gradient_accumulation_steps=16

# 3. Use 8-bit optimizers
from transformers import Trainer, TrainingArguments
import bitsandbytes as bnb

optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=2e-5)
```

### For CPU Training

```python
# In train.py
training_args = TrainingArguments(
    fp16=False,  # Disable mixed precision
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32
)
```

## Validation

### Evaluate on Test Set

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Evaluate
metrics = trainer.evaluate()
print(metrics)
```

### Manual Testing

```python
from model import DARA

dara = DARA()
# Load fine-tuned weights
dara.model.load_adapter("./dara_checkpoints/checkpoint-500")

result = dara.detect("test_image.jpg", mode="scene")
print(result["result"])
```

## Saving & Exporting

### Save LoRA Adapter

```python
# Automatically saved during training
# Location: ./dara_checkpoints/checkpoint-XXX/

# Manual save
model.save_pretrained("./dara_finetuned")
```

### Merge Adapter (Optional)

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base")
merged_model = PeftModel.from_pretrained(base_model, "./dara_checkpoints/checkpoint-500")
merged_model = merged_model.merge_and_unload()

# Save full model
merged_model.save_pretrained("./dara_full")
```

### Upload to Hugging Face

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./dara_finetuned",
    repo_id="yourusername/DARA-Lite",
    repo_type="model"
)
```

## Troubleshooting

### Issue: Training Loss Not Decreasing

**Solution:**
- Increase learning rate to `5e-5`
- Check data quality (images match descriptions?)
- Verify dataset.json format

### Issue: Out of Memory

**Solution:**
```python
# Clear cache
torch.cuda.empty_cache()

# Use gradient checkpointing
model.gradient_checkpointing_enable()
```

### Issue: Overfitting

**Solution:**
- Increase `lora_dropout` to `0.1`
- Reduce `num_train_epochs` to `2`
- Add more training data

## Best Practices

1. **Start Small**: Test with 100 examples first
2. **Validate Early**: Check outputs after 100 steps
3. **Use Checkpoints**: Resume from best checkpoint if training fails
4. **Monitor Metrics**: Watch loss curves in tensorboard
5. **Version Control**: Tag each experiment

## Advanced: Multi-GPU Training

```python
# Use accelerate
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

# Or use Trainer with DDP
python -m torch.distributed.launch --nproc_per_node=2 train.py
```

## Hyperparameter Tuning

### Grid Search Example

```python
learning_rates = [1e-5, 2e-5, 5e-5]
lora_ranks = [8, 16, 32]

for lr in learning_rates:
    for r in lora_ranks:
        # Run training
        # Evaluate
        # Save best config
```

## Performance Benchmarks

| Configuration | Batch Size | VRAM | Time/Epoch |
|---------------|------------|------|------------|
| RTX 3090 (24GB) | 16 | 18GB | 2 hours |
| RTX 3060 (12GB) | 8 | 10GB | 4 hours |
| CPU (32GB RAM) | 1 | N/A | 48 hours |

## Next Steps

After training:
1. Export to ONNX for mobile deployment
2. Quantize to INT8 for faster inference
3. Create model card on Hugging Face
4. Test with real users for feedback
