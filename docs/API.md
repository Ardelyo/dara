# DARA API Reference

## Core Classes

### `DARA`

Main class for loading and running inference.

#### Constructor

```python
DARA()
```

Initializes the DARA model by loading Florence-2-base and processor.

**Parameters:** None

**Attributes:**
- `device` (str): "cuda" or "cpu"
- `model` (AutoModelForCausalLM): Loaded Florence-2 model
- `processor` (AutoProcessor): Florence-2 processor

**Example:**
```python
from model import DARA
dara = DARA()
```

---

#### `detect(image_path, mode)`

Performs detection and generates smart assist output.

**Parameters:**
- `image_path` (str): Path to input image
- `mode` (str): Detection mode. Options:
  - `"scene"` - Scene description
  - `"emotion"` - Emotion detection
  - `"medicine"` - Medicine label reading
  - `"currency"` - Currency identification
  - `"text"` - OCR text reading

**Returns:**
- `dict`: 
  ```python
  {
      "mode": str,      # Selected mode
      "result": str,    # Smart assist output
      "audio": str      # Path to TTS audio file
  }
  ```

**Example:**
```python
result = dara.detect("photo.jpg", mode="scene")
print(result["result"])
# Output: "Scene Description: Kitchen with table and chairs. Stove is on."
```

**Raises:**
- `FileNotFoundError`: If image_path doesn't exist
- `PIL.UnidentifiedImageError`: If file is not a valid image

---

#### `_process_output(parsed_answer, mode, task_prompt)` (Private)

Applies smart assist logic to raw model output.

**Parameters:**
- `parsed_answer` (dict): Raw model output
- `mode` (str): Detection mode
- `task_prompt` (str): Florence-2 task prompt

**Returns:**
- `str`: Processed, human-friendly output

---

#### `_generate_audio(text)` (Private)

Generates TTS audio from text.

**Parameters:**
- `text` (str): Text to convert to speech

**Returns:**
- `str`: Path to saved MP3 file or `None` if error

---

## Configuration (`config.py`)

### `Config`

Static configuration class.

**Attributes:**
- `MODEL_ID` (str): Hugging Face model identifier
- `DEVICE` (str): Compute device ("cuda" or "cpu")
- `torch_dtype` (torch.dtype): FP16 (GPU) or FP32 (CPU)

**Mode Constants:**
- `MODE_SCENE` = `"scene"`
- `MODE_EMOTION` = `"emotion"`
- `MODE_MEDICINE` = `"medicine"`
- `MODE_CURRENCY` = `"currency"`
- `MODE_TEXT` = `"text"`

**Prompt Mapping:**
```python
PROMPTS = {
    "scene": "<CAPTION>",
    "emotion": "<CAPTION>",
    "medicine": "<OCR>",
    "currency": "<CAPTION>",
    "text": "<OCR>"
}
```

---

## Dataset (`dataset.py`)

### `DARADataset`

PyTorch Dataset for multi-task training.

#### Constructor

```python
DARADataset(data_root, processor, split="train")
```

**Parameters:**
- `data_root` (str): Path to dataset directory
- `processor` (AutoProcessor): Florence-2 processor
- `split` (str): "train" or "eval"

**Data Format:**

Expected `dataset.json`:
```json
[
  {
    "image": "images/sample1.jpg",
    "mode": "scene",
    "text": "A modern kitchen with appliances."
  }
]
```

---

## Training (`train.py`)

### `train()`

Runs LoRA fine-tuning.

**Configuration:**
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)

TrainingArguments(
    output_dir="./dara_checkpoints",
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    num_train_epochs=3
)
```

**Usage:**
```bash
python train.py
```

---

## Gradio App (`app.py`)

### `process_image(image, mode)`

Gradio callback function.

**Parameters:**
- `image` (PIL.Image): Uploaded image
- `mode` (str): Selected mode

**Returns:**
- `tuple`: (result_text, audio_path)

**Launch:**
```bash
python app.py
```

---

## CLI Example

```python
import argparse
from model import DARA

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mode", default="scene")
    args = parser.parse_args()
    
    dara = DARA()
    result = dara.detect(args.image, args.mode)
    print(result["result"])

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
python cli.py --image photo.jpg --mode emotion
```

---

## Error Handling

### Common Errors

**1. CUDA Out of Memory**
```python
# Solution: Force CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

**2. Model Download Issues**
```python
# Solution: Manual download
from transformers import AutoModelForCausalLM
AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-base",
    cache_dir="./models"
)
```

**3. TTS Fails**
```python
# Fallback: Skip audio
result = dara.detect(image, mode)
text_only = result["result"]  # Ignore audio
```

---

## Performance Tips

1. **Batch Processing**: Process multiple images in sequence
2. **Quantization**: Use INT8 for faster inference (future)
3. **GPU Usage**: Set `CUDA_VISIBLE_DEVICES` for multi-GPU systems
