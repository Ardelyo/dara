# 📖 Panduan Training DARA | DARA Training Guide

[🇮🇩 Bahasa Indonesia](#bahasa-indonesia) | [🇺🇸 English](#english)

---

## Bahasa Indonesia

### Gambaran Umum Training

DARA menggunakan **LoRA (Low-Rank Adaptation)** untuk fine-tuning yang efisien, memungkinkan pelatihan pada GPU konsumer dengan memori terbatas.

### Persyaratan Sistem

| Komponen | Minimum | Direkomendasikan |
|----------|---------|------------------|
| **GPU VRAM** | 4 GB | 8+ GB |
| **RAM** | 8 GB | 16+ GB |
| **Storage** | 5 GB | 20+ GB |
| **Python** | 3.8+ | 3.10+ |

### Struktur Dataset

```
data/
├── train/
│   ├── dataset.json          # Metadata training
│   └── images/               # Folder gambar
│       ├── scene_001.jpg
│       ├── medicine_002.jpg
│       └── ...
└── eval/
    ├── dataset.json          # Metadata evaluasi
    └── images/
```

**Format `dataset.json`:**
```json
[
    {
        "image": "images/dapur_001.jpg",
        "mode": "scene",
        "text": "Dapur modern dengan meja kayu dan kompor gas yang menyala."
    },
    {
        "image": "images/obat_001.jpg",
        "mode": "medicine",
        "text": "Paracetamol 500mg. Diminum 3 kali sehari setelah makan."
    },
    {
        "image": "images/uang_50rb.jpg",
        "mode": "currency",
        "text": "Uang kertas Rp 50.000 warna biru dengan gambar I Gusti Ngurah Rai."
    }
]
```

### Menjalankan Training

```bash
# Dari direktori dara_project
cd dara_project

# Jalankan training
python scripts/train.py
```

### Konfigurasi LoRA

```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=16,                    # Rank dimensi (lebih tinggi = lebih ekspresif)
    lora_alpha=32,           # Faktor skala
    target_modules=[         # Layer yang di-tune
        "q_proj",            # Query projection
        "v_proj"             # Value projection
    ],
    lora_dropout=0.05,       # Dropout untuk regularisasi
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Parameter Training

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./dara_checkpoints",
    per_device_train_batch_size=4,    # Sesuaikan dengan VRAM
    gradient_accumulation_steps=4,     # Efektif batch = 16
    learning_rate=2e-5,
    num_train_epochs=10,
    save_steps=100,
    logging_steps=1,
    fp16=True,                         # Mixed precision
    remove_unused_columns=False
)
```

### Tips Optimasi

1. **VRAM Terbatas**: Kurangi `per_device_train_batch_size` ke 1-2
2. **Training Lebih Stabil**: Gunakan `gradient_accumulation_steps` lebih tinggi
3. **Overfitting**: Tingkatkan `lora_dropout` ke 0.1
4. **Underfitting**: Tingkatkan `r` ke 32 atau 64

### Evaluasi Model

```python
from dara import DARA

# Load model hasil training
dara = DARA(model_id="./dara_model_final")

# Evaluasi pada dataset test
result = dara.detect("test_image.jpg", mode="scene")
print(f"Confidence: {result['confidence']:.2f}")
```

---

## English

### Training Overview

DARA uses **LoRA (Low-Rank Adaptation)** for efficient fine-tuning, enabling training on consumer GPUs with limited memory.

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU VRAM** | 4 GB | 8+ GB |
| **RAM** | 8 GB | 16+ GB |
| **Storage** | 5 GB | 20+ GB |
| **Python** | 3.8+ | 3.10+ |

### Dataset Structure

```
data/
├── train/
│   ├── dataset.json          # Training metadata
│   └── images/               # Image folder
│       ├── scene_001.jpg
│       ├── medicine_002.jpg
│       └── ...
└── eval/
    ├── dataset.json          # Evaluation metadata
    └── images/
```

**`dataset.json` Format:**
```json
[
    {
        "image": "images/kitchen_001.jpg",
        "mode": "scene",
        "text": "Modern kitchen with wooden table and gas stove that is on."
    },
    {
        "image": "images/medicine_001.jpg",
        "mode": "medicine",
        "text": "Paracetamol 500mg. Take 3 times daily after meals."
    },
    {
        "image": "images/money_50k.jpg",
        "mode": "currency",
        "text": "Rp 50,000 banknote, blue color with I Gusti Ngurah Rai image."
    }
]
```

### Running Training

```bash
# From dara_project directory
cd dara_project

# Run training
python scripts/train.py
```

### LoRA Configuration

```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=16,                    # Rank dimension (higher = more expressive)
    lora_alpha=32,           # Scaling factor
    target_modules=[         # Layers to tune
        "q_proj",            # Query projection
        "v_proj"             # Value projection
    ],
    lora_dropout=0.05,       # Dropout for regularization
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Training Parameters

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./dara_checkpoints",
    per_device_train_batch_size=4,    # Adjust to VRAM
    gradient_accumulation_steps=4,     # Effective batch = 16
    learning_rate=2e-5,
    num_train_epochs=10,
    save_steps=100,
    logging_steps=1,
    fp16=True,                         # Mixed precision
    remove_unused_columns=False
)
```

### Optimization Tips

1. **Limited VRAM**: Reduce `per_device_train_batch_size` to 1-2
2. **Stable Training**: Use higher `gradient_accumulation_steps`
3. **Overfitting**: Increase `lora_dropout` to 0.1
4. **Underfitting**: Increase `r` to 32 or 64

### Model Evaluation

```python
from dara import DARA

# Load trained model
dara = DARA(model_id="./dara_model_final")

# Evaluate on test dataset
result = dara.detect("test_image.jpg", mode="scene")
print(f"Confidence: {result['confidence']:.2f}")
```

---

## 📊 Metrik Training | Training Metrics

| Metric | Nilai Awal | Setelah Training |
|--------|------------|------------------|
| **Loss** | ~2.5 | ~0.8 |
| **Accuracy (Scene)** | 65% | 85%+ |
| **Accuracy (Currency)** | 70% | 90%+ |
| **Inference Time** | ~500ms | ~200ms |
