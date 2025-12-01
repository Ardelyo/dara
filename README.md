# 🪔 DARA — Detect & Assist Recognition AI

> **"Mata untuk semua" (Eyes for everyone)**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg)](https://huggingface.co/DARA)

[🇺🇸 English README](README.md) | [🇮🇩 Bahasa Indonesia](README_ID.md)

## 🎯 Mission

**DARA** is an open-source, lightweight Vision Language Model (VLM) designed for assistive technology. It provides a "second pair of eyes" for visually impaired individuals, elderly users, and the general public.

**Key Features:**
- 🚀 **Ultra-fast**: <200ms inference on CPU
- 📱 **Mobile-ready**: Runs on phones and edge devices
- 🧠 **5 Smart Modes**: Scene, Emotion, Medicine, Currency, Text
- 🔊 **Voice Output**: Integrated TTS for accessibility
- 🌐 **Open Source**: Full transparency and community-driven

## 🌟 The 5 Modes

| Mode | Icon | Function | Example Output |
|------|------|----------|----------------|
| **Scene** | 👁️ | Describes environment | "Kitchen with table and chairs. Stove is on." |
| **Emotion** | 😊 | Reads facial expressions | "Person looks happy. They seem approachable." |
| **Medicine** | 💊 | Reads medicine labels | "Paracetamol 500mg. Take after meals." |
| **Currency** | 💵 | Identifies money | "Rp 50,000 note. Blue color." |
| **Text** | 📖 | OCR for any text | "Exit sign detected. Door to your left." |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/ardelyo/dara.git
cd dara/dara_project

# Install dependencies
pip install -r requirements.txt
```

### Run the Demo

```bash
python app.py
```

The Gradio interface will launch at `http://localhost:7860`

### Basic Usage

```python
from model import DARA

# Initialize
dara = DARA()

# Detect and assist
result = dara.detect(
    image_path="photo.jpg",
    mode="scene"  # or "emotion", "medicine", "currency", "text"
)

print(result["result"])  # Smart assist output
# Audio saved to: result["audio"]
```

## 📊 Model Lineup

| Model | Base | Params | Size | Target Device | Status |
|-------|------|--------|------|---------------|--------|
| **DARA-Lite** | Florence-2 | 0.23B | ~500MB | Mobile, CPU | ✅ Available |
| **DARA** | SmolVLM | 1B | ~1GB | Laptop | 🔄 Coming Soon |
| **DARA-Pro** | Qwen2-VL | 2-3B | ~2GB | GPU | 🔄 Planned |

## 📁 Project Structure

```
dara_project/
├── config.py          # Configuration & mode definitions
├── model.py           # Core DARA class
├── app.py             # Gradio web interface
├── dataset.py         # Multi-task dataset loader
├── train.py           # LoRA fine-tuning script
├── docs/              # Documentation
│   ├── ARCHITECTURE.md
│   ├── API.md
│   └── TRAINING.md
└── requirements.txt   # Dependencies
```

## 🛠️ Tech Stack

- **Framework**: PyTorch, Transformers
- **Base Model**: Microsoft Florence-2-base
- **Fine-tuning**: LoRA (PEFT)
- **Interface**: Gradio
- **TTS**: gTTS
- **Optimization**: ONNX, INT8 quantization (planned)

## 📚 Documentation

- [Architecture](docs/ARCHITECTURE.md) - Technical design and model details
- [API Reference](docs/API.md) - Complete API documentation
- [Training Guide](docs/TRAINING.md) - How to fine-tune DARA
- [Context & Vision](docs/CONTEXT.md) - Project philosophy and roadmap

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

Apache 2.0 License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Microsoft for [Florence-2](https://huggingface.co/microsoft/Florence-2-base)
- Hugging Face for the transformers library
- The open-source community

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/ardelyo/dara/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ardelyo/dara/discussions)

---

**Made with ❤️ for accessibility**
