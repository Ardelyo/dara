# 🪔 DARA — Detect & Assist Recognition AI

> **"Mata untuk semua" (Eyes for everyone)**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg)](https://huggingface.co/DARA)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)]()

[🇺🇸 English](#english) | [🇮🇩 Bahasa Indonesia](#bahasa-indonesia)

---

## English

### 🎯 Mission

**DARA** is an open-source, lightweight Vision Language Model (VLM) designed for assistive technology. It provides a "second pair of eyes" for visually impaired individuals, elderly users, and the general public.

**Key Features:**
- 🚀 **Ultra-fast**: 300-500ms inference on CPU
- 📱 **Mobile-ready**: Runs on phones and edge devices
- 🧠 **5 Smart Modes**: Scene, Emotion, Medicine, Currency, Text
- 🔊 **Voice Output**: Integrated TTS for accessibility
- 🌐 **Offline**: Works without internet connection
- 🇮🇩 **Bilingual**: English and Indonesian support

### 🌟 The 5 Smart Modes

| Mode | Icon | Function | Example Output |
|------|------|----------|----------------|
| **Scene** | 🏞️ | Describes environment + hazard detection | "Kitchen with table. ⚠️ Stove is on." |
| **Emotion** | 😊 | Reads facial expressions + social guidance | "Happy. They seem approachable." |
| **Medicine** | 💊 | Reads labels + dosage extraction | "Paracetamol 500mg. Take after meals." |
| **Currency** | 💵 | IDR detection with colors | "Rp 50.000 (blue color)" |
| **Text** | 📝 | OCR for any text | "Exit sign. Door to your left." |

### 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/ardelyo/dara.git
cd dara/dara_project

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run demo
python app.py
```

### 💻 Basic Usage

```python
from dara import DARA

# Initialize
dara = DARA()

# Detect with mode  
result = dara.detect("photo.jpg", mode="scene", language="en")

print(result["result"])      # "Modern kitchen with table..."
print(result["confidence"])  # 0.85
print(result["audio"])       # Path to TTS audio
```

### 📊 Performance

| Metric | Value |
|--------|-------|
| Import Time | ~16s (first load) |
| Inference (CPU) | 300-500ms |
| Cache Hit | <1ms |
| Memory | ~1.5GB |

---

## Bahasa Indonesia

### 🎯 Misi

**DARA** adalah Model Vision-Language (VLM) open-source yang ringan, dirancang untuk teknologi asistif. Menyediakan "sepasang mata kedua" untuk tunanetra, lansia, dan masyarakat umum.

**Fitur Utama:**
- 🚀 **Ultra-cepat**: Inferensi 300-500ms di CPU
- 📱 **Siap Mobile**: Berjalan di HP dan edge device
- 🧠 **5 Mode Cerdas**: Scene, Emosi, Obat, Mata Uang, Teks
- 🔊 **Output Suara**: TTS terintegrasi untuk aksesibilitas
- 🌐 **Offline**: Bekerja tanpa koneksi internet
- 🇮🇩 **Bilingual**: Dukungan Inggris dan Indonesia

### 🌟 5 Mode Cerdas

| Mode | Ikon | Fungsi | Contoh Output |
|------|------|--------|---------------|
| **Scene** | 🏞️ | Deskripsi lingkungan + deteksi bahaya | "Dapur dengan meja. ⚠️ Kompor menyala." |
| **Emotion** | 😊 | Baca ekspresi + saran sosial | "Senang. Terlihat ramah." |
| **Medicine** | 💊 | Baca label + ekstraksi dosis | "Paracetamol 500mg. Minum setelah makan." |
| **Currency** | 💵 | Deteksi Rupiah dengan warna | "Rp 50.000 (warna biru)" |
| **Text** | 📝 | OCR untuk teks apapun | "Tanda keluar. Pintu di kiri." |

### 🚀 Mulai Cepat

```bash
# Clone repository
git clone https://github.com/ardelyo/dara.git
cd dara/dara_project

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Jalankan demo
python app.py
```

### 💻 Penggunaan Dasar

```python
from dara import DARA

# Inisialisasi
dara = DARA()

# Deteksi dengan mode
result = dara.detect("foto.jpg", mode="scene", language="id")

print(result["result"])      # "Dapur modern dengan meja..."
print(result["confidence"])  # 0.85
print(result["audio"])       # Path ke audio TTS
```

### 📊 Performa

| Metrik | Nilai |
|--------|-------|
| Waktu Import | ~16 detik (muat pertama) |
| Inferensi (CPU) | 300-500ms |
| Cache Hit | <1ms |
| Memori | ~1.5GB |

---

## 📁 Project Structure | Struktur Proyek

```
dara_project/
├── src/dara/              # Main package
│   ├── core/              # Model, processor, inference
│   ├── modes/             # 5 mode handlers
│   ├── services/          # TTS, translation, cache
│   ├── utils/             # Utilities
│   ├── config.py          # Configuration
│   └── __init__.py        # Public API
├── demo/                  # Demo applications
├── scripts/               # Training & benchmark
├── docs/                  # Documentation
│   ├── QUICKSTART.md      # ⭐ Start here!
│   ├── API.md             # API reference
│   ├── ARCHITECTURE.md    # System design
│   ├── TRAINING.md        # Training guide
│   ├── RESEARCH.md        # Research paper
│   ├── STATISTICS.md      # Benchmark data
│   └── CONTEXT.md         # Vision & roadmap
├── data/                  # Training data
└── tests/                 # Unit tests
```

## 📚 Documentation | Dokumentasi

| Document | Description | Deskripsi |
|----------|-------------|-----------|
| [QUICKSTART.md](docs/QUICKSTART.md) | Quick start guide | Panduan mulai cepat |
| [API.md](docs/API.md) | API reference | Referensi API |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design | Desain sistem |
| [TRAINING.md](docs/TRAINING.md) | Training guide | Panduan training |
| [RESEARCH.md](docs/RESEARCH.md) | Research paper | Makalah penelitian |
| [STATISTICS.md](docs/STATISTICS.md) | Benchmark data | Data benchmark |
| [CONTEXT.md](docs/CONTEXT.md) | Vision & roadmap | Visi & roadmap |

## 🔢 Version History | Riwayat Versi

| Version | Date | Changes |
|---------|------|---------|
| **0.2.0** | Dec 2024 | Modular architecture, caching, bilingual docs |
| **0.1.0** | Nov 2024 | Initial release with 5 modes |

## 🤝 Contributing | Kontribusi

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Kontribusi dipersilakan! Lihat [CONTRIBUTING.md](CONTRIBUTING.md) untuk panduan.

## 📄 License | Lisensi

Apache 2.0 - See [LICENSE](LICENSE) for details.

---

**Built with ❤️ for accessibility | Dibangun dengan ❤️ untuk aksesibilitas**

*"Mata untuk semua" | "Eyes for everyone"*
