# 🪔 DARA — Detect & Assist Recognition AI

> **"Mata untuk semua"**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg)](https://huggingface.co/DARA)

[🇺🇸 English README](README.md) | [🇮🇩 Bahasa Indonesia](README_ID.md)

## 🎯 Misi

**DARA** adalah Vision Language Model (VLM) open-source yang ringan, dirancang untuk teknologi asistif. DARA menyediakan "mata kedua" bagi penyandang tunanetra, lansia, dan masyarakat umum.

**Fitur Utama:**
- 🚀 **Super Cepat**: Inferensi <200ms di CPU
- 📱 **Siap Mobile**: Berjalan di ponsel dan perangkat edge
- 🧠 **5 Mode Pintar**: Pemandangan, Emosi, Obat, Mata Uang, Teks
- 🔊 **Output Suara**: TTS terintegrasi (Bahasa Indonesia & Inggris)
- 🌐 **Open Source**: Transparansi penuh dan berbasis komunitas

## 🌟 5 Mode Pintar

| Mode | Ikon | Fungsi | Contoh Output |
|------|------|--------|---------------|
| **Pemandangan** | 👁️ | Mendeskripsikan lingkungan | "Dapur dengan meja dan kursi. Kompor menyala." |
| **Emosi** | 😊 | Membaca ekspresi wajah | "Orang terlihat bahagia. Mereka tampak ramah." |
| **Obat** | 💊 | Membaca label obat | "Paracetamol 500mg. Minum setelah makan." |
| **Mata Uang** | 💵 | Mengidentifikasi uang | "Uang Rp 50.000. Warna biru." |
| **Teks** | 📖 | OCR untuk teks apa pun | "Tanda keluar terdeteksi. Pintu di sebelah kiri Anda." |

## 🚀 Mulai Cepat

### Instalasi

```bash
# Clone repositori
git clone https://github.com/ardelyo/dara.git
cd dara/dara_project

# Instal dependensi
pip install -r requirements.txt
```

### Jalankan Demo

```bash
python app.py
```

Antarmuka Gradio akan terbuka di `http://localhost:7860`

### Penggunaan Dasar

```python
from model import DARA

# Inisialisasi
dara = DARA()

# Deteksi dan bantu (dengan Bahasa Indonesia)
result = dara.detect(
    image_path="foto.jpg",
    mode="scene",  # atau "emotion", "medicine", "currency", "text"
    language="id"  # Output Bahasa Indonesia
)

print(result["result"])  # Output bantuan cerdas
# Audio tersimpan di: result["audio"]
```

## 📊 Jajaran Model

| Model | Basis | Parameter | Ukuran | Perangkat Target | Status |
|-------|-------|-----------|--------|------------------|--------|
| **DARA-Lite** | Florence-2 | 0.23B | ~500MB | Ponsel, CPU | ✅ Tersedia |
| **DARA** | SmolVLM | 1B | ~1GB | Laptop | 🔄 Segera |
| **DARA-Pro** | Qwen2-VL | 2-3B | ~2GB | GPU | 🔄 Direncanakan |

## 📁 Struktur Proyek

```
dara_project/
├── config.py          # Konfigurasi & definisi mode
├── model.py           # Kelas inti DARA
├── app.py             # Antarmuka web Gradio
├── dataset.py         # Pemuat dataset multi-task
├── train.py           # Skript fine-tuning LoRA
├── publish_to_hub.py  # Skrip upload ke Hugging Face
├── docs/              # Dokumentasi
└── requirements.txt   # Dependensi
```

## 🛠️ Teknologi

- **Framework**: PyTorch, Transformers
- **Model Dasar**: Microsoft Florence-2-base
- **Fine-tuning**: LoRA (PEFT)
- **Antarmuka**: Gradio
- **TTS**: gTTS
- **Terjemahan**: Deep Translator

## 🤝 Berkontribusi

Kami menyambut kontribusi! Silakan lihat [CONTRIBUTING.md](CONTRIBUTING.md) untuk panduan.

## 📄 Lisensi

Lisensi Apache 2.0 - lihat [LICENSE](LICENSE) untuk detailnya.

## 🙏 Ucapan Terima Kasih

- Microsoft untuk [Florence-2](https://huggingface.co/microsoft/Florence-2-base)
- Hugging Face untuk pustaka transformers
- Komunitas open-source

## 📞 Kontak

- **Issues**: [GitHub Issues](https://github.com/ardelyo/dara/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ardelyo/dara/discussions)

---

**Dibuat dengan ❤️ untuk aksesibilitas**
