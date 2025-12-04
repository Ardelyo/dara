# 🚀 Panduan Cepat DARA | DARA Quick Start Guide

[🇮🇩 Bahasa Indonesia](#bahasa-indonesia) | [🇺🇸 English](#english)

---

## Bahasa Indonesia

### Instalasi dalam 3 Langkah

```bash
# 1. Clone repository
git clone https://github.com/ardelyo/dara.git
cd dara/dara_project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install package
pip install -e .
```

### Penggunaan Pertama

```python
from dara import DARA

# Inisialisasi (muat model ~15 detik pertama kali)
dara = DARA()

# Deteksi scene
result = dara.detect("foto.jpg", mode="scene", language="id")
print(result["result"])
# Output: "Dapur modern dengan meja dan kompor yang menyala"
```

### 5 Mode yang Tersedia

| Mode | Kegunaan | Contoh Perintah |
|------|----------|-----------------|
| 🏞️ `scene` | Deskripsi lingkungan | `dara.detect(img, mode="scene")` |
| 😊 `emotion` | Baca ekspresi wajah | `dara.detect(img, mode="emotion")` |
| 💊 `medicine` | Baca label obat | `dara.detect(img, mode="medicine")` |
| 💵 `currency` | Identifikasi uang | `dara.detect(img, mode="currency")` |
| 📝 `text` | Baca teks apapun | `dara.detect(img, mode="text")` |

### Demo Web (Gradio)

```bash
python app.py
# Buka http://localhost:7860
```

### Tips Performa

- ✅ Gunakan `language="id"` untuk output Bahasa Indonesia
- ✅ Set `generate_audio=False` untuk inferensi lebih cepat
- ✅ Aktifkan GPU dengan set `CUDA_VISIBLE_DEVICES=0`
- ✅ Gunakan cache (default aktif) untuk query berulang

---

## English

### Installation in 3 Steps

```bash
# 1. Clone repository
git clone https://github.com/ardelyo/dara.git
cd dara/dara_project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install package
pip install -e .
```

### First Usage

```python
from dara import DARA

# Initialize (loads model ~15s first time)
dara = DARA()

# Detect scene
result = dara.detect("photo.jpg", mode="scene", language="en")
print(result["result"])
# Output: "Modern kitchen with table and stove that is on"
```

### 5 Available Modes

| Mode | Purpose | Example Command |
|------|---------|-----------------|
| 🏞️ `scene` | Environment description | `dara.detect(img, mode="scene")` |
| 😊 `emotion` | Read facial expressions | `dara.detect(img, mode="emotion")` |
| 💊 `medicine` | Read medicine labels | `dara.detect(img, mode="medicine")` |
| 💵 `currency` | Identify currency | `dara.detect(img, mode="currency")` |
| 📝 `text` | Read any text | `dara.detect(img, mode="text")` |

### Web Demo (Gradio)

```bash
python app.py
# Open http://localhost:7860
```

### Performance Tips

- ✅ Use `language="id"` for Indonesian output
- ✅ Set `generate_audio=False` for faster inference
- ✅ Enable GPU by setting `CUDA_VISIBLE_DEVICES=0`
- ✅ Use cache (enabled by default) for repeated queries

---

## 📸 Contoh Lengkap | Full Example

```python
from dara import DARA

# Inisialisasi
dara = DARA()

# Deteksi mata uang (Currency detection)
result = dara.detect(
    "uang_50rb.jpg",
    mode="currency",
    language="id",
    generate_audio=True
)

print("Mode:", result["mode"])           # currency
print("Hasil:", result["result"])        # Terdeteksi: Rp 50.000 (warna biru)
print("Kepercayaan:", result["confidence"])  # 0.85
print("Audio:", result["audio"])         # output_abc123.mp3
print("Saran:", result["suggestions"])   # ["Periksa ciri keamanan uang"]

# Metadata tambahan
print("Total IDR:", result["metadata"]["total_idr"])  # 50000
```

---

## 🔗 Link Penting | Important Links

- 📖 [API Reference](API.md)
- 🏗️ [Architecture](ARCHITECTURE.md)
- 📊 [Statistics](STATISTICS.md)
- 🔬 [Research Paper](RESEARCH.md)
- 📚 [Training Guide](TRAINING.md)

---

*"Mata untuk semua" | "Eyes for everyone"*
