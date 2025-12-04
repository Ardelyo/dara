# 📚 Referensi API DARA | DARA API Reference

[🇮🇩 Bahasa Indonesia](#bahasa-indonesia) | [🇺🇸 English](#english)

---

## Bahasa Indonesia

### Instalasi Cepat

```bash
# Clone repository
git clone https://github.com/ardelyo/dara.git
cd dara/dara_project

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Penggunaan Dasar

```python
from dara import DARA

# Inisialisasi
dara = DARA()

# Deteksi dengan mode
result = dara.detect("foto.jpg", mode="scene", language="id")

print(result["result"])     # Teks output
print(result["confidence"]) # Skor kepercayaan
print(result["audio"])      # Path ke file audio
```

---

### Kelas Utama

#### `DARA`

Kelas utama untuk memuat dan menjalankan inferensi.

##### Konstruktor

```python
DARA(
    model_id: str = None,        # ID model Hugging Face
    config: Config = None,       # Objek konfigurasi
    enable_tts: bool = True,     # Aktifkan text-to-speech
    enable_cache: bool = True,   # Aktifkan cache inferensi
    log_level: str = "INFO"      # Level logging
)
```

**Contoh:**
```python
from dara import DARA, Config

# Konfigurasi default
dara = DARA()

# Konfigurasi kustom
config = Config.from_env()
dara = DARA(config=config)
```

##### Method `detect()`

```python
detect(
    image_input,           # Path gambar atau PIL Image
    mode: str = "scene",   # Mode deteksi
    language: str = "en",  # Bahasa output ("en" atau "id")
    generate_audio: bool = True  # Generate audio TTS
) -> dict
```

**Parameter:**
| Parameter | Tipe | Deskripsi |
|-----------|------|-----------|
| `image_input` | `str`, `Path`, `PIL.Image` | Input gambar |
| `mode` | `str` | Mode deteksi (lihat tabel di bawah) |
| `language` | `str` | `"en"` (Inggris) atau `"id"` (Indonesia) |
| `generate_audio` | `bool` | Apakah generate file audio |

**Mode yang Tersedia:**
| Mode | Deskripsi | Contoh Output |
|------|-----------|---------------|
| `"scene"` | Deskripsi lingkungan | "Dapur modern dengan meja dan kursi" |
| `"emotion"` | Analisis emosi | "Senang. Mereka terlihat bersemangat!" |
| `"medicine"` | Baca label obat | "Dosis: 500mg. Minum setelah makan" |
| `"currency"` | Deteksi mata uang | "Terdeteksi: Rp 50.000 (warna biru)" |
| `"text"` | OCR teks umum | "Pintu keluar di sebelah kiri" |

**Return Value:**
```python
{
    "mode": "currency",           # Mode yang dipilih
    "result": "Terdeteksi: Rp 50.000 (warna biru)",
    "confidence": 0.85,           # Skor kepercayaan (0.0-1.0)
    "audio": "output_abc123.mp3", # Path audio (jika diaktifkan)
    "language": "id",             # Bahasa output
    "metadata": {                 # Data tambahan per mode
        "idr_detected": [...],
        "total_idr": 50000
    },
    "suggestions": [              # Saran untuk pengguna
        "Periksa ciri keamanan uang"
    ]
}
```

**Contoh Penggunaan:**
```python
# Deteksi scene dalam Bahasa Indonesia
result = dara.detect("foto_dapur.jpg", mode="scene", language="id")
print(result["result"])
# Output: "Dapur dengan meja dan kompor yang menyala"

# Deteksi mata uang
result = dara.detect("uang.jpg", mode="currency", language="id")
print(f"Nilai: {result['result']}")
print(f"Total: Rp {result['metadata']['total_idr']:,}")

# Tanpa audio (lebih cepat)
result = dara.detect("teks.jpg", mode="text", generate_audio=False)
```

##### Method `detect_all()`

Jalankan semua mode sekaligus:

```python
results = dara.detect_all("foto.jpg", language="id")

for mode, result in results.items():
    print(f"{mode}: {result['result']}")
```

##### Method `get_available_modes()`

```python
modes = dara.get_available_modes()
# Output: ['scene', 'emotion', 'medicine', 'currency', 'text']
```

##### Method `clear_cache()`

```python
cleared = dara.clear_cache()
print(f"Membersihkan {cleared} entry cache")
```

##### Property `cache_stats`

```python
stats = dara.cache_stats
# Output: {'hits': 10, 'misses': 3, 'hit_rate': 0.769, 'size': 13}
```

---

### Konfigurasi

#### `Config`

```python
from dara import Config

# Konfigurasi default
config = Config()

# Dari environment variables
config = Config.from_env()

# Akses setting
print(config.device)           # "cuda" atau "cpu"
print(config.model.model_id)   # "microsoft/Florence-2-base"
print(config.inference.cache_size)  # 100
```

**Environment Variables:**
| Variable | Deskripsi | Default |
|----------|-----------|---------|
| `DARA_MODEL_ID` | ID model Hugging Face | `microsoft/Florence-2-base` |
| `DARA_ENABLE_CACHE` | Aktifkan cache | `true` |
| `DARA_CACHE_SIZE` | Ukuran cache | `100` |
| `DARA_QUANTIZATION` | Mode quantization | `none` |
| `DARA_TTS_ENGINE` | Engine TTS | `pyttsx3` |
| `DARA_TTS_RATE` | Kecepatan suara | `150` |

---

### Mode Handlers

Akses langsung ke handler mode:

```python
from dara.modes import CurrencyMode, MedicineMode

# Gunakan handler secara langsung
currency = CurrencyMode()
result = currency.process("Rp 100.000", language="id")

print(result.text)       # "Terdeteksi: Rp 100.000 (warna merah)"
print(result.confidence) # 0.9
print(result.metadata)   # {"idr_detected": [...]}
```

---

### Services

#### TTSService

```python
from dara.services import TTSService

tts = TTSService(rate=150, enable_cache=True)
audio_path = tts.generate("Halo dunia", language="id")
```

#### TranslationService

```python
from dara.services import TranslationService

translator = TranslationService()
text_id = translator.to_indonesian("Hello world")
# Output: "Halo dunia"
```

#### InferenceCache

```python
from dara.services import InferenceCache

cache = InferenceCache(maxsize=100)
cache.set("hash123", "prompt", "result")
result = cache.get("hash123", "prompt")
```

---

### Penanganan Error

| Error | Penyebab | Solusi |
|-------|----------|--------|
| `FileNotFoundError` | File gambar tidak ditemukan | Periksa path file |
| `ValueError` | Mode tidak valid | Gunakan mode yang tersedia |
| `RuntimeError` | CUDA out of memory | Set `CUDA_VISIBLE_DEVICES=""` |

```python
try:
    result = dara.detect("foto.jpg", mode="scene")
except FileNotFoundError:
    print("Gambar tidak ditemukan!")
except ValueError as e:
    print(f"Error: {e}")
```

---

## English

### Quick Installation

```bash
# Clone repository
git clone https://github.com/ardelyo/dara.git
cd dara/dara_project

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Basic Usage

```python
from dara import DARA

# Initialize
dara = DARA()

# Detect with mode
result = dara.detect("photo.jpg", mode="scene", language="en")

print(result["result"])     # Text output
print(result["confidence"]) # Confidence score
print(result["audio"])      # Path to audio file
```

---

### Main Classes

#### `DARA`

Main class for loading and running inference.

##### Constructor

```python
DARA(
    model_id: str = None,        # Hugging Face model ID
    config: Config = None,       # Configuration object
    enable_tts: bool = True,     # Enable text-to-speech
    enable_cache: bool = True,   # Enable inference caching
    log_level: str = "INFO"      # Logging level
)
```

**Example:**
```python
from dara import DARA, Config

# Default configuration
dara = DARA()

# Custom configuration
config = Config.from_env()
dara = DARA(config=config)
```

##### Method `detect()`

```python
detect(
    image_input,           # Image path or PIL Image
    mode: str = "scene",   # Detection mode
    language: str = "en",  # Output language ("en" or "id")
    generate_audio: bool = True  # Generate TTS audio
) -> dict
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `image_input` | `str`, `Path`, `PIL.Image` | Image input |
| `mode` | `str` | Detection mode (see table below) |
| `language` | `str` | `"en"` (English) or `"id"` (Indonesian) |
| `generate_audio` | `bool` | Whether to generate audio file |

**Available Modes:**
| Mode | Description | Example Output |
|------|-------------|----------------|
| `"scene"` | Environment description | "Modern kitchen with table and chairs" |
| `"emotion"` | Emotion analysis | "Happy. They seem in good spirits!" |
| `"medicine"` | Read medicine labels | "Dosage: 500mg. Take after meals" |
| `"currency"` | Currency detection | "Detected: Rp 50,000 (blue color)" |
| `"text"` | General OCR | "Exit door on the left" |

**Return Value:**
```python
{
    "mode": "currency",           # Selected mode
    "result": "Detected: Rp 50,000 (blue color)",
    "confidence": 0.85,           # Confidence score (0.0-1.0)
    "audio": "output_abc123.mp3", # Audio path (if enabled)
    "language": "en",             # Output language
    "metadata": {                 # Mode-specific data
        "idr_detected": [...],
        "total_idr": 50000
    },
    "suggestions": [              # User suggestions
        "Verify security features"
    ]
}
```

**Usage Examples:**
```python
# Scene detection in English
result = dara.detect("kitchen.jpg", mode="scene", language="en")
print(result["result"])
# Output: "Kitchen with table and stove that is on"

# Currency detection
result = dara.detect("money.jpg", mode="currency", language="en")
print(f"Value: {result['result']}")
print(f"Total: Rp {result['metadata']['total_idr']:,}")

# Without audio (faster)
result = dara.detect("text.jpg", mode="text", generate_audio=False)
```

##### Method `detect_all()`

Run all modes at once:

```python
results = dara.detect_all("photo.jpg", language="en")

for mode, result in results.items():
    print(f"{mode}: {result['result']}")
```

##### Method `get_available_modes()`

```python
modes = dara.get_available_modes()
# Output: ['scene', 'emotion', 'medicine', 'currency', 'text']
```

##### Method `clear_cache()`

```python
cleared = dara.clear_cache()
print(f"Cleared {cleared} cache entries")
```

##### Property `cache_stats`

```python
stats = dara.cache_stats
# Output: {'hits': 10, 'misses': 3, 'hit_rate': 0.769, 'size': 13}
```

---

### Configuration

#### `Config`

```python
from dara import Config

# Default configuration
config = Config()

# From environment variables
config = Config.from_env()

# Access settings
print(config.device)           # "cuda" or "cpu"
print(config.model.model_id)   # "microsoft/Florence-2-base"
print(config.inference.cache_size)  # 100
```

**Environment Variables:**
| Variable | Description | Default |
|----------|-------------|---------|
| `DARA_MODEL_ID` | Hugging Face model ID | `microsoft/Florence-2-base` |
| `DARA_ENABLE_CACHE` | Enable caching | `true` |
| `DARA_CACHE_SIZE` | Cache size | `100` |
| `DARA_QUANTIZATION` | Quantization mode | `none` |
| `DARA_TTS_ENGINE` | TTS engine | `pyttsx3` |
| `DARA_TTS_RATE` | Speech rate | `150` |

---

### Mode Handlers

Direct access to mode handlers:

```python
from dara.modes import CurrencyMode, MedicineMode

# Use handler directly
currency = CurrencyMode()
result = currency.process("Rp 100,000", language="en")

print(result.text)       # "Detected: Rp 100,000 (red color)"
print(result.confidence) # 0.9
print(result.metadata)   # {"idr_detected": [...]}
```

---

### Services

#### TTSService

```python
from dara.services import TTSService

tts = TTSService(rate=150, enable_cache=True)
audio_path = tts.generate("Hello world", language="en")
```

#### TranslationService

```python
from dara.services import TranslationService

translator = TranslationService()
text_id = translator.to_indonesian("Hello world")
# Output: "Halo dunia"
```

#### InferenceCache

```python
from dara.services import InferenceCache

cache = InferenceCache(maxsize=100)
cache.set("hash123", "prompt", "result")
result = cache.get("hash123", "prompt")
```

---

### Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError` | Image file not found | Check file path |
| `ValueError` | Invalid mode | Use available modes |
| `RuntimeError` | CUDA out of memory | Set `CUDA_VISIBLE_DEVICES=""` |

```python
try:
    result = dara.detect("photo.jpg", mode="scene")
except FileNotFoundError:
    print("Image not found!")
except ValueError as e:
    print(f"Error: {e}")
```

---

## 🔧 CLI Usage | Penggunaan CLI

```python
# cli.py
import argparse
from dara import DARA

def main():
    parser = argparse.ArgumentParser(description="DARA CLI")
    parser.add_argument("--image", "-i", required=True, help="Path to image")
    parser.add_argument("--mode", "-m", default="scene", 
                       choices=["scene", "emotion", "medicine", "currency", "text"])
    parser.add_argument("--language", "-l", default="en", choices=["en", "id"])
    args = parser.parse_args()
    
    dara = DARA()
    result = dara.detect(args.image, mode=args.mode, language=args.language)
    
    print(f"Mode: {result['mode']}")
    print(f"Result: {result['result']}")
    print(f"Confidence: {result['confidence']:.2f}")

if __name__ == "__main__":
    main()
```

**Usage / Penggunaan:**
```bash
python cli.py --image photo.jpg --mode scene --language id
```
