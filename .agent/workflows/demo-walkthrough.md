---
description: Panduan lengkap mencoba demo DARA
---

# 🪔 DARA Demo - Walkthrough Lengkap

> **Panduan step-by-step untuk mencoba demonstrasi DARA (Detect & Assist Recognition AI)**

## 📋 Prasyarat

Sebelum memulai, pastikan Anda telah menyiapkan:

- ✅ Python 3.8 atau lebih tinggi terinstal
- ✅ Git terinstal (untuk clone repository)
- ✅ Koneksi internet (untuk download dependencies)
- ✅ Webcam (opsional, untuk fitur kamera)

---

## 🚀 Langkah 1: Setup Environment

### 1.1 Clone Repository (Jika Belum)

```bash
git clone https://github.com/ardelyo/dara.git
cd dara/dara_project
```

### 1.2 Install Dependencies

// turbo
```bash
pip install -r requirements.txt
```

**Catatan**: Proses instalasi mungkin memakan waktu 3-5 menit tergantung koneksi internet Anda.

### 1.3 Verifikasi Instalasi

// turbo
```bash
python -c "from dara import DARA; print('✓ DARA berhasil terinstal!')"
```

---

## 🎨 Langkah 2: Persiapan Sample Images

Pastikan folder `sampleimages/` berisi contoh gambar untuk testing:

// turbo
```bash
ls sampleimages/
```

Anda akan melihat:
- `food table.jpg` - Untuk mode Scene
- `sad person.jpg` - Untuk mode Emotion
- `medic.jpg` - Untuk mode Medicine
- `park signs.jpg` - Untuk mode Text

---

## 🖥️ Langkah 3: Menjalankan Demo

### 3.1 Launch Gradio Interface

// turbo
```bash
python app.py
```

atau menggunakan file demo:

```bash
python demo/app.py
```

**Output yang diharapkan**:
```
Initializing DARA...
DARA Initialized Successfully!
Running on local URL:  http://127.0.0.1:7860
```

### 3.2 Akses Web Interface

1. Buka browser Anda
2. Navigate ke: `http://localhost:7860`
3. Anda akan melihat interface DARA yang elegan!

---

## 🧪 Langkah 4: Mencoba 5 Mode DARA

### Mode 1: 👁️ Scene Detection

**Tujuan**: Mendeskripsikan lingkungan sekitar

1. Upload gambar `sampleimages/food table.jpg`
2. Pilih mode: **Scene**
3. Pilih bahasa: **English** atau **Indonesian (Bahasa Indonesia)**
4. Klik tombol **👁️ Detect & Assist**

**Output yang diharapkan**:
- **Text**: Deskripsi detail tentang meja makan dan makanan di atasnya
- **Audio**: Narasi suara otomatis memutar hasil deteksi

---

### Mode 2: 😊 Emotion Recognition

**Tujuan**: Membaca ekspresi wajah

1. Upload gambar `sampleimages/sad person.jpg`
2. Pilih mode: **Emotion**
3. Pilih bahasa: **Indonesian (Bahasa Indonesia)**
4. Klik **👁️ Detect & Assist**

**Output yang diharapkan**:
- Deteksi emosi (sedih, senang, marah, dll.)
- Konteks tambahan tentang bahasa tubuh

---

### Mode 3: 💊 Medicine Label Reader

**Tujuan**: Membaca label obat-obatan

1. Upload gambar `sampleimages/medic.jpg`
2. Pilih mode: **Medicine**
3. Pilih bahasa: **English**
4. Klik **👁️ Detect & Assist**

**Output yang diharapkan**:
- Nama obat
- Dosis
- Instruksi penggunaan
- ⚠️ **Disclaimer**: Verifikasi dengan profesional medis

---

### Mode 4: 💵 Currency Detection

**Tujuan**: Identifikasi uang kertas/koin

1. Upload gambar uang (bisa Indonesian Rupiah atau currency lainnya)
2. Pilih mode: **Currency**
3. Pilih bahasa: **Indonesian (Bahasa Indonesia)**
4. Klik **👁️ Detect & Assist**

**Output yang diharapkan**:
- Nominal uang
- Warna/ciri khas
- Keaslian (jika terlatih)

---

### Mode 5: 📖 Text Recognition (OCR)

**Tujuan**: Membaca teks di sekitar

1. Upload gambar `sampleimages/park signs.jpg`
2. Pilih mode: **Text**
3. Pilih bahasa: **English**
4. Klik **👁️ Detect & Assist**

**Output yang diharapkan**:
- Semua teks yang terdeteksi
- Konteks lokasi teks dalam gambar

---

## 📸 Langkah 5: Mencoba dengan Webcam

1. Klik tab **webcam** di bagian input gambar
2. Izinkan akses kamera saat browser meminta
3. Ambil foto menggunakan webcam
4. Pilih mode yang diinginkan
5. Klik **👁️ Detect & Assist**

**Use Case Real-World**:
- Arahkan kamera ke menu restoran → Mode: Text
- Arahkan ke orang → Mode: Emotion
- Arahkan ke botol obat → Mode: Medicine

---

## 🔄 Langkah 6: Mencoba Examples

DARA menyediakan contoh pre-loaded untuk testing cepat:

1. Scroll ke bagian **📸 Try with Examples**
2. Klik salah satu contoh
3. Gambar, mode, dan bahasa akan otomatis terisi
4. Hasil akan langsung ditampilkan

---

## 🎯 Tips & Best Practices

### Untuk Hasil Terbaik:
- ✅ **Pencahayaan**: Gunakan gambar dengan pencahayaan baik
- ✅ **Fokus**: Pastikan objek utama jelas dan tidak blur
- ✅ **Jarak**: Tidak terlalu dekat atau jauh
- ✅ **Resolusi**: Gunakan gambar dengan resolusi minimal 640x480

### Troubleshooting:

**Problem**: Model tidak initialize
```bash
# Solusi: Re-install dependencies
pip install --upgrade transformers torch pillow
```

**Problem**: Audio tidak keluar
- Pastikan speaker/headphone aktif
- Cek browser audio permissions

**Problem**: Proses lambat
- DARA-Lite dirancang untuk CPU, tunggu ~1-3 detik
- Tutup aplikasi lain untuk free up RAM

---

## 🧹 Langkah 7: Cleanup

Setelah selesai testing:

1. Tekan `Ctrl + C` di terminal untuk stop server
2. File temporary `temp_input.jpg` akan otomatis terhapus
3. Audio output disimpan sementara di `output.mp3`

---

## 🚀 Next Steps

Setelah mencoba demo, Anda bisa:

1. **Integrasi ke Aplikasi**: Gunakan DARA API dalam project Anda
2. **Training Custom**: Fine-tune dengan dataset sendiri
3. **Deploy to HuggingFace**: Share model Anda ke komunitas
4. **Mobile Deployment**: Convert to TFLite/ONNX untuk mobile

Lihat dokumentasi lengkap di:
- [Architecture Guide](../docs/ARCHITECTURE.md)
- [API Reference](../docs/API.md)
- [Training Guide](../docs/TRAINING.md)

---

## 📞 Bantuan

Jika mengalami masalah:
- 📖 Baca [Troubleshooting Guide](../docs/TROUBLESHOOTING.md)
- 💬 Diskusi di [GitHub Discussions](https://github.com/ardelyo/dara/discussions)
- 🐛 Laporkan bug di [GitHub Issues](https://github.com/ardelyo/dara/issues)

---

**Selamat Mencoba! 🎉**

*"DARA - Mata untuk semua"*
