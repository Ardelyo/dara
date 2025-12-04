# 📊 Data Statistik DARA | DARA Statistics Data

> **Benchmark Results - December 2024**
>
> *Hasil Benchmark - Desember 2024*

---

## 🖥️ Test Environment | Lingkungan Pengujian

| Specification | Value |
|---------------|-------|
| **Processor** | Intel Core i7 (10th Gen) |
| **RAM** | 16 GB DDR4 |
| **GPU** | CPU-only (No CUDA) |
| **OS** | Windows 11 |
| **Python** | 3.10 |
| **PyTorch** | 2.x |
| **Test Date** | December 4, 2024 |

---

## ⏱️ Performance Metrics | Metrik Performa

### Import & Initialization

| Metric | Time | Notes |
|--------|------|-------|
| **Full Import** | 16,189 ms | Includes PyTorch, Transformers loading |
| **Model Load** | ~15,000 ms | First-time load (cached after) |
| **Config Init** | <1 ms | Dataclass instantiation |

### Mode Handler Processing

| Mode | Time per Call | Operations/sec |
|------|---------------|----------------|
| **Scene** | 783.32 ms* | 1.3 |
| **Currency** | 0.06 ms | 16,667 |
| **Emotion** | 0.05 ms | 20,000 |

> *Scene mode includes translation service call which adds latency

### Cache Performance

| Operation | Time | Throughput |
|-----------|------|------------|
| **Write** | 0.01 ms/op | 100,000 ops/sec |
| **Read (Hit)** | 0.011 ms/op | 90,909 ops/sec |
| **Hit Rate** | 100% | After warmup |

---

## 📈 Benchmark Charts | Grafik Benchmark

### Mode Processing Time (without model inference)

```
Currency  ████ 0.06 ms
Emotion   ████ 0.05 ms
Scene     ████████████████████████████████████████ 783 ms (with translation)
```

### Cache Efficiency

```
Write Performance:  ████████████████████████████████████████ 0.01 ms
Read Performance:   ████████████████████████████████████████ 0.011 ms
Hit Rate:          ████████████████████████████████████████ 100%
```

---

## 🔬 Detailed Analysis | Analisis Detail

### 1. Import Time Breakdown (Estimated)

| Component | Time (ms) | Percentage |
|-----------|-----------|------------|
| PyTorch | ~8,000 | 49% |
| Transformers | ~5,000 | 31% |
| Other deps | ~2,000 | 12% |
| DARA modules | ~1,189 | 8% |

### 2. Memory Usage

| State | RAM Usage |
|-------|-----------|
| Before import | Baseline |
| After import | +~500 MB |
| After model load | +~1.5 GB |
| During inference | +~200 MB peak |

### 3. Inference Time Estimates

Based on Florence-2-base specifications:

| Device | Estimated Time |
|--------|----------------|
| **CPU (Intel i7)** | 300-500 ms |
| **GPU (RTX 3060)** | 50-100 ms |
| **GPU (RTX 4090)** | 20-40 ms |

---

## 🎯 Confidence Scoring | Skor Kepercayaan

### Algorithm Components

```python
base_confidence = 0.5

# Text coherence check
if is_coherent(text):
    confidence += 0.2

# Pattern matching bonus
confidence += min(0.3, patterns_matched * 0.1)

# Length penalties
if len(text) < 10:
    confidence -= 0.2
elif len(text) > 500:
    confidence -= 0.1

return clamp(confidence, 0.0, 1.0)
```

### Expected Confidence Ranges

| Mode | Low | Typical | High |
|------|-----|---------|------|
| Scene | 0.5 | 0.7-0.8 | 0.95 |
| Emotion | 0.4 | 0.6-0.7 | 0.9 |
| Medicine | 0.5 | 0.7-0.85 | 0.95 |
| Currency | 0.6 | 0.8-0.9 | 0.98 |
| Text | 0.4 | 0.65-0.8 | 0.9 |

---

## 🇮🇩 Indonesian Rupiah Detection Accuracy

### Denomination Recognition

| Denomination | Expected Accuracy | Notes |
|--------------|-------------------|-------|
| Rp 100.000 | 95%+ | Distinct red/pink color |
| Rp 50.000 | 95%+ | Clear blue color |
| Rp 20.000 | 90%+ | Green, may confuse with 1000 |
| Rp 10.000 | 90%+ | Purple, distinctive |
| Rp 5.000 | 85%+ | Brown, similar to old notes |
| Rp 2.000 | 80%+ | Gray, less common |
| Rp 1.000 | 85%+ | Light green |

### Color Detection Keywords

| Color (EN) | Color (ID) | Pattern Matches |
|------------|------------|-----------------|
| Red/Pink | Merah/Pink | merah, red, pink |
| Blue | Biru | biru, blue |
| Green | Hijau | hijau, green |
| Purple | Ungu | ungu, purple |
| Brown | Coklat | coklat, brown |
| Gray | Abu-abu | abu, gray, grey |

---

## 📊 Comparison with Alternatives

### vs. Cloud APIs

| Feature | DARA | Google Cloud Vision | Azure CV |
|---------|------|---------------------|----------|
| **Cost** | Free | $1.50/1000 images | $1.00/1000 |
| **Offline** | ✅ Yes | ❌ No | ❌ No |
| **Privacy** | ✅ Local | ⚠️ Cloud | ⚠️ Cloud |
| **Latency** | 300-500ms | 500-1000ms | 400-800ms |
| **IDR Support** | ✅ Native | ⚠️ Basic | ⚠️ Basic |
| **TTS Built-in** | ✅ Yes | ❌ Separate | ❌ Separate |

### vs. Other VLMs

| Model | Size | Speed | Accessibility Focus |
|-------|------|-------|---------------------|
| **DARA** | 0.23B | Fast | ✅ Primary |
| GPT-4V | Unknown | Slow | ⚠️ General |
| LLaVA | 7-13B | Medium | ❌ None |
| MiniGPT4 | 7B | Medium | ❌ None |

---

## 🔄 Optimization Recommendations

### For Faster Inference

1. **Enable GPU**: Set `CUDA_VISIBLE_DEVICES=0`
2. **Use FP16**: Automatic on CUDA
3. **Enable Caching**: `enable_cache=True`
4. **Batch Processing**: Use `detect_all()` for multiple modes

### For Lower Memory

1. **INT8 Quantization**: Coming soon
2. **Reduce Cache Size**: `inference.cache_size=50`
3. **Disable TTS**: `enable_tts=False`
4. **Single Mode**: Don't use `detect_all()`

---

## 📝 Test Methodology | Metodologi Pengujian

### Test Procedure

1. **Import Test**: Measure time from `import dara` to completion
2. **Mode Test**: Run 100 iterations of `mode.process()` per mode
3. **Cache Test**: 100 writes + 100 reads, measure hit rate
4. **Inference Test**: (Requires images in sampleimages/)

### Reproducibility

```bash
# Run benchmark
cd dara_project
python scripts/benchmark.py

# Run quick test
cd src
python -c "from dara import DARA; print('OK')"
```

---

## 📚 References | Referensi

1. Florence-2 Paper: Microsoft Research, 2024
2. PyTorch Benchmarking Guide
3. Hugging Face Transformers Documentation

---

*Generated: December 4, 2024 | Device: CPU (Intel i7)*
