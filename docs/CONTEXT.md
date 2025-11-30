# 🪔 DARA — Project Context & Vision

> **"Mata untuk semua" (Eyes for everyone)**

## Project Philosophy

DARA was born from a simple question: *What if AI could be the "eyes" for those who need them most?*

This project is not just about building another VLM. It's about creating technology that:
- **Empowers** visually impaired individuals to navigate the world independently
- **Runs anywhere** - from high-end servers to 3-year-old smartphones
- **Speaks the local language** - Indonesian and English support
- **Remains open** - fully transparent, auditable, and community-owned

## The Problem

Current VLMs face critical limitations for assistive technology:
1. **Too Large**: Models like GPT-4V require cloud APIs and fast internet
2. **Too Expensive**: Commercial APIs cost prohibitive for daily use
3. **Privacy Concerns**: Medical/financial data sent to external servers
4. **Language Barriers**: Poor support for non-English languages
5. **Context-Blind**: Generic descriptions without actionable advice

## DARA's Solution

### 1. Lightweight Architecture
- **DARA-Lite**: 0.23B parameters (Florence-2)
- **Fits in**: <500MB on disk, <2GB RAM
- **Runs on**: CPU, mobile SoCs, edge devices

### 2. "Smart Assist" Logic
Unlike generic VLMs that only describe, DARA provides **actionable guidance**:

```
❌ Generic VLM: "A bottle with text on it"
✅ DARA: "Paracetamol 500mg. Standard dose: 1-2 tablets after food. Do not exceed 8 tablets in 24h."
```

### 3. Multi-Modal Intelligence
The 5 modes cover daily life scenarios:
- **Scene**: Spatial awareness (obstacle detection, room layout)
- **Emotion**: Social interaction support
- **Medicine**: Health safety and compliance
- **Currency**: Financial independence
- **Text**: Information access (signs, menus, labels)

## Technical Innovation

### Task-Specific Prompting
Instead of one generic model, DARA uses **mode-aware prompting**:

```python
# Efficient mode switching without model reload
if mode == "scene":
    prompt = "<CAPTION>"  # Detailed description
elif mode == "text":
    prompt = "<OCR>"      # Precise text extraction
```

### LoRA Fine-Tuning
- Train only **0.1%** of parameters
- **10x faster** than full fine-tuning
- **1/10th memory** requirement

### Offline-First Design
- No internet required after download
- All processing on-device
- Privacy-preserving by default

## Roadmap

### Phase 1: Foundation (✅ Complete)
- [x] Core architecture with Florence-2
- [x] 5-mode implementation
- [x] Gradio web demo
- [x] Basic TTS integration

### Phase 2: Intelligence (🔄 In Progress)
- [ ] Collect domain-specific datasets
- [ ] Fine-tune on medicine labels (Indonesia)
- [ ] Train emotion recognition model
- [ ] Build currency classifier (IDR, USD, EUR)

### Phase 3: Optimization (📋 Planned)
- [ ] ONNX export for mobile
- [ ] INT8 quantization (<100ms latency)
- [ ] Android/iOS app wrappers
- [ ] Offline voice packs (Indonesian)

### Phase 4: Community (🎯 Future)
- [ ] Hugging Face release
- [ ] Public dataset contributions
- [ ] User studies with visually impaired community
- [ ] Multi-language expansion (Javanese, Sundanese, etc.)

## Impact Goals

### Accessibility Metrics
- **Target Users**: 250M+ visually impaired people worldwide
- **Device Reach**: Works on $100 smartphones
- **Latency**: <100ms for real-time assistance
- **Offline**: 100% functionality without internet

### Open Source Commitment
- **License**: Apache 2.0 / MIT
- **Transparency**: All training data documented
- **Community**: Issue tracker, discussions, contributions welcome
- **Education**: Tutorials for assistive tech developers

## Design Principles

### 1. Speed Over Perfection
Better to have 85% accuracy in 50ms than 95% in 2 seconds.
**Why**: Real-time feedback is critical for safety.

### 2. Actionable Over Descriptive
Don't just say "red pill", say "This is ibuprofen, reduce pain and fever."
**Why**: Users need guidance, not just information.

### 3. Local Over Cloud
Process on-device whenever possible.
**Why**: Privacy, reliability, cost.

### 4. Inclusive Design
Support Bahasa Indonesia as a first-class citizen.
**Why**: 270M+ speakers deserve accessible AI.

## User Stories

### Story 1: Medicina Safety
*Siti, 65, diabetic, visually impaired*
> "I use DARA to scan my medicine bottles every morning. It tells me which is insulin and which is metformin. It even reminds me to take them with food."

### Story 2: Currency Independence
*Budi, 40, entrepreneur*
> "When customers pay cash, I use DARA's currency mode to verify bills. It's faster than asking my assistant."

### Story 3: Social Confidence
*Rina, 28, office worker*
> "DARA's emotion mode helps me 'read the room' in meetings. It gives me confidence to speak up."

## Why Florence-2?

We chose Microsoft's Florence-2 as DARA-Lite's base for specific reasons:

1. **Native Multi-Task**: Built-in OCR, captioning, object detection
2. **Tiny Size**: 0.23B vs 7B+ for competitors
3. **Public Weights**: Fully open-sourced
4. **Proven Performance**: SOTA on lightweight VLM benchmarks
5. **Fine-Tunable**: LoRA support, efficient training

## Challenges & Limitations

### Current Limitations
- **Smart Assist**: Mostly rule-based (limited medical database)
- **Emotion Detection**: Relies on caption keywords, not trained classifier
- **Multi-Language**: TTS supports Indonesian, but model is English-dominant
- **Edge Cases**: Struggles with low-light, blurry images

### Active Research
- [ ] Vision-only emotion classifier (no text needed)
- [ ] Integrate with Indonesian medical drug database
- [ ] Low-light image enhancement pre-processor
- [ ] Multilingual training (Indonesian captions)

## Get Involved

### For Developers
- Contribute code improvements
- Add new modes (e.g., "Food" mode for nutrition)
- Optimize inference speed

### For Data Scientists
- Curate Indonesian-language datasets
- Improve Smart Assist logic
- Benchmark on edge devices

### For Accessibility Experts
- User testing with visually impaired community
- UX/UI feedback on voice prompts
- Feature requests for real-world needs

### For Donors/Sponsors
- Fund dataset annotation (medicine labels)
- Support app store deployment
- Sponsor cloud hosting for web demo

## Long-Term Vision

**DARA 2030**: Every smartphone ships with an on-device assistive AI.
- Universally accessible
- Privacy-preserving
- Culturally aware
- Community-owned

---

**Join us in building eyes for everyone. 🪔**

GitHub: [github.com/yourusername/dara]
Hugging Face: [huggingface.co/DARA]
Email: dara.project@example.com
