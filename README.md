# FDN Reverb

High-performance Feedback Delay Network (FDN) reverb with traditional DSP and learnable DDSP versions.

[![Performance](https://img.shields.io/badge/performance-81x%20realtime-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.8+-blue)]()
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange)]()

Process 2 minutes of stereo audio in ~2.4 seconds on M4 MacBook Pro!

---

## Table of Contents

- [What is This?](#what-is-this)
- [Quick Start (30 seconds)](#quick-start-30-seconds)
- [Learning Path](#learning-path)
- [Installation](#installation)
- [Repository Overview](#repository-overview)
- [Documentation](#documentation)
- [Citation](#citation)

---

## What is This?

This repository provides **two complementary approaches** to FDN reverb:

| Approach | Best For | Key Feature |
|----------|----------|-------------|
| **Traditional FDN** | Production, real-time processing | 81x realtime (C++ optimized) |
| **DDSP FDN** | Research, learning from data | Learnable parameters via gradient descent |

**Choose Traditional** when you know your reverb parameters and need speed.

**Choose DDSP** when you want to:
- Learn reverb from reference recordings
- Auto-optimize parameters from target audio
- Integrate with neural networks
- Clone hardware/spaces automatically

---

## Quick Start (30 seconds)

### 1. Install

```bash
# Clone repository
git clone https://github.com/yourusername/FDN-Reverb.git
cd FDN-Reverb

# Install dependencies
pip install -r requirements.txt
```

### 2. Process Audio (Traditional)

```bash
# Apply reverb with default settings
python main.py -i dry.mp3 -o wet.mp3
```

### 3. Try DDSP Demo

```bash
# See DDSP learning in action
python demo_ddsp.py
```

**Done!** Continue to [Learning Path](#learning-path) for next steps.

---

## Learning Path

Follow this path from beginner to advanced:

### Level 1: Basic Usage (5 minutes)

**Goal**: Apply reverb to audio files

```bash
# Process audio with different reverb settings
python main.py -i song.mp3 -o song_reverb.mp3 --feedback_gain 0.95 --wet 0.8

# Try different presets
python main.py -i song.mp3 -o short_reverb.mp3 --feedback_gain 0.7 --wet 0.3  # Short
python main.py -i song.mp3 -o long_reverb.mp3 --feedback_gain 0.95 --wet 0.9   # Long
```

**Learn more**: [Basic Usage Guide](#basic-usage)

---

### Level 2: Understanding Parameters (10 minutes)

**Goal**: Understand what each parameter does

| Parameter | Range | Effect | Example |
|-----------|-------|--------|---------|
| `feedback_gain` | 0-1 | Reverb tail length | 0.7=short, 0.95=long |
| `damp` | 0-1 | High-freq decay | 0.2=bright, 0.5=dark |
| `wet` | 0-1 | Dry/wet mix | 0.3=subtle, 0.9=obvious |
| `delays_ms` | ms values | Delay line lengths | Use prime numbers |

**Try this**:
```bash
# Bright, short reverb (vocals)
python main.py -i vocals.mp3 -o vocals_reverb.mp3 --feedback_gain 0.75 --damp 0.2 --wet 0.4

# Dark, long reverb (ambient)
python main.py -i pad.mp3 -o pad_reverb.mp3 --feedback_gain 0.92 --damp 0.5 --wet 0.8
```

**Learn more**: See [Parameter Guide](README.md#parameters) for details

---

### Level 3: DDSP Basics (15 minutes)

**Goal**: Learn reverb parameters from target audio

```bash
# Demo 1: See DDSP learn and apply reverb
python demo_ddsp.py

# Demo 2: Train reverb to match a target
python train_ddsp.py --dry input.mp3 --target reference_reverb.mp3 --epochs 200
```

**What happens**: DDSP automatically discovers the parameters that transform `dry` → `target`

**Learn more**: [README_DDSP.md](README_DDSP.md)

---

### Level 4: Advanced Applications (30 minutes)

**Goal**: Use DDSP for intelligent audio processing

#### Application 1: Neural Reverb Controller
Neural network predicts optimal reverb from audio content.

```bash
python app_neural_reverb.py
```

**Result**: Network learns "high-pitched → short reverb", "low-pitched → long reverb"

#### Application 2: Style Transfer
Clone reverb from reference recordings.

```bash
# Learn reverb from a famous studio/hardware
python app_style_transfer.py --dry studio_dry.mp3 --wet studio_wet.mp3 --new my_audio.mp3
```

**Result**: Your audio gets the same reverb character as the reference

#### Application 3: Adaptive Reverb
Context-aware reverb that adapts to audio dynamics.

```bash
# Automatic adaptation: quiet → more reverb, loud → less reverb
python app_adaptive_reverb.py --input song.mp3 --output song_adaptive.mp3
```

**Learn more**: [README_APPLICATIONS.md](README_APPLICATIONS.md)

---

### Level 5: Benchmarking & Comparison (20 minutes)

**Goal**: Understand performance trade-offs

```bash
# Full comparison: performance, quality, visualizations
python run_comparison.py --input dry.mp3 --full

# Results saved to comparison_results/
```

**What you get**:
- Performance benchmarks (Traditional: 81x realtime, DDSP: 21x realtime)
- Audio quality metrics (RT60, spectral analysis)
- Visualizations (waveforms, spectrograms, frequency response)

**Learn more**: [README_COMPARISON.md](README_COMPARISON.md)

---

### Level 6: Research & Theory (30+ minutes)

**Goal**: Deep understanding of algorithms

Read these in order:
1. [Blog: FDN Reverb Theory](https://puar-playground.github.io/posts/reverb/) - Mathematical foundations
2. [COMPARISON.md](COMPARISON.md) - vs dafx25-ddsp-tutorial repository
3. [Paper: Jot & Chaigne 1991](#citation) - Original FDN paper

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- FFmpeg (for MP3 support)

### Step-by-Step

```bash
# 1. Clone repository
git clone https://github.com/yourusername/FDN-Reverb.git
cd FDN-Reverb

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install FFmpeg (for MP3 support)
# macOS:
brew install ffmpeg

# Linux:
sudo apt-get install ffmpeg

# Windows:
# Download from https://ffmpeg.org/
```

### Verify Installation

```bash
python main.py -i dry.mp3 -o test.mp3
# Should process in ~2-3 seconds
```

---

## Repository Overview

### Core Files

**Traditional FDN** (fast, production-ready):
- `main.py` - CLI for audio processing
- `reverb_util.py` - FDN implementation
- `fdn_cpu.cpp` - C++ acceleration (auto-compiled)
- `io_utils.py` - Audio I/O utilities

**DDSP FDN** (learnable, research-oriented):
- `ddsp_reverb.py` - Differentiable FDN
- `train_ddsp.py` - Training script
- `demo_ddsp.py` - Interactive demos

**Applications** (advanced DDSP demos):
- `app_neural_reverb.py` - Neural network controller
- `app_style_transfer.py` - Reverb cloning
- `app_adaptive_reverb.py` - Context-aware reverb

**Comparison Tools**:
- `run_comparison.py` - Complete benchmark suite
- `compare_performance.py` - Speed benchmarks
- `compare_quality.py` - Audio metrics
- `visualize_comparison.py` - Plots & visualizations

### Documentation

| Document | Purpose | Read When |
|----------|---------|-----------|
| [README.md](README.md) | This file - overview & learning path | Start here |
| [README_DDSP.md](README_DDSP.md) | DDSP implementation guide | Level 3+ |
| [README_APPLICATIONS.md](README_APPLICATIONS.md) | Application demos | Level 4+ |
| [README_COMPARISON.md](README_COMPARISON.md) | Benchmarking guide | Level 5+ |
| [COMPARISON.md](COMPARISON.md) | Theory vs dafx25-ddsp-tutorial | Level 6+ |

---

## Basic Usage

### Command Line

```bash
# Simple reverb
python main.py -i input.mp3 -o output.mp3

# Custom parameters
python main.py -i input.mp3 -o output.mp3 \
    --feedback_gain 0.85 \
    --damp 0.3 \
    --wet 0.7

# All options
python main.py --help
```

### Python API

```python
from reverb_util import FDNReverb
from io_utils import load_audio, save_audio

# Load audio
audio, sr = load_audio("input.mp3")

# Create reverb
reverb = FDNReverb(
    sr=sr,
    delays_ms=(29, 37, 43, 53, 61, 71, 79, 89),
    feedback_gain=0.9,
    damp=0.25,
    wet=0.9
)

# Process
output = reverb.process(audio)

# Save
save_audio(output, sr, "output.mp3")
```

### DDSP Training

```python
from ddsp_reverb import LearnableFDNReverb, CombinedLoss
from io_utils import load_audio

# Load dry and target audio
dry, sr = load_audio("dry.mp3")
target, _ = load_audio("target_reverb.mp3")

# Create learnable reverb
reverb = LearnableFDNReverb(sr=sr, num_delays=8)

# Setup training
optimizer = torch.optim.Adam(reverb.parameters(), lr=0.01)
criterion = CombinedLoss()

# Train
for epoch in range(200):
    output = reverb(dry)
    loss = criterion(output, target)['total']
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Get learned parameters
params = reverb.get_parameters()
print(params)
```

---

## Common Use Cases

### 1. Audio Production (Traditional FDN)

**Scenario**: Add reverb to vocals, instruments, or full mixes

```bash
# Subtle vocal reverb
python main.py -i vocals.mp3 -o vocals_reverb.mp3 --feedback_gain 0.7 --wet 0.3

# Lush pad/strings reverb
python main.py -i strings.mp3 -o strings_reverb.mp3 --feedback_gain 0.95 --wet 0.8
```

### 2. Clone Hardware Reverb (DDSP Style Transfer)

**Scenario**: Reverse-engineer expensive hardware (e.g., Lexicon 480L)

```bash
# 1. Record dry input + Lexicon output
# 2. Train DDSP to learn parameters
python app_style_transfer.py \
    --dry lexicon_dry.wav \
    --wet lexicon_output.wav \
    --new my_mix.mp3

# Result: Software clone with interpretable parameters
```

### 3. Intelligent Mixing (DDSP Adaptive)

**Scenario**: Auto-adjust reverb based on audio dynamics

```bash
python app_adaptive_reverb.py \
    --input podcast.mp3 \
    --output podcast_enhanced.mp3 \
    --rules smart

# More reverb during quiet sections
# Less reverb during loud sections
```

### 4. Research & Experiments (Comparison Tools)

**Scenario**: Benchmark different implementations

```bash
python run_comparison.py --input dry.mp3 --full

# Compare Traditional vs DDSP:
# - Performance (81x vs 21x realtime)
# - Quality (spectral distance < 1000 = very similar)
# - Trade-offs for your use case
```

---

## Performance

### Speed Comparison

| Implementation | Realtime Factor | Best For |
|----------------|-----------------|----------|
| Traditional FDN (C++) | 81x | Production, real-time |
| Traditional FDN (Python) | 4x | Development, debugging |
| DDSP FDN | 21x | Training, research |

**Realtime factor**: How many seconds of audio processed per second of time
- 81x = Process 81 seconds of audio in 1 second

### Memory Usage

- Traditional: ~12 MB
- DDSP: ~45 MB

### Scalability

Both scale linearly with audio length. For very long files (>5 minutes), consider batch processing.

---

## Troubleshooting

### Issue: "No module named torch"

**Solution**: Install PyTorch
```bash
pip install torch torchaudio
```

### Issue: MP3 files won't load

**Solution**: Install FFmpeg
```bash
# macOS
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg
```

### Issue: C++ extension compilation fails

**Solution**: Python fallback will be used automatically (10-20x slower but works)

If you need C++ speed, ensure you have a C++ compiler:
```bash
# macOS
xcode-select --install

# Linux
sudo apt-get install build-essential
```

### Issue: CUDA out of memory (DDSP)

**Solution**: DDSP automatically falls back to CPU. To force CPU:
```python
device = torch.device('cpu')
```

### Issue: Training not converging (DDSP)

**Solution**:
- Increase epochs (try 500+)
- Lower learning rate (try 0.005)
- Use longer reference audio (10+ seconds)
- Check dry/wet alignment

---

## Contributing

Contributions welcome! Areas of interest:
- New DDSP applications
- Performance optimizations
- Additional loss functions
- Integration with other frameworks

Please open an issue to discuss before submitting PRs.

---

## Citation

If you use this in research:

```bibtex
@misc{fdn_reverb_2025,
  title={FDN Reverb: High-Performance Traditional and DDSP Implementations},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/FDN-Reverb}
}
```

**Original FDN paper**:
```bibtex
@inproceedings{jot1991digital,
  title={Digital delay networks for designing artificial reverberators},
  author={Jot, Jean-Marc and Chaigne, Antoine},
  booktitle={Audio Engineering Society Convention 90},
  year={1991},
  organization={Audio Engineering Society}
}
```

**DDSP framework**:
```bibtex
@article{engel2020ddsp,
  title={DDSP: Differentiable digital signal processing},
  author={Engel, Jesse and Hantrakul, Lamtharn and Gu, Chenjie and Roberts, Adam},
  journal={arXiv preprint arXiv:2001.04643},
  year={2020}
}
```

---

## Related Projects

- [dafx25-ddsp-tutorial](https://github.com/yzyouzhang/dafx25-ddsp-tutorial) - FLAMO-based DDSP tutorial
- [FLAMO](https://github.com/gdalsanto/flamo) - Flexible Audio Modeling library
- [Google DDSP](https://github.com/magenta/ddsp) - Original DDSP library

See [COMPARISON.md](COMPARISON.md) for detailed comparison.

---

## License

[Your License Here]

---

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/FDN-Reverb/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/FDN-Reverb/discussions)
- **Email**: your.email@example.com

---

**Built with ❤️ for audio researchers and producers**

Happy reverb processing! 🎵
