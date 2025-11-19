# FDN Reverb - High-Performance Audio Reverb Processor

A fast Feedback Delay Network (FDN) reverb implementation using PyTorch, with optional C++ extensions for maximum performance. Processes a 2 minutes 48khz stereo song in ~2.4 seconds on M4 MacBook Pro!

**NEW**: 🎓 **DDSP Version Available!** - Learn reverb parameters automatically from target audio using gradient descent. See [README_DDSP.md](README_DDSP.md) for details.

**NEW**: 📊 **Comparison Tools!** - Comprehensive benchmarks comparing performance, quality, and characteristics. See [README_COMPARISON.md](README_COMPARISON.md) for details.

**NEW**: 🚀 **DDSP Applications!** - Neural reverb controller, style transfer, and adaptive reverb demos. See [README_APPLICATIONS.md](README_APPLICATIONS.md) for details.

## Quick Links

- [Traditional FDN Usage](#command-line-usage) - Fast audio processing
- [DDSP Version](README_DDSP.md) - Learnable reverb
- [DDSP Applications](README_APPLICATIONS.md) - Neural networks, style transfer, adaptive
- [Comparison Tools](README_COMPARISON.md) - Benchmarks and metrics
- [Theory Comparison](COMPARISON.md) - vs dafx25-ddsp-tutorial

## Two Versions

This repository now includes **two implementations**:

1. **Traditional FDN** (`reverb_util.py`, `main.py`)
   - Manual parameter control
   - C++ optimized for speed
   - Great for audio production

2. **DDSP FDN** (`ddsp_reverb.py`, `train_ddsp.py`, `demo_ddsp.py`)
   - Learnable parameters via gradient descent
   - Automatically match target reverb
   - Great for research and learning from examples
   - See [README_DDSP.md](README_DDSP.md) and [COMPARISON.md](COMPARISON.md)

## DDSP Applications

Practical demos showcasing DDSP capabilities:

```bash
# Neural network predicts optimal reverb
python app_neural_reverb.py

# Learn reverb from reference recordings
python app_style_transfer.py --demo

# Context-aware adaptive reverb
python app_adaptive_reverb.py --demo
```

Applications:
- **Neural Controller**: Auto-suggest reverb based on audio content
- **Style Transfer**: Clone reverb from famous studios/hardware
- **Adaptive Reverb**: Real-time parameter adjustment based on dynamics

See [README_APPLICATIONS.md](README_APPLICATIONS.md) for details.

## Comparison Tools

Benchmark and compare implementations:

```bash
# Run complete comparison suite
python run_comparison.py --input dry.mp3 --full

# Quick comparison
python run_comparison.py --quick
```

Generates:
- Performance benchmarks (speed, memory, scalability)
- Audio quality metrics (RT60, spectral analysis)
- Visualizations (waveforms, spectrograms, frequency response)

See [README_COMPARISON.md](README_COMPARISON.md) for details.

## Technical Details

If you are interested in more details, please read my [blog](https://puar-playground.github.io/posts/reverb/): 🎵 FDN Reverb - Ripples of Space and Time.


## Features

- 🚀 **High Performance**: C++ extension for > 100x speedup over pure Python
- 🎛️ **Cross-Platform**: Works on CPU (Mac, Linux, Windows)
- 🎵 **Multi-Format Support**: Load/save MP3, WAV, and other audio formats
- 🎚️ **Flexible Parameters**: Adjustable feedback gain, damping, wet/dry mix, and modulation
- 🔊 **Automatic Volume Matching**: Peak-based normalization to match input/output levels
- 🎧 **Stereo Support**: Process mono or stereo audio with batch processing

## Demo Music Files
**Note**: The default input file `dry.mp3` is a MIDI-rendered version of “Prelude” from the Final Fantasy series, without any reverb.

<audio controls>
  <source src="https://raw.githubusercontent.com/puar-playground/FDN-Reverb/main/dry.mp3" type="audio/mpeg">
</audio>

And `wet_fdn.mp3` is the output processed with default parameters.

<audio controls>
  <source src="https://raw.githubusercontent.com/puar-playground/FDN-Reverb/main/wet_fdn.mp3" type="audio/mpeg">
</audio>


### Command-Line Usage

```bash
# Basic usage with default parameters
python main.py

# Custom input/output files
python main.py -i input.mp3 -o output.mp3

# Adjust reverb parameters
python main.py --feedback_gain 0.95 --wet 0.8 --damp 0.3

# Custom delay times
python main.py --delays_ms 20 30 40 50 60 70 80 90

# See all options
python main.py --help
```

### Python API Usage

```python
import torch
from reverb_util import FDNReverb
from io_utils import save_audio, load_audio

# Load audio
audio, sr = load_audio("input.mp3", mono=False, target_sample_rate=None)

# Create reverb
reverb = FDNReverb(
    sr=sr,
    delays_ms=(29, 37, 43, 53, 61, 71, 79, 89),
    feedback_gain=0.9,
    damp=0.25,
    wet=0.9,
    mod_depth_ms=0.5,
    mod_rate_hz=0.3
)

# Process audio
wet = reverb.process(audio)

# Save output
save_audio(wet, sample_rate=sr, file_path="output.mp3")
```

## Installation

### Requirements

- PyTorch and torchaudio
- FFmpeg (for MP3 support)

### Install Dependencies

```bash
pip install torch torchaudio
```

**Note**: For MP3 support, you may need to install FFmpeg:
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt-get install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/)

## Usage

### Command-Line Options

#### Input/Output
- `--input`, `-i`: Input audio file path (default: `dry.mp3`)
- `--output`, `-o`: Output audio file path (default: `wet_fdn.mp3`)
- `--mono`: Convert input to mono (default: stereo)
- `--target_sample_rate`: Resample to target sample rate (default: preserve original)

#### Reverb Parameters
- `--delays_ms`: Delay times in milliseconds, space-separated (default: `29 37 43 53 61 71 79 89`)
- `--feedback_gain`: Feedback gain 0~1, larger = longer tail, must be <1 (default: `0.9`)
- `--damp`: Damping factor 0~1, larger = faster high-frequency decay (default: `0.25`)
- `--wet`: Wet/dry mix ratio 0~1, 0=dry, 1=fully wet (default: `0.9`)
- `--mod_depth_ms`: Modulation depth in milliseconds (default: `0.5`)
- `--mod_rate_hz`: Modulation rate in Hz (default: `0.3`)
- `--output_gain`: Output gain scaling factor (default: `1.0`)

#### Other Options
- `--no_volume_match`: Disable automatic volume matching between input and output

### Example Commands

```bash
# Quick processing with defaults
python main.py -i song.mp3 -o reverb_song.mp3

# Create a longer, more pronounced reverb
python main.py --feedback_gain 0.95 --wet 0.9 --damp 0.2

# Short, subtle reverb
python main.py --feedback_gain 0.7 --wet 0.3 --damp 0.4

# Custom delay configuration
python main.py --delays_ms 15 23 31 47 59 67 73 83
```

## Parameters

### FDNReverb Parameters

- `sr` (int): Sample rate in Hz
- `delays_ms` (tuple): Delay times in milliseconds. Use coprime/prime numbers for better diffusion
- `feedback_gain` (float, 0~1): Feedback gain. Larger values = longer reverb tail. Must be < 1.0
- `damp` (float, 0~1): Damping factor. Larger values = faster high-frequency decay
- `wet` (float, 0~1): Wet/dry mix ratio (0 = dry, 1 = fully wet)
- `mod_depth_ms` (float): Modulation depth in milliseconds. Light modulation avoids metallic artifacts
- `mod_rate_hz` (float): Modulation rate in Hz
- `output_gain` (float): Output gain scaling factor (default: 1.0)

### Number of Delay Lines

**Important Constraint**: The number of delay lines must be a **power of 2** (2, 4, 8, 16, 32, 64, etc.) because the Hadamard feedback matrix requires this. The default is **8 delay lines**.

**What happens if you add more delay lines?**

**Pros:**
- ✅ **Richer, more complex reverb** - More delay lines create better diffusion and more echo paths
- ✅ **Better spatial spread** - More paths for sound to travel through
- ✅ **More realistic tails** - Denser echo patterns sound more natural

**Cons:**
- ⚠️ **Slower processing** - Feedback matrix multiplication is O(N²), so 16 lines is ~4x slower than 8
- ⚠️ **More memory** - Each delay line needs its own buffer
- ⚠️ **Diminishing returns** - Beyond 16 lines, the improvement is often minimal

**Performance Impact:**
- **8 lines** (default): Baseline speed
- **16 lines**: ~2x slower
- **32 lines**: ~4x slower  
- **64 lines**: ~8x slower

**Recommendations:**
- **8 lines**: Good balance of quality and speed (default)
- **16 lines**: Richer reverb, ~2x slower - good for final mixes
- **32+ lines**: Usually not worth the slowdown unless you need maximum quality

**Example with 16 delay lines:**
```bash
python main.py --delays_ms 19 23 29 31 37 41 43 47 53 59 61 67 71 73 79 83
```

**Note**: The code will raise an error if you provide a number of delay times that is not a power of 2.

## Performance

### Automatic Optimization

The code automatically tries to use the fastest available implementation:

1. **CPU C++ Extension**: Fast compiled C++ code, works on all platforms (auto-compiles on first run)
2. **Python Fallback**: Pure PyTorch implementation (slower but always works)

### Expected Performance

- **With C++ Extension**: 5-20x faster than Python
- **Processing Time** (tested on M4 MacBook Pro, CPU-only):
  - ~2.4 seconds for a 2-minute stereo song (with C++ extension)
  - ~1.2 seconds per minute of audio (approximately)
  - Python fallback is significantly slower (10-20x slower)

**Note**: FDN reverb is inherently sequential due to feedback loops, so each sample depends on previous samples. This provides unique sonic characteristics.

### Performance Tips

1. **Use C++ Extension**: The code automatically compiles and uses C++ extensions on first run
2. **Lower Sample Rate**: Processing at 44.1kHz instead of 48kHz is ~10% faster
3. **Adjust Block Size**: For very long files, consider processing in chunks


## File Structure

```
Reverb/
├── main.py              # Command-line interface and example usage script
├── reverb_util.py       # Main FDN reverb implementation
├── io_utils.py          # Audio I/O utilities (load/save MP3, WAV, etc.)
├── fdn_cpu.cpp          # CPU C++ extension (optional, auto-compiles)\
└── README.md           # This file
```


## Reference

```
@inproceedings{jot1991digital,
  title={Digital delay networks for designing artificial reverberators},
  author={Jot, Jean-Marc and Chaigne, Antoine},
  booktitle={Audio Engineering Society Convention 90},
  year={1991},
  organization={Audio Engineering Society}
}
```
