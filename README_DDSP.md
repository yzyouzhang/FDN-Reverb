# DDSP FDN Reverb - Learnable Version

This repository now includes a **DDSP (Differentiable Digital Signal Processing)** version of the FDN reverb, where parameters can be learned automatically from target audio using gradient descent.

## What's New?

### Traditional FDN (reverb_util.py)
- **Manual parameters**: You set delays, feedback, damping, etc.
- **Fast processing**: C++ optimized for real-time use
- **Great for**: Audio production when you know what you want

### DDSP FDN (ddsp_reverb.py)
- **Learnable parameters**: Automatically optimized from target audio
- **Gradient-based training**: Uses PyTorch autograd
- **Great for**: Learning from examples, reverse-engineering, research

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

Requirements:
- PyTorch 2.0+
- torchaudio
- matplotlib (for plotting)
- numpy

### Demo - See DDSP in Action

Run the interactive demo to see DDSP learning in action:

```bash
python demo_ddsp.py
```

This will:
1. **Learn from target**: Create a reverb with known parameters, then use DDSP to discover them
2. **Clone existing reverb**: Learn to replicate an existing reverb effect
3. **Optimize for goals**: Find parameters that meet specific objectives

Generated files:
- `demo_target.mp3` - Target reverb (known parameters)
- `demo_learned.mp3` - DDSP discovered parameters
- `demo_cloned.mp3` - Cloned from existing reverb
- `demo_optimized.mp3` - Optimized for specific goal

## Usage Examples

### Example 1: Train to Match Target Audio

You have a recording of a space (e.g., concert hall) and want to learn its reverb characteristics:

```bash
# Record dry and wet audio from the target space
python train_ddsp.py --dry dry.mp3 --target concert_hall.mp3 --epochs 200 --output learned_hall.mp3
```

This will:
- Load your dry and target audio
- Train an FDN to match the target
- Save the learned reverb and trained model
- Generate training curves

### Example 2: Clone a Reverb You Love

You have a reverb sound you like (maybe from hardware) and want to replicate it:

```bash
python train_ddsp.py --dry input.mp3 --target beloved_reverb.mp3 --epochs 300
```

After training, you'll have:
- Interpretable parameters (delays, feedback, damping, etc.)
- A software version you can modify
- Understanding of why it sounds that way

### Example 3: Python API

```python
import torch
from ddsp_reverb import LearnableFDNReverb, CombinedLoss
from io_utils import load_audio

# Load audio
dry, sr = load_audio("dry.mp3", mono=True)
target, _ = load_audio("target_reverb.mp3", mono=True)

# Create learnable reverb
reverb = LearnableFDNReverb(
    sr=sr,
    num_delays=8,
    learnable_delays=True,      # Optimize delay times
    learnable_feedback=True,    # Optimize feedback gain
    learnable_damping=True,     # Optimize damping
    learnable_wet=True,         # Optimize wet/dry mix
)

# Setup training
optimizer = torch.optim.Adam(reverb.parameters(), lr=0.01)
criterion = CombinedLoss(spectral_weight=1.0, time_weight=0.1)

# Train
for epoch in range(200):
    optimizer.zero_grad()
    output = reverb(dry)
    loss_dict = criterion(output, target)
    loss = loss_dict['total']
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Inspect learned parameters
params = reverb.get_parameters()
print(f"Learned delays: {params['delays_ms']}")
print(f"Learned feedback: {params['feedback_gain']:.3f}")
print(f"Learned damping: {params['damping']:.3f}")

# Use the trained reverb
reverb.eval()
with torch.no_grad():
    new_output = reverb(new_audio)
```

## Command-Line Options

### Training Script (train_ddsp.py)

```bash
python train_ddsp.py [options]
```

**Input/Output**:
- `--dry`, `-i`: Dry input audio file
- `--target`, `-t`: Target reverb audio (what to match)
- `--output`, `-o`: Output file path (default: `ddsp_trained.mp3`)
- `--model_save`: Save trained model (default: `ddsp_reverb_model.pt`)
- `--plot`: Save training curve plot (default: `training_curves.png`)

**Training Parameters**:
- `--epochs`: Number of training iterations (default: 200)
- `--lr`: Learning rate (default: 0.01)
- `--num_delays`: Number of delay lines, must be power of 2 (default: 8)

**Loss Weights**:
- `--spectral_weight`: Weight for spectral loss (default: 1.0)
- `--time_weight`: Weight for time-domain loss (default: 0.1)

**Freeze Parameters** (make them non-learnable):
- `--freeze_delays`: Don't optimize delay times
- `--freeze_feedback`: Don't optimize feedback gain
- `--freeze_damping`: Don't optimize damping
- `--freeze_wet`: Don't optimize wet/dry mix
- `--freeze_modulation`: Don't optimize modulation

**Examples**:

```bash
# Basic training
python train_ddsp.py --dry input.mp3 --target hall.mp3 --epochs 200

# Only learn feedback and damping (freeze others)
python train_ddsp.py --dry input.mp3 --target hall.mp3 \
    --freeze_delays --freeze_wet --freeze_modulation

# Use more spectral loss for better frequency matching
python train_ddsp.py --dry input.mp3 --target hall.mp3 \
    --spectral_weight 2.0 --time_weight 0.05

# Train with 16 delay lines for richer reverb
python train_ddsp.py --dry input.mp3 --target hall.mp3 --num_delays 16
```

## Understanding the Parameters

After training, you can inspect what DDSP learned:

```python
params = reverb.get_parameters()
```

Returns:
```python
{
    'delays_ms': array([29.3, 38.1, 43.7, ...]),  # Delay times in milliseconds
    'feedback_gain': 0.87,                         # Overall feedback strength
    'damping': 0.32,                               # High-frequency decay
    'wet_mix': 0.65,                               # Wet/dry ratio
    'mod_depth_ms': 0.8,                           # Modulation depth
    'mod_rate_hz': 0.23,                           # Modulation rate
}
```

### Parameter Meanings

**delays_ms**:
- Delay line lengths (coprime/prime numbers work best)
- Longer delays = longer reverb tail
- Different lengths = better diffusion

**feedback_gain** (0-1):
- Controls reverb tail length
- Higher = longer tail
- Must be < 1.0 for stability
- Typical: 0.7-0.95

**damping** (0-1):
- High-frequency absorption
- Higher = darker sound (faster HF decay)
- Simulates air absorption
- Typical: 0.2-0.5

**wet_mix** (0-1):
- 0 = completely dry (no reverb)
- 1 = completely wet (all reverb)
- Typical: 0.3-0.7

**mod_depth_ms**, **mod_rate_hz**:
- LFO modulation to prevent metallic artifacts
- Depth: how much delays vary (ms)
- Rate: how fast they vary (Hz)
- Typical: 0.5-2.0ms depth, 0.1-0.5Hz rate

## Loss Functions

### SpectralLoss
Compares magnitude spectrograms at multiple FFT sizes (512, 1024, 2048).
- Good for matching frequency content
- Perceptually relevant
- Used in most DDSP papers

### TimeDomainLoss
Simple L1 or L2 loss in time domain.
- Good for matching waveform shape
- Helps with transients

### CombinedLoss
Combines spectral + time-domain loss.
- Best of both worlds
- Weights control balance
- Default: `spectral_weight=1.0, time_weight=0.1`

## Use Cases

### 1. Reverse-Engineer Spaces
Record a real space and learn its reverb:
```bash
# Record dry sound and its reverb in a cathedral
python train_ddsp.py --dry dry_clap.wav --target cathedral_clap.wav
```

### 2. Clone Hardware Reverb
Replicate expensive reverb units:
```bash
# Record input/output from hardware
python train_ddsp.py --dry input.wav --target lexicon_output.wav
```

### 3. Match Reference Tracks
Learn reverb from your favorite recordings:
```bash
# Extract vocal stems
python train_ddsp.py --dry dry_vocal.wav --target reference_vocal.wav
```

### 4. Research & Experimentation
Explore reverb algorithms:
```python
# What if we use 32 delay lines?
reverb = LearnableFDNReverb(sr=48000, num_delays=32)
# Let DDSP find optimal parameters!
```

### 5. Adaptive Reverb
Combine with neural networks for context-aware reverb:
```python
class SmartReverb(nn.Module):
    def __init__(self):
        self.analyzer = YourNeuralNet()
        self.reverb = LearnableFDNReverb(sr=48000)

    def forward(self, audio, context):
        params = self.analyzer(context)  # e.g., room photo
        return self.reverb(audio)
        # Parameters adjust based on context!
```

## Technical Details

### How Training Works

1. **Forward Pass**: Process dry audio through FDN with current parameters
2. **Compute Loss**: Compare output to target using spectral + time-domain loss
3. **Backward Pass**: Compute gradients via PyTorch autograd
4. **Update**: Adjust parameters using optimizer (Adam)
5. **Repeat**: Until loss converges

### Parameter Constraints

Parameters are constrained using transformations:

- **Delays**: Stored in log space → `delays = exp(log_delays)` (always positive)
- **Feedback/Damping/Wet**: Stored as logits → `value = sigmoid(logit)` (always 0-1)
- **Modulation**: Stored in log space → `value = exp(log_value)` (always positive)

This ensures parameters stay in valid ranges during optimization.

### Differentiability

All operations are differentiable:
- ✓ Delay line reading (linear interpolation)
- ✓ Feedback mixing (matrix multiplication)
- ✓ Damping (lowpass filter)
- ✓ Modulation (sinusoidal LFO)

Gradients flow through the entire processing chain.

## Comparison: Traditional vs. DDSP

| Aspect | Traditional FDN | DDSP FDN |
|--------|----------------|----------|
| **Parameter Setting** | Manual tuning | Automatic learning |
| **Speed** | Very fast (C++) | Slower (training overhead) |
| **Use Case** | Production audio | Research, learning from data |
| **Control** | Direct | Learned from examples |
| **Flexibility** | Fixed algorithms | Trainable, composable |
| **Interpretability** | Parameters set by user | Parameters discovered |

**When to use each**:
- **Traditional**: You know what you want, need speed
- **DDSP**: You want to match a target, discover parameters

## Tips for Training

### 1. Start with Good Initialization
Initialize close to expected values:
```python
reverb = LearnableFDNReverb(
    sr=sr,
    init_feedback_gain=0.8,  # If you expect long tail
    init_damp=0.3,            # If you expect bright reverb
)
```

### 2. Use Appropriate Loss Weights
- More spectral weight → better frequency matching
- More time weight → better waveform matching
- Default (1.0 spectral, 0.1 time) works well

### 3. Freeze Some Parameters
If you know certain parameters:
```python
reverb = LearnableFDNReverb(
    sr=sr,
    learnable_feedback=False,  # Keep feedback fixed
    init_feedback_gain=0.9,    # At this value
)
```

### 4. Training Duration
- Simple matching: 100-200 epochs
- Complex matching: 300-500 epochs
- Watch loss curves to avoid overfitting

### 5. Audio Length
- Longer audio = better learning
- At least 5-10 seconds recommended
- Include diverse content (transients + sustained)

## Limitations

**DDSP FDN is not a silver bullet**:

1. **Training Time**: Requires minutes to train (vs. instant with traditional)
2. **Quality Depends on Data**: Can only learn what's in the target
3. **Local Minima**: May not find global optimum (try multiple runs)
4. **Computational Cost**: Slower than C++ optimized version during training

**When DDSP might not help**:
- You already know exact parameters you want
- Need real-time performance during development
- Working with very short audio clips
- Target reverb is too complex for FDN to model

## Future Improvements

Possible extensions:
- [ ] Learnable feedback matrix (beyond Hadamard)
- [ ] Time-varying parameters (reverb that changes)
- [ ] Multi-band damping (per-frequency control)
- [ ] Perceptual loss functions (using pre-trained models)
- [ ] Integration with neural networks for control

## References

### DDSP
- [DDSP: Differentiable Digital Signal Processing (Engel et al., 2020)](https://arxiv.org/abs/2001.04643)
- [DDSP-VST: Data-Driven Synthesizer Plugin (Masuda & Saito, 2021)](https://arxiv.org/abs/2109.00103)

### FDN Theory
```
@inproceedings{jot1991digital,
  title={Digital delay networks for designing artificial reverberators},
  author={Jot, Jean-Marc and Chaigne, Antoine},
  booktitle={Audio Engineering Society Convention 90},
  year={1991},
  organization={Audio Engineering Society}
}
```

### Related Work
- [FLAMO: Flexible Audio Modeling](https://github.com/gdalsanto/flamo)
- [DAFx25 DDSP Tutorial](https://github.com/yzyouzhang/dafx25-ddsp-tutorial)

## Contributing

Contributions welcome! Possible areas:
- New loss functions
- Additional learnable components
- Integration with neural networks
- Better optimization strategies

## License

Same as main repository.

---

**Happy DDSP learning!** 🎵🤖

For questions or issues, please open a GitHub issue.
