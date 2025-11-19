# Repository Comparison: FDN-Reverb vs. DAFx25-DDSP-Tutorial

## Executive Summary

Both repositories implement **Feedback Delay Network (FDN) reverb**, but serve different purposes:

- **FDN-Reverb** (this repo): Production-ready, high-performance audio processing tool
- **dafx25-ddsp-tutorial**: Educational/research framework for differentiable DSP with machine learning

---

## Side-by-Side Comparison

| Aspect | FDN-Reverb | dafx25-ddsp-tutorial |
|--------|------------|---------------------|
| **Primary Purpose** | Audio processing tool | Educational tutorial / ML research |
| **Target Users** | Audio engineers, musicians | Researchers, ML practitioners |
| **Approach** | Fixed parameters (user-controlled) | Learnable parameters (ML-based) |
| **Framework** | PyTorch + custom C++ | FLAMO library (DDSP framework) |
| **Performance** | Optimized for speed (C++ extensions) | Optimized for differentiability |
| **Parameter Control** | Manual tuning | Gradient-based optimization |
| **Use Case** | Real-time/offline audio processing | Training reverb models from data |
| **Complexity** | Single-purpose, streamlined | Multi-component DDSP pipeline |

---

## Detailed Comparison

### 1. Architecture & Implementation

#### FDN-Reverb
- **Pure implementation**: Custom PyTorch implementation with optional C++ acceleration
- **8 delay lines** (default, configurable to powers of 2)
- **Hadamard feedback matrix** for orthogonal mixing
- **Fixed architecture**: All components are pre-defined
- **Performance**: ~2.4 seconds to process 2-minute song (with C++ extension)
- **Platform**: CPU-only (Mac, Linux, Windows)

#### dafx25-ddsp-tutorial
- **Framework-based**: Uses FLAMO library for differentiable DSP
- **Configurable architecture**: Allows experimentation with different FDN topologies
- **Differentiable components**: All operations support backpropagation
- **Training capability**: Can learn parameters from target reverb examples
- **Educational focus**: Jupyter notebooks with step-by-step explanations

### 2. Parameter Philosophy

#### FDN-Reverb - Manual Control
Users directly set parameters:
- Delay times (ms): `(29, 37, 43, 53, 61, 71, 79, 89)`
- Feedback gain: `0.9` (controls tail length)
- Damping: `0.25` (high-frequency decay)
- Wet/dry mix: `0.9`
- Modulation depth/rate: `0.5 ms`, `0.3 Hz`

**Design goal**: Intuitive control for musicians and audio engineers

#### dafx25-ddsp-tutorial - Learned Control
Uses gradient descent to optimize parameters:
- **Loss functions**: Compare output to target reverb
- **Backpropagation**: Automatically adjust all parameters
- **Training data**: Audio examples with desired reverb characteristics
- **Differentiable pipeline**: All DSP operations support gradients

**Design goal**: Learn reverb characteristics from data

### 3. Core Technology Differences

#### FDN-Reverb
```python
# Direct parameter specification
reverb = FDNReverb(
    sr=48000,
    delays_ms=(29, 37, 43, 53, 61, 71, 79, 89),
    feedback_gain=0.9,
    damp=0.25,
    wet=0.9
)
output = reverb.process(audio)
```

#### dafx25-ddsp-tutorial (Conceptual)
```python
# Learnable parameters via FLAMO
fdn = FlameableFDN(num_delays=8)  # Parameters are learnable
optimizer = torch.optim.Adam(fdn.parameters())

# Training loop
for epoch in range(epochs):
    output = fdn(input_audio)
    loss = loss_function(output, target_audio)
    loss.backward()  # Gradients flow through DSP operations
    optimizer.step()
```

---

## Theoretical Background

### What is Feedback Delay Network (FDN)?

FDN is a classic artificial reverberation algorithm introduced by Jot & Chaigne (1991). It models how sound reflects in physical spaces.

#### Core Concept
Sound in a room bounces off walls/surfaces multiple times, creating:
1. **Early reflections**: Distinct echoes (first few bounces)
2. **Late reverberation**: Dense, diffuse sound (many overlapping reflections)

FDN focuses on modeling the **late reverberation** using a network of delay lines.

#### FDN Architecture

```
Input Signal (x)
      |
      v
  [Distribute to N delay lines]
      |
      v
  +---------------------------+
  | Delay Line 1   Delay Line 2   ...  Delay Line N |
  |     |              |                    |        |
  |     v              v                    v        |
  |  [Buffer]      [Buffer]             [Buffer]    |
  |     |              |                    |        |
  +-----|--------------|--------------------|---------+
        |              |                    |
        v              v                    v
      [Feedback Matrix A - Mixes all delay outputs]
        |              |                    |
        v              v                    v
      [Damping Filter (lowpass for high-freq decay)]
        |              |                    |
        +-------> [Feed back to delay inputs] <------+
        |
        v
   [Output: Mix/Sum of delay lines]
        |
        v
   [Wet/Dry Mix with Input]
        |
        v
    Output Signal (y)
```

#### Key Components

1. **Delay Lines (N buffers)**
   - Store audio samples for different time periods
   - Typical delays: 20-90ms (coprime/prime numbers for better diffusion)
   - Each line represents different reflection paths

2. **Feedback Matrix (A)**
   - Mixes outputs from all delay lines before feeding back
   - **Hadamard matrix**: Orthogonal, energy-preserving
   - Creates complex interaction between delay paths
   - **Critical for stability**: Must be orthogonal and scaled by feedback gain < 1.0

3. **Damping (Lowpass Filters)**
   - Models air absorption (high frequencies decay faster in real rooms)
   - Simple one-pole lowpass filter in each feedback path
   - Controls "brightness" of reverb tail

4. **Modulation (LFO)**
   - Slight time-varying changes to delay times
   - Prevents metallic "ringing" artifacts
   - Simulates small variations in reflection paths

5. **Wet/Dry Mix**
   - Blends processed (wet) signal with original (dry) signal
   - Controls reverb amount

#### Mathematical Formulation

For delay line `i` at time `n`:

```
s_i[n] = delay_buffer_i[n - D_i]                    (read delayed signal)
y[n] = Σ s_i[n] / N                                 (output = average of delays)
feedback_i[n] = Σ A_ij * s_j[n]                     (feedback mixing)
damped_i[n] = (1-d) * lp_i[n-1] + d * feedback_i[n] (lowpass damping)
delay_buffer_i[n] = damped_i[n] + x[n] / N          (write to buffer)
output[n] = (1-w) * x[n] + w * y[n]                 (wet/dry mix)
```

Where:
- `D_i`: Delay time for line i (samples)
- `A_ij`: Feedback matrix coefficients
- `d`: Damping factor (0-1)
- `w`: Wet mix (0-1)
- `N`: Number of delay lines

#### Why Hadamard Matrix?

The feedback matrix must be:
1. **Orthogonal**: Preserves energy (prevents buildup/decay without feedback gain)
2. **Efficient**: Hadamard matrix has simple construction
3. **Scalable**: Works for any power-of-2 size

The Hadamard matrix recursively defined as:
```
H_1 = [1]

H_n = [H_{n/2}   H_{n/2}  ]
      [H_{n/2}  -H_{n/2}  ]
```

Normalized: `A = (feedback_gain / √N) * H_N`

---

### What is DDSP (Differentiable Digital Signal Processing)?

DDSP is a modern approach combining classical DSP with machine learning.

#### Core Innovation
Make traditional DSP operations **differentiable** so they can be:
1. Trained with gradient descent
2. Integrated into neural networks
3. Learned from data instead of hand-tuned

#### Traditional DSP vs. DDSP

**Traditional DSP** (like FDN-Reverb):
- Parameters set manually by audio engineers
- Based on acoustic theory and experience
- Fixed behavior for given parameters

**DDSP** (like dafx25-ddsp-tutorial):
- Parameters learned from examples
- Uses backpropagation to optimize
- Can match target audio characteristics

#### DDSP Pipeline Example

```
Input Audio
    |
    v
[Neural Network] --> Extract features/control signals
    |
    v
[Differentiable DSP] --> FDN with learnable parameters
    |                     (delays, feedback, damping, etc.)
    v
Output Audio
    |
    v
[Loss Function] --> Compare with target
    |
    v
[Backpropagation] --> Update all parameters
```

#### Why DDSP Matters

1. **Data-driven**: Learn from recordings of real spaces
2. **Controllable**: Combine learned models with intuitive controls
3. **Efficient**: Physics-based DSP is more efficient than pure neural audio
4. **Interpretable**: Parameters have physical meaning (unlike black-box neural nets)

#### FLAMO Library

The dafx25-ddsp-tutorial uses FLAMO (Flexible Audio Modeling), which provides:
- Differentiable DSP building blocks
- Pre-built modules (filters, delays, oscillators)
- Training utilities for audio ML
- Integration with PyTorch ecosystem

---

## When to Use Each Approach?

### Use FDN-Reverb when:
- You need to process audio files with reverb
- You want manual control over reverb characteristics
- Performance/speed is critical
- You're building audio production tools
- You have specific reverb parameters in mind

### Use dafx25-ddsp-tutorial when:
- You want to learn reverb parameters from data
- You're researching new reverb algorithms
- You need to match a specific reverb "sound"
- You're building ML-based audio tools
- You want to understand DDSP concepts

---

## Technical Highlights

### FDN-Reverb Strengths
1. **Performance**: C++ extensions for 100x+ speedup
2. **Simplicity**: Single-purpose, easy to use
3. **Production-ready**: Robust, well-tested
4. **Documentation**: Clear examples and API

### dafx25-ddsp-tutorial Strengths
1. **Educational**: Jupyter notebooks with explanations
2. **Flexibility**: Experiment with different architectures
3. **Research-oriented**: Explore novel reverb designs
4. **ML Integration**: Train reverb models from data

---

## Complementary Use Cases

These repositories complement each other:

1. **Learn with dafx25-ddsp-tutorial**
   - Understand FDN theory
   - Experiment with differentiable DSP
   - Train custom reverb models

2. **Deploy with FDN-Reverb**
   - Take learned parameters/insights
   - Implement in optimized production code
   - Process audio efficiently

---

## Key References

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

### DDSP Approach
- FLAMO library: https://github.com/gdalsanto/flamo
- DDSP framework: Google Magenta's DDSP library
- DAFx 2025 Tutorial: "Building Flexible Audio DDSP Pipelines"

---

## Conclusion

Both repositories implement the same core algorithm (FDN reverb) but with different philosophies:

- **FDN-Reverb**: Fast, practical tool for audio processing
- **dafx25-ddsp-tutorial**: Educational framework for ML-based audio research

Understanding both approaches provides a complete picture of modern reverb design:
- Classical DSP theory (FDN)
- Modern ML techniques (DDSP)
- Production optimization (C++ extensions)
- Research flexibility (differentiable frameworks)

Choose based on your goal: production audio processing or research/learning about ML-based audio synthesis.
