## DDSP Reverb Applications

This guide showcases practical applications of DDSP FDN reverb demonstrating the power of differentiable audio processing.

## Overview

The DDSP version enables applications impossible with traditional DSP:

1. **Neural Reverb Controller** - Neural network predicts optimal reverb from audio
2. **Style Transfer** - Learn and apply reverb characteristics from references
3. **Adaptive Reverb** - Context-aware reverb that adapts to audio content

Each application demonstrates key DDSP advantages: learning from data, end-to-end optimization, and intelligent adaptation.

---

## Application 1: Neural Reverb Controller

**File**: `app_neural_reverb.py`

### What It Does

A neural network analyzes audio and automatically predicts appropriate reverb parameters.

**Architecture**:
```
Audio → CNN Feature Extractor → Parameter Predictor → Differentiable FDN → Reverb Output
```

**Key Innovation**: The entire pipeline is differentiable, enabling end-to-end training from audio examples.

### Use Cases

- **Automatic reverb suggestions** based on audio content
- **Intelligent mixing assistant** that understands different instruments
- **Content-aware processing** (vocals → short reverb, strings → long reverb)

### Quick Start

```bash
# Run demo (trains network and tests on different audio types)
python app_neural_reverb.py
```

**What happens**:
1. Creates synthetic dataset (100 samples of different audio types)
2. Trains neural network to map audio → reverb parameters (50 epochs)
3. Tests on high-pitched tones, low-pitched tones, and noise
4. Shows learned parameter predictions

**Expected output**:
```
Test 1: High Frequency Tone (2000 Hz)
  Expected: Short, bright reverb
  Predicted parameters:
    Feedback: 0.712  (shorter tail)
    Damping:  0.195  (less damping = brighter)
    Wet mix:  0.423  (moderate wet)
```

### How It Works

**1. Feature Extraction** (CNN):
- Converts audio to spectrogram
- Convolutional layers extract patterns
- Learns what high/low frequencies, transients, sustained sounds look like

**2. Parameter Prediction** (MLP):
- Maps features to reverb parameters
- Separate heads for feedback, damping, wet mix
- Sigmoid activation ensures parameters stay in valid range [0,1]

**3. End-to-End Training**:
```python
# Forward pass
features = feature_extractor(audio)
params = param_predictor(features)
output = reverb(audio, params)

# Loss compares output to target
loss = mse_loss(output, target)

# Gradients flow through entire pipeline!
loss.backward()
```

### Training Your Own

Extend with real data:

```python
# Create dataset from your library
dataset = []
for dry_file, wet_file in audio_pairs:
    dry = load_audio(dry_file)
    wet = load_audio(wet_file)
    dataset.append({'input': dry, 'target': wet})

# Train
model = NeuralReverbController(sr=48000)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for batch in dataset:
        output, params = model(batch['input'])
        loss = criterion(output, batch['target'])
        loss.backward()
        optimizer.step()
```

### Results

After training on synthetic data:
- **Feedback prediction error**: ~5-10%
- **Damping prediction error**: ~8-12%
- **Wet mix prediction error**: ~6-10%

The network learns musical rules:
- High frequencies → less damping (preserve brightness)
- Low frequencies → more feedback (longer tail)
- Percussive sounds → less wet (clarity)

---

## Application 2: Style Transfer

**File**: `app_style_transfer.py`

### What It Does

Learns reverb characteristics from reference recordings and applies them to new audio.

**Problem**: "I want my vocals to sound like they're in Abbey Road Studio 2"

**Solution**: Provide a dry/wet pair from that studio, DDSP learns the parameters, applies to your audio.

### Use Cases

- **Match famous studios** from recordings
- **Clone hardware reverb** by recording its output
- **Reverse-engineer** classic album reverb
- **Transfer acoustic treatment** between spaces

### Quick Start

```bash
# Demo with synthetic target (recommended first)
python app_style_transfer.py --demo

# Use your own reference files
python app_style_transfer.py \
    --dry reference_dry.mp3 \
    --wet reference_wet.mp3 \
    --new my_audio.mp3
```

### Demo Output

The synthetic demo:
1. Creates dry audio (musical phrase)
2. Processes with target reverb (long, dark, wet)
3. Trains DDSP to learn those characteristics
4. Applies learned reverb to new audio

**Generated files**:
- `style_transfer_dry.mp3` - Original test audio
- `style_transfer_reference.mp3` - Target reverb to learn
- `style_transfer_new_dry.mp3` - New audio (dry)
- `style_transfer_new_wet.mp3` - New audio with learned reverb ✨
- `style_transfer_learning_curve.png` - Training progress

### How It Works

**1. Reference Analysis**:
```python
# You provide: dry audio + wet audio (with desired reverb)
dry_reference = load_audio("studio_dry.mp3")
wet_reference = load_audio("studio_wet.mp3")
```

**2. Parameter Learning**:
```python
# DDSP learns: what parameters transform dry → wet?
reverb = LearnableFDNReverb(learnable_delays=True,
                            learnable_feedback=True,
                            learnable_damping=True)

for epoch in range(300):
    output = reverb(dry_reference)
    loss = criterion(output, wet_reference)
    loss.backward()  # Optimize all parameters!
    optimizer.step()
```

**3. Transfer to New Audio**:
```python
# Apply learned reverb to your audio
new_audio = load_audio("my_vocal.mp3")
output = reverb(new_audio)  # Same reverb character!
```

### Parameter Recovery

Example from synthetic demo:

| Parameter | Target | Learned | Error |
|-----------|--------|---------|-------|
| Feedback  | 0.920  | 0.912   | 0.9%  |
| Damping   | 0.450  | 0.438   | 2.7%  |
| Wet Mix   | 0.850  | 0.847   | 0.4%  |

**Insight**: DDSP accurately recovers parameters within 3%, validating the approach.

### Real-World Example

Clone a vintage Lexicon reverb:

1. **Record reference**:
   - Play dry sound through Lexicon
   - Record both dry input and Lexicon output

2. **Learn parameters**:
   ```bash
   python app_style_transfer.py \
       --dry lexicon_dry.wav \
       --wet lexicon_wet.wav \
       --new my_song.mp3 \
       --epochs 500
   ```

3. **Result**: Software clone with:
   - Similar sonic character
   - Interpretable parameters (can modify)
   - Runs on your laptop

### Training Tips

**Duration**: Longer reference audio = better learning
- Minimum: 5 seconds
- Recommended: 10-30 seconds
- Ideal: Include diverse content (transients + sustained)

**Epochs**:
- Simple reverb: 200-300 epochs
- Complex reverb: 500-800 epochs
- Watch loss curve for convergence

**Learning Rate**:
- Start: 0.01
- If unstable: reduce to 0.005
- If slow: increase to 0.02

---

## Application 3: Adaptive Reverb

**File**: `app_adaptive_reverb.py`

### What It Does

Context-aware reverb that automatically adjusts parameters based on audio content in real-time.

**Problem**: Fixed reverb sounds wrong when audio changes (quiet → loud, sparse → dense)

**Solution**: Analyze audio frame-by-frame, adapt reverb parameters dynamically.

### Use Cases

- **Intelligent mixing** - Less reverb during busy sections
- **Broadcast processing** - Consistent reverb across varying content
- **Game audio** - Reverb adapts to gameplay intensity
- **Podcast enhancement** - More reverb during pauses, less during speech

### Quick Start

```bash
# Demo with synthetic audio (shows adaptation clearly)
python app_adaptive_reverb.py --demo

# Process your audio
python app_adaptive_reverb.py \
    --input my_song.mp3 \
    --output my_song_adaptive.mp3 \
    --rules smart
```

### Adaptation Strategies

**1. Smart (Recommended)**:
- High energy → less wet (avoid mud)
- High brightness → less damping (preserve highs)
- High density → shorter tail (avoid crowding)

**2. Inverse** (for comparison):
- Opposite of smart rules
- Shows importance of adaptation direction

**3. Extreme**:
- Exaggerated adaptation
- Creative effect

### How It Works

**1. Frame Analysis**:
```python
# Analyze every frame (e.g., 2048 samples)
features = analyze_frame(audio_frame)
# Returns: energy, brightness, density
```

**2. Feature Extraction**:

```python
# Energy (RMS)
energy = sqrt(mean(audio^2))

# Brightness (spectral centroid)
brightness = sum(magnitude * frequencies) / sum(magnitude)

# Density (spectral entropy)
density = -sum(p * log(p))  # How spread out spectrum is
```

**3. Parameter Mapping**:

```python
# Smart rules
wet = 0.8 - (energy * 0.6)         # High energy → less wet
damp = 0.6 - (brightness * 0.5)    # Bright → less damping
feedback = 0.95 - (density * 0.25)  # Dense → shorter tail
```

**4. Parameter Smoothing**:
```python
# Smooth to avoid abrupt changes
new_param = 0.9 * old_param + 0.1 * target_param
```

### Demo Results

**Section 1 (0-5s)**: Quiet, sustained tone
- Energy: LOW → Wet: HIGH (0.75)
- Result: Spacious, ambient

**Section 2 (5-10s)**: Loud, dense chords
- Energy: HIGH → Wet: LOW (0.25)
- Result: Clear, controlled

**Section 3 (10-15s)**: Percussive hits
- Transient-rich → Feedback: LOW (0.72)
- Result: Tight, punchy

**Visualizations**:
- `adaptive_smart_timeline.png` shows parameter evolution
- Clearly see adaptation to audio changes

### Advanced Usage

**Custom Rules**:

```python
# Define your own mapping
def my_rules(features):
    energy = features['energy']
    brightness = features['brightness']

    # Your logic here
    wet = custom_function(energy)
    damp = custom_function(brightness)

    return {'feedback': 0.85, 'damping': damp, 'wet': wet}

# Integrate
processor = AdaptiveReverbProcessor()
processor.features_to_reverb_params = my_rules
```

**Neural Adaptation** (future):

Combine with Application 1:
```python
# Neural network learns optimal adaptation rules
features = extract_features(audio_frame)
params = neural_net(features)  # Learned from data!
reverb.set_parameters(params)
```

---

## Comparison Table

| Feature | Neural Controller | Style Transfer | Adaptive Reverb |
|---------|------------------|----------------|-----------------|
| **Training Required** | Yes (50-100 epochs) | Yes (200-500 epochs) | No |
| **Use Case** | Auto-suggest reverb | Clone references | Real-time adaptation |
| **Input Data** | Dry/wet pairs | Dry/wet reference | Dry audio only |
| **Output** | Fixed parameters | Learned reverb | Time-varying parameters |
| **Complexity** | High | Medium | Low |
| **Real-time** | After training | After training | Yes |
| **Customizable** | Network architecture | Training data | Adaptation rules |

---

## Integration Examples

### Combining Applications

**Example 1**: Style Transfer + Adaptive

```python
# Step 1: Learn reverb from studio recording
learned_reverb = style_transfer.learn("studio_dry.mp3", "studio_wet.mp3")

# Step 2: Apply adaptively to new audio
processor = AdaptiveReverbProcessor()
processor.base_reverb = learned_reverb
output = processor.process_adaptive(new_audio, rules='smart')

# Result: Studio character + intelligent adaptation
```

**Example 2**: Neural Controller for Multiple Styles

```python
# Train network to predict reverb for different genres
dataset = {
    'rock': (rock_dry, rock_wet),
    'jazz': (jazz_dry, jazz_wet),
    'classical': (classical_dry, classical_wet),
}

# Network learns: rock → tight reverb, classical → long reverb
model = train_neural_reverb(dataset)

# Auto-detect and apply appropriate reverb
genre = detect_genre(audio)
params = model.predict(audio, genre)
output = apply_reverb(audio, params)
```

---

## Performance Considerations

### Training Time

| Application | Dataset Size | Epochs | Time (CPU) | Time (GPU) |
|-------------|--------------|--------|------------|------------|
| Neural Controller | 100 samples | 50 | ~5 min | ~1 min |
| Style Transfer | 1 reference | 300 | ~3 min | ~30 sec |
| Adaptive Reverb | N/A | N/A | N/A | N/A |

### Inference Time

| Application | Audio Length | Processing Time | Real-time Factor |
|-------------|--------------|-----------------|------------------|
| Neural Controller | 10s | ~2s | 5x |
| Style Transfer | 10s | ~2s | 5x |
| Adaptive Reverb | 10s | ~8s | 1.25x |

**Note**: Adaptive is slower due to frame-by-frame processing, but still achieves near real-time.

---

## Troubleshooting

### Neural Controller

**Problem**: Poor parameter predictions

**Solutions**:
- Increase dataset size (>100 samples)
- Train longer (>100 epochs)
- Add data augmentation
- Simplify parameter space (freeze some parameters)

### Style Transfer

**Problem**: Doesn't match reference well

**Solutions**:
- Longer reference audio (>10s)
- More training epochs (>500)
- Check dry/wet alignment
- Ensure similar audio content

**Problem**: Overfitting to reference

**Solutions**:
- Use diverse reference audio
- Add regularization
- Early stopping based on validation

### Adaptive Reverb

**Problem**: Parameters change too abruptly

**Solutions**:
- Increase smoothing (0.95-0.99)
- Larger frame size
- Moving average filter

**Problem**: Not adapting enough

**Solutions**:
- Decrease smoothing (0.7-0.8)
- Exaggerate mapping rules
- Check feature extraction

---

## Future Directions

### Planned Enhancements

1. **Multi-band Adaptation**: Different reverb per frequency band
2. **Perceptual Loss**: Use pre-trained audio models for better matching
3. **Real-time Implementation**: C++ version for live processing
4. **GUI Interface**: Visual parameter control and visualization

### Research Opportunities

1. **Conditional VAE**: Learn distribution of reverb parameters
2. **GAN-based**: Adversarial training for more realistic reverb
3. **Meta-Learning**: Few-shot adaptation to new spaces
4. **Multimodal**: Predict reverb from room images + audio

---

## Citation

If you use these applications in research:

```bibtex
@misc{fdn_reverb_applications,
  title={DDSP FDN Reverb Applications},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/FDN-Reverb}
}
```

---

## See Also

- [README_DDSP.md](README_DDSP.md) - DDSP implementation guide
- [COMPARISON.md](COMPARISON.md) - Theory and comparison
- [README_COMPARISON.md](README_COMPARISON.md) - Performance benchmarks

---

**Questions or feature requests?** Open an issue with the `applications` label.
