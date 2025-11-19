# FDN Reverb Comparison Tools

This repository includes comprehensive comparison tools to evaluate different FDN implementations across multiple dimensions: performance, audio quality, and training efficiency.

## Quick Start

### Run Complete Comparison

```bash
# Full comparison suite (recommended first run)
python run_comparison.py --input dry.mp3 --full

# Quick comparison (faster, fewer tests)
python run_comparison.py --input dry.mp3 --quick

# With custom input
python run_comparison.py --input your_audio.mp3 --output_dir my_results
```

This will generate:
- Performance benchmarks (speed, throughput, memory)
- Audio quality metrics (spectral, temporal, RT60)
- Visualizations (waveforms, spectrograms, frequency response)
- Comprehensive report with key insights

### Output Files

After running `python run_comparison.py --full`, you'll find in `comparison_results/`:

**Performance:**
- `performance_results.json` - Detailed performance metrics
- `performance_comparison.png` - Bar charts comparing speed
- `scalability.png` - Performance vs audio length

**Quality:**
- `quality_results.json` - Audio quality metrics
- `output_traditional.mp3` - Traditional FDN output
- `output_ddsp.mp3` - DDSP FDN output

**Visualizations:**
- `waveforms.png` - Time-domain comparison
- `spectrograms.png` - Frequency-time analysis
- `frequency_response.png` - Magnitude spectrum

## Individual Comparison Scripts

### 1. Performance Benchmark (`compare_performance.py`)

Compares processing speed, throughput, and memory usage.

```bash
# Basic benchmark
python compare_performance.py --duration 10 --runs 5

# Quick test
python compare_performance.py --duration 5 --runs 3 --no_scalability

# Save results
python compare_performance.py --save performance_results.json
```

**Metrics Measured:**
- **Processing time**: How long to process audio
- **Throughput**: Real-time factor (>1.0 = faster than real-time)
- **Memory usage**: RAM consumption
- **Scalability**: Performance with different audio lengths

**Example Output:**
```
SUMMARY COMPARISON
================================================================

Processing Time:
  Traditional FDN: 0.1234s
  DDSP FDN:        0.4567s
  Speedup:         3.7x (Traditional faster)

Throughput (real-time factor):
  Traditional FDN: 81.0x
  DDSP FDN:        21.9x

Memory Usage:
  Traditional FDN: 12.5 MB
  DDSP FDN:        45.2 MB
```

### 2. Audio Quality Metrics (`compare_quality.py`)

Compares sonic characteristics and audio quality.

```bash
# Quality comparison
python compare_quality.py --input dry.mp3

# Save outputs for listening test
python compare_quality.py --input dry.mp3 --save_outputs
```

**Metrics Measured:**
- **Spectral centroid**: Brightness measure (Hz)
- **Spectral rolloff**: High-frequency content threshold
- **Spectral flatness**: Tonality vs noise-like (0-1)
- **RT60**: Decay time (full, low, mid, high bands)
- **RMS level**: Average energy
- **Peak level**: Maximum amplitude
- **Spectral distance**: L2 norm of magnitude spectra
- **Temporal distance**: Envelope difference

**Example Output:**
```
QUALITY METRICS COMPARISON
================================================================

Metric                         Traditional FDN      DDSP FDN            Diff
----------------------------------------------------------------------------------
Brightness                            3245.2Hz          3198.4Hz         1.4%
High-freq content                     8912.5Hz          8876.3Hz         0.4%
Tonality vs Noise                         0.23              0.24         4.3%
Decay time (full band)                   2.45s             2.41s         1.6%
Decay time (low freq)                    2.89s             2.85s         1.4%
Decay time (mid freq)                    2.32s             2.28s         1.7%
Decay time (high freq)                   1.78s             1.75s         1.7%
RMS level                                 0.12              0.12         0.8%
Peak level                                0.89              0.88         1.1%

Distance Metrics
------------------------------------------------------------
Spectral distance                      342.5
Temporal distance                        0.08

Interpretation:
  ✓ Very similar spectral characteristics
  ✓ Very similar temporal envelopes
```

### 3. Visualizations (`visualize_comparison.py`)

Creates visual comparisons of audio signals.

```bash
# Generate all visualizations
python visualize_comparison.py --input dry.mp3 --duration 10

# Custom output directory
python visualize_comparison.py --input dry.mp3 --output_dir my_plots
```

**Visualizations Created:**
1. **Waveforms**: Time-domain representation
   - Shows amplitude over time
   - First 5 seconds for clarity

2. **Spectrograms**: Time-frequency analysis
   - Color-coded magnitude (dB)
   - Frequency range: 0-10kHz
   - Reveals temporal and spectral evolution

3. **Frequency Response**: Magnitude spectrum
   - Smoothed for readability
   - Log-scale frequency axis
   - Overlaid comparison

## Understanding the Results

### Performance Metrics

**Real-time Factor**:
- `> 1.0`: Faster than real-time (can process live audio)
- `= 1.0`: Exactly real-time
- `< 1.0`: Slower than real-time (offline only)

**Example**: 81.0x real-time factor means it can process 81 seconds of audio in 1 second.

**Speedup**:
- How many times faster one implementation is than another
- Example: 3.7x means Traditional is 3.7 times faster than DDSP

### Quality Metrics

**Spectral Centroid** (Brightness):
- Higher = brighter sound
- Typical range: 1000-5000 Hz for reverb
- Difference <5% = very similar brightness

**RT60** (Decay Time):
- Time for sound to decay by 60 dB
- Longer = more spacious sound
- Cathedral: 2-10s, Room: 0.3-1.5s
- Difference <10% = similar reverb tail

**Spectral Distance**:
- L2 norm of magnitude spectra
- `< 1000`: Very similar
- `1000-5000`: Moderately similar
- `> 5000`: Different

**Temporal Distance**:
- Envelope difference
- `< 0.1`: Very similar dynamics
- `0.1-1.0`: Moderately similar
- `> 1.0`: Different dynamics

## Comparison Scenarios

### Scenario 1: Implementation Equivalence

**Question**: Do Traditional and DDSP FDN produce the same sound?

```bash
python compare_quality.py --input dry.mp3 --save_outputs
```

Listen to:
- `output_traditional.mp3`
- `output_ddsp.mp3`

Check metrics:
- Spectral distance (should be < 1000 for equivalence)
- RT60 difference (should be < 5%)

**Expected**: Very similar, minor numerical differences

### Scenario 2: Performance Trade-offs

**Question**: How much faster is Traditional FDN?

```bash
python compare_performance.py --duration 30 --runs 10
```

Check:
- Processing time ratio
- Real-time factors
- Scalability curves

**Expected**: Traditional 3-5x faster due to C++ optimizations

### Scenario 3: Different Parameters

**Question**: How do implementations compare with different settings?

Modify `compare_quality.py` to test different configs:
```python
config = {
    'delays_ms': (31, 41, 47, 59, 67, 73, 83, 97),  # Different delays
    'feedback_gain': 0.95,  # Longer tail
    'damp': 0.4,            # More damping
    'wet': 0.8,             # Wetter mix
}
```

## Advanced Usage

### Custom Metrics

Add your own metrics to `compare_quality.py`:

```python
def my_custom_metric(self, audio):
    # Your metric computation
    return metric_value

# In compute_all_metrics():
metrics['my_metric'] = self.my_custom_metric(audio)
```

### Batch Comparison

Compare multiple audio files:

```bash
for file in *.mp3; do
    python run_comparison.py --input "$file" --output_dir "results_${file%.mp3}"
done
```

### Export for Analysis

Results are saved as JSON for further analysis:

```python
import json

# Load results
with open('comparison_results/performance_results.json') as f:
    perf = json.load(f)

with open('comparison_results/quality_results.json') as f:
    quality = json.load(f)

# Analyze
print(f"Speedup: {perf['ddsp_fdn']['avg_time'] / perf['traditional_fdn']['avg_time']:.2f}x")
print(f"RT60 diff: {abs(quality['signal1']['rt60_full'] - quality['signal2']['rt60_full']):.3f}s")
```

## Integration with dafx25-ddsp-tutorial

To compare with the FLAMO-based implementation from dafx25-ddsp-tutorial:

1. Clone and set up dafx25-ddsp-tutorial repository
2. Install FLAMO: `pip install flamo`
3. Extend `compare_performance.py` to include FLAMO benchmark
4. Run comparison

Example integration:

```python
# In compare_performance.py

def benchmark_flamo_fdn(self, audio, num_runs=3):
    """Benchmark FLAMO-based FDN (requires dafx25-ddsp-tutorial setup)."""
    try:
        from flamo.processor import dsp
        # Setup FLAMO FDN
        # Run benchmark
        # Return results
    except ImportError:
        print("FLAMO not available, skipping")
        return None
```

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce test duration
```bash
python run_comparison.py --test_duration 5 --quick
```

### Issue: Slow Performance Tests

**Solution**: Reduce number of runs
```bash
python compare_performance.py --runs 3 --no_scalability
```

### Issue: Different Results Each Run

**Solution**: Increase number of runs for averaging
```bash
python compare_performance.py --runs 10
```

### Issue: CUDA Out of Memory (DDSP)

**Solution**: DDSP will automatically fall back to CPU if CUDA fails. If issues persist, force CPU:
```python
# In compare scripts, change:
device = torch.device('cpu')  # Force CPU
```

## Interpreting Results for Research

### For Papers/Reports

Key metrics to report:

1. **Performance**:
   - Mean processing time ± std
   - Real-time factor
   - Speedup ratio

2. **Quality**:
   - Spectral distance
   - RT60 comparison
   - Listening test results

3. **Visualization**:
   - Include spectrogram comparison
   - Frequency response overlay

### Statistical Significance

With 10+ runs, you can compute:
```python
import scipy.stats as stats

# T-test for processing times
t_stat, p_value = stats.ttest_ind(trad_times, ddsp_times)
print(f"p-value: {p_value}")

if p_value < 0.05:
    print("Difference is statistically significant")
```

## Citation

If you use these comparison tools in your research:

```bibtex
@misc{fdn_reverb_comparison,
  title={FDN Reverb Comparison Tools},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/FDN-Reverb}
}
```

## See Also

- [COMPARISON.md](COMPARISON.md) - Conceptual comparison with dafx25-ddsp-tutorial
- [README_DDSP.md](README_DDSP.md) - DDSP implementation guide
- [README.md](README.md) - Main documentation

---

**Questions or issues?** Open an issue on GitHub with the `comparison` label.
