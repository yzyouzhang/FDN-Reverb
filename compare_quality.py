"""
Audio Quality Comparison Script

Compares audio quality metrics between different FDN implementations:
1. Spectral characteristics
2. Temporal characteristics
3. Perceptual metrics
4. RT60 decay time
5. Frequency response

This helps understand if implementations produce similar or different sonic results.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from io_utils import load_audio, save_audio
from reverb_util import FDNReverb
from ddsp_reverb import LearnableFDNReverb


class AudioQualityMetrics:
    """Compute various audio quality metrics."""

    def __init__(self, sr=48000):
        self.sr = sr

    def spectral_centroid(self, audio):
        """Compute spectral centroid (brightness measure)."""
        # STFT
        stft = torch.stft(
            audio,
            n_fft=2048,
            hop_length=512,
            window=torch.hann_window(2048),
            return_complex=True
        )
        magnitude = torch.abs(stft)

        # Frequency bins
        freqs = torch.fft.rfftfreq(2048, 1/self.sr)

        # Weighted mean
        centroid = (magnitude * freqs.unsqueeze(1)).sum(dim=0) / (magnitude.sum(dim=0) + 1e-8)
        return centroid.mean().item()

    def spectral_rolloff(self, audio, percentile=0.85):
        """Frequency below which percentile of energy is contained."""
        stft = torch.stft(
            audio,
            n_fft=2048,
            hop_length=512,
            window=torch.hann_window(2048),
            return_complex=True
        )
        magnitude = torch.abs(stft) ** 2

        cumsum = torch.cumsum(magnitude, dim=0)
        total = cumsum[-1]
        threshold = percentile * total

        rolloff_bins = (cumsum < threshold).sum(dim=0)
        freqs = torch.fft.rfftfreq(2048, 1/self.sr)
        rolloff_freq = freqs[rolloff_bins.long().clamp(0, len(freqs)-1)].mean()

        return rolloff_freq.item()

    def spectral_flatness(self, audio):
        """Measure of noisiness vs tonality (0=tonal, 1=noise-like)."""
        stft = torch.stft(
            audio,
            n_fft=2048,
            hop_length=512,
            window=torch.hann_window(2048),
            return_complex=True
        )
        magnitude = torch.abs(stft) + 1e-10

        # Geometric mean / Arithmetic mean
        geo_mean = torch.exp(torch.log(magnitude).mean(dim=0))
        arith_mean = magnitude.mean(dim=0)
        flatness = (geo_mean / (arith_mean + 1e-10)).mean()

        return flatness.item()

    def estimate_rt60(self, audio, band='full'):
        """
        Estimate RT60 decay time (time for 60dB decay).

        Args:
            audio: Audio signal
            band: 'full' or frequency band ('low', 'mid', 'high')

        Returns:
            RT60 in seconds
        """
        # Band-pass filter if needed
        if band == 'low':
            # Simple lowpass at 500 Hz
            audio = self.lowpass_filter(audio, cutoff=500)
        elif band == 'mid':
            audio = self.bandpass_filter(audio, low=500, high=2000)
        elif band == 'high':
            audio = self.highpass_filter(audio, cutoff=2000)

        # Compute energy envelope
        energy = audio ** 2

        # Smooth with moving average
        window_size = int(0.01 * self.sr)  # 10ms windows
        kernel = torch.ones(window_size) / window_size
        energy_smooth = F.conv1d(
            energy.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=window_size//2
        ).squeeze()

        # Convert to dB
        energy_db = 10 * torch.log10(energy_smooth + 1e-10)

        # Find decay region (after peak)
        peak_idx = energy_db.argmax()
        decay_curve = energy_db[peak_idx:]

        if len(decay_curve) < 100:
            return 0.0

        # Find time to drop 60 dB (or estimate from 20dB drop)
        initial_db = decay_curve[0]
        target_db = initial_db - 60

        # Find where it crosses target (or extrapolate)
        below_target = decay_curve < target_db
        if below_target.any():
            rt60_samples = below_target.nonzero()[0].item()
        else:
            # Extrapolate from 20dB decay
            target_20db = initial_db - 20
            below_20 = decay_curve < target_20db
            if below_20.any():
                t20_samples = below_20.nonzero()[0].item()
                rt60_samples = t20_samples * 3  # RT60 ≈ 3 * T20
            else:
                return 0.0

        rt60_time = rt60_samples / self.sr
        return rt60_time

    def lowpass_filter(self, audio, cutoff=500):
        """Simple lowpass filter."""
        # FFT-based filtering
        fft = torch.fft.rfft(audio)
        freqs = torch.fft.rfftfreq(len(audio), 1/self.sr)
        fft[freqs > cutoff] = 0
        return torch.fft.irfft(fft, n=len(audio))

    def highpass_filter(self, audio, cutoff=2000):
        """Simple highpass filter."""
        fft = torch.fft.rfft(audio)
        freqs = torch.fft.rfftfreq(len(audio), 1/self.sr)
        fft[freqs < cutoff] = 0
        return torch.fft.irfft(fft, n=len(audio))

    def bandpass_filter(self, audio, low=500, high=2000):
        """Simple bandpass filter."""
        fft = torch.fft.rfft(audio)
        freqs = torch.fft.rfftfreq(len(audio), 1/self.sr)
        fft[(freqs < low) | (freqs > high)] = 0
        return torch.fft.irfft(fft, n=len(audio))

    def signal_to_noise_ratio(self, signal, noise):
        """Compute SNR between signal and noise."""
        signal_power = (signal ** 2).mean()
        noise_power = (noise ** 2).mean()
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
        return snr.item()

    def spectral_distance(self, audio1, audio2):
        """Compute spectral distance (L2 norm of magnitude spectra)."""
        stft1 = torch.stft(audio1, n_fft=2048, hop_length=512,
                          window=torch.hann_window(2048), return_complex=True)
        stft2 = torch.stft(audio2, n_fft=2048, hop_length=512,
                          window=torch.hann_window(2048), return_complex=True)

        mag1 = torch.abs(stft1)
        mag2 = torch.abs(stft2)

        distance = torch.norm(mag1 - mag2).item()
        return distance

    def temporal_envelope_distance(self, audio1, audio2):
        """Compute distance between temporal envelopes."""
        # Compute envelopes
        window_size = int(0.01 * self.sr)
        kernel = torch.ones(window_size) / window_size

        env1 = F.conv1d(
            (audio1 ** 2).unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=window_size//2
        ).squeeze()

        env2 = F.conv1d(
            (audio2 ** 2).unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=window_size//2
        ).squeeze()

        # L2 distance
        min_len = min(len(env1), len(env2))
        distance = torch.norm(env1[:min_len] - env2[:min_len]).item()
        return distance

    def compute_all_metrics(self, audio, label="Audio"):
        """Compute all metrics for a single audio signal."""
        metrics = {
            'label': label,
            'spectral_centroid': self.spectral_centroid(audio),
            'spectral_rolloff': self.spectral_rolloff(audio),
            'spectral_flatness': self.spectral_flatness(audio),
            'rt60_full': self.estimate_rt60(audio, 'full'),
            'rt60_low': self.estimate_rt60(audio, 'low'),
            'rt60_mid': self.estimate_rt60(audio, 'mid'),
            'rt60_high': self.estimate_rt60(audio, 'high'),
            'rms': audio.square().mean().sqrt().item(),
            'peak': audio.abs().max().item(),
        }
        return metrics

    def compare_two_signals(self, audio1, audio2, label1="Signal 1", label2="Signal 2"):
        """Compare two audio signals."""
        # Individual metrics
        metrics1 = self.compute_all_metrics(audio1, label1)
        metrics2 = self.compute_all_metrics(audio2, label2)

        # Comparison metrics
        min_len = min(len(audio1), len(audio2))
        spectral_dist = self.spectral_distance(audio1[:min_len], audio2[:min_len])
        temporal_dist = self.temporal_envelope_distance(audio1[:min_len], audio2[:min_len])

        comparison = {
            'signal1': metrics1,
            'signal2': metrics2,
            'spectral_distance': spectral_dist,
            'temporal_distance': temporal_dist,
        }

        return comparison


class ReverbQualityComparison:
    """Compare quality of different reverb implementations."""

    def __init__(self, sr=48000):
        self.sr = sr
        self.metrics = AudioQualityMetrics(sr=sr)

    def compare_implementations(self, dry_audio, config=None):
        """
        Compare different FDN implementations on the same dry audio.

        Args:
            dry_audio: Input audio signal
            config: Reverb configuration (delays, feedback, etc.)

        Returns:
            Dictionary with comparison results
        """
        if config is None:
            config = {
                'delays_ms': (29, 37, 43, 53, 61, 71, 79, 89),
                'feedback_gain': 0.9,
                'damp': 0.25,
                'wet': 0.9,
            }

        print(f"\n{'='*60}")
        print(f"Comparing Reverb Implementations")
        print(f"{'='*60}")
        print(f"Configuration:")
        for key, val in config.items():
            print(f"  {key}: {val}")

        # Traditional FDN
        print(f"\nProcessing with Traditional FDN...")
        trad_reverb = FDNReverb(
            sr=self.sr,
            delays_ms=config['delays_ms'],
            feedback_gain=config['feedback_gain'],
            damp=config['damp'],
            wet=config['wet'],
        )
        trad_output = trad_reverb.process(dry_audio.clone())

        # DDSP FDN
        print(f"Processing with DDSP FDN...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ddsp_reverb = LearnableFDNReverb(
            sr=self.sr,
            init_delays_ms=config['delays_ms'],
            init_feedback_gain=config['feedback_gain'],
            init_damp=config['damp'],
            init_wet=config['wet'],
            learnable_delays=False,
            learnable_feedback=False,
            learnable_damping=False,
            learnable_wet=False,
            learnable_modulation=False,
        ).to(device)
        ddsp_reverb.eval()
        with torch.no_grad():
            ddsp_output = ddsp_reverb(dry_audio.to(device)).cpu()

        # Compute metrics
        print(f"\nComputing quality metrics...")
        comparison = self.metrics.compare_two_signals(
            trad_output,
            ddsp_output,
            label1="Traditional FDN",
            label2="DDSP FDN"
        )

        # Print results
        self.print_comparison(comparison)

        return {
            'comparison': comparison,
            'traditional_output': trad_output,
            'ddsp_output': ddsp_output,
        }

    def print_comparison(self, comparison):
        """Print comparison results."""
        print(f"\n{'='*60}")
        print(f"QUALITY METRICS COMPARISON")
        print(f"{'='*60}")

        s1 = comparison['signal1']
        s2 = comparison['signal2']

        metrics_to_compare = [
            ('spectral_centroid', 'Hz', 'Brightness'),
            ('spectral_rolloff', 'Hz', 'High-freq content'),
            ('spectral_flatness', '', 'Tonality (0) vs Noise (1)'),
            ('rt60_full', 's', 'Decay time (full band)'),
            ('rt60_low', 's', 'Decay time (low freq)'),
            ('rt60_mid', 's', 'Decay time (mid freq)'),
            ('rt60_high', 's', 'Decay time (high freq)'),
            ('rms', '', 'RMS level'),
            ('peak', '', 'Peak level'),
        ]

        print(f"\n{'Metric':<30} {s1['label']:<20} {s2['label']:<20} {'Diff':<15}")
        print(f"{'-'*90}")

        for metric, unit, description in metrics_to_compare:
            val1 = s1[metric]
            val2 = s2[metric]
            diff = abs(val1 - val2)
            rel_diff = (diff / (abs(val1) + 1e-10)) * 100

            print(f"{description:<30} {val1:>15.2f}{unit:<5} {val2:>15.2f}{unit:<5} "
                  f"{rel_diff:>10.1f}%")

        print(f"\n{'Distance Metrics':<30}")
        print(f"{'-'*60}")
        print(f"{'Spectral distance':<30} {comparison['spectral_distance']:>15.4f}")
        print(f"{'Temporal distance':<30} {comparison['temporal_distance']:>15.4f}")

        # Interpretation
        print(f"\nInterpretation:")
        if comparison['spectral_distance'] < 1000:
            print(f"  ✓ Very similar spectral characteristics")
        elif comparison['spectral_distance'] < 5000:
            print(f"  → Moderately similar spectral characteristics")
        else:
            print(f"  ✗ Different spectral characteristics")

        if comparison['temporal_distance'] < 0.1:
            print(f"  ✓ Very similar temporal envelopes")
        elif comparison['temporal_distance'] < 1.0:
            print(f"  → Moderately similar temporal envelopes")
        else:
            print(f"  ✗ Different temporal envelopes")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Audio Quality Comparison')
    parser.add_argument('--input', '-i', type=str, default='dry.mp3',
                       help='Input dry audio file')
    parser.add_argument('--sr', type=int, default=48000,
                       help='Sample rate')
    parser.add_argument('--save_outputs', action='store_true',
                       help='Save output audio files for comparison')

    args = parser.parse_args()

    # Load audio
    print(f"Loading audio: {args.input}")
    dry_audio, sr = load_audio(args.input, mono=True, target_sample_rate=args.sr)

    # Truncate to manageable length (30 seconds max)
    max_samples = 30 * sr
    if len(dry_audio) > max_samples:
        dry_audio = dry_audio[:max_samples]
        print(f"Truncated to 30 seconds")

    # Run comparison
    comparator = ReverbQualityComparison(sr=sr)
    results = comparator.compare_implementations(dry_audio)

    # Save outputs if requested
    if args.save_outputs:
        save_audio(results['traditional_output'].unsqueeze(0), sr, 'output_traditional.mp3')
        save_audio(results['ddsp_output'].unsqueeze(0), sr, 'output_ddsp.mp3')
        print(f"\nOutputs saved:")
        print(f"  - output_traditional.mp3")
        print(f"  - output_ddsp.mp3")


if __name__ == '__main__':
    main()
