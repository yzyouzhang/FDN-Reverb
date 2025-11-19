"""
Adaptive Reverb - Application Demo 3

Context-aware reverb that adapts parameters based on audio content.

Use cases:
- Automatically adjust reverb for vocals vs instruments
- Reduce reverb during busy sections, increase during sparse sections
- Match reverb to musical intensity/dynamics
- Content-adaptive mixing assistant

The reverb parameters change over time based on:
- Spectral content (brightness)
- Energy/dynamics (RMS level)
- Density (how busy the audio is)

This demonstrates combining traditional DSP analysis with DDSP for
intelligent, adaptive audio processing.
"""

import torch
import torch.nn.functional as F
import numpy as np
from io_utils import load_audio, save_audio
from ddsp_reverb import LearnableFDNReverb
import matplotlib.pyplot as plt


class AdaptiveReverbProcessor:
    """
    Adaptive reverb that adjusts parameters based on audio content.

    Analyzes audio in frames and adjusts reverb parameters dynamically:
    - High energy → less wet, tighter reverb
    - Low energy → more wet, longer reverb
    - High brightness → less damping
    - Low brightness → more damping
    """

    def __init__(self, sr=48000, frame_size=2048, hop_size=512):
        self.sr = sr
        self.frame_size = frame_size
        self.hop_size = hop_size

    def analyze_frame(self, frame):
        """
        Analyze audio frame to extract content features.

        Returns:
            features: dict with energy, brightness, density
        """
        # Energy (RMS)
        energy = torch.sqrt(torch.mean(frame ** 2))

        # Spectral centroid (brightness)
        fft = torch.fft.rfft(frame)
        magnitude = torch.abs(fft)
        freqs = torch.fft.rfftfreq(len(frame), 1/self.sr)

        # Weighted mean frequency
        centroid = torch.sum(magnitude * freqs) / (torch.sum(magnitude) + 1e-8)

        # Spectral density (how spread out the spectrum is)
        normalized_mag = magnitude / (torch.sum(magnitude) + 1e-8)
        spectral_entropy = -torch.sum(normalized_mag * torch.log(normalized_mag + 1e-10))
        density = spectral_entropy / np.log(len(magnitude))  # Normalize

        return {
            'energy': energy.item(),
            'brightness': centroid.item(),
            'density': density.item(),
        }

    def features_to_reverb_params(self, features, rules='smart'):
        """
        Map audio features to reverb parameters using rules.

        Args:
            features: dict from analyze_frame
            rules: 'smart', 'inverse', or 'extreme'

        Returns:
            params: dict with feedback, damping, wet
        """
        energy = features['energy']
        brightness = features['brightness']
        density = features['density']

        if rules == 'smart':
            # Smart mixing rules:
            # - High energy → less wet (avoid mud)
            # - High brightness → less damping (preserve highs)
            # - High density → shorter reverb (avoid crowding)

            # Map energy (0-1) to wet (0.2-0.8)
            # Lower energy = more reverb
            wet = 0.8 - (energy * 0.6)
            wet = np.clip(wet, 0.2, 0.8)

            # Map brightness (0-10000 Hz) to damping (0.1-0.6)
            # Higher brightness = less damping
            damp = 0.6 - (brightness / 10000 * 0.5)
            damp = np.clip(damp, 0.1, 0.6)

            # Map density to feedback (0.7-0.95)
            # Higher density = shorter tail
            feedback = 0.95 - (density * 0.25)
            feedback = np.clip(feedback, 0.7, 0.95)

        elif rules == 'inverse':
            # Opposite behavior (for comparison)
            wet = 0.2 + (energy * 0.6)
            damp = 0.1 + (brightness / 10000 * 0.5)
            feedback = 0.7 + (density * 0.25)

        else:  # extreme
            # Exaggerated adaptation
            wet = 0.9 - (energy * 0.8)
            wet = np.clip(wet, 0.1, 0.9)
            damp = 0.7 - (brightness / 10000 * 0.6)
            damp = np.clip(damp, 0.05, 0.7)
            feedback = 0.95 - (density * 0.35)
            feedback = np.clip(feedback, 0.6, 0.95)

        return {
            'feedback': feedback,
            'damping': damp,
            'wet': wet,
        }

    def process_adaptive(self, audio, rules='smart', smoothing=0.9):
        """
        Process audio with adaptive reverb.

        Args:
            audio: Input audio tensor
            rules: Parameter mapping strategy
            smoothing: Parameter smoothing factor (0-1, higher = smoother)

        Returns:
            output: Processed audio
            param_timeline: List of parameters over time
        """
        print(f"\nProcessing with adaptive reverb...")
        print(f"  Rules: {rules}")
        print(f"  Smoothing: {smoothing}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Analyze audio in frames
        num_frames = (len(audio) - self.frame_size) // self.hop_size + 1

        param_timeline = []
        prev_params = None

        print(f"  Analyzing {num_frames} frames...")

        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.frame_size

            frame = audio[start:end]

            # Analyze
            features = self.analyze_frame(frame)

            # Map to parameters
            params = self.features_to_reverb_params(features, rules)

            # Smooth parameters
            if prev_params is not None:
                for key in params:
                    params[key] = (smoothing * prev_params[key] +
                                 (1 - smoothing) * params[key])

            param_timeline.append({
                'time': start / self.sr,
                'features': features,
                'params': params.copy(),
            })

            prev_params = params

        # Process audio with time-varying parameters
        print(f"  Applying adaptive reverb...")

        output = torch.zeros_like(audio)
        chunk_size = self.sr  # Process 1 second chunks

        for chunk_start in range(0, len(audio), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(audio))
            chunk = audio[chunk_start:chunk_end]

            # Get parameters for this time
            chunk_time = chunk_start / self.sr
            param_idx = int(chunk_time / (self.hop_size / self.sr))
            param_idx = min(param_idx, len(param_timeline) - 1)

            params = param_timeline[param_idx]['params']

            # Create reverb with these parameters
            reverb = LearnableFDNReverb(
                sr=self.sr,
                init_feedback_gain=params['feedback'],
                init_damping=params['damping'],
                init_wet=params['wet'],
                learnable_delays=False,
                learnable_feedback=False,
                learnable_damping=False,
                learnable_wet=False,
            ).to(device)

            reverb.eval()
            with torch.no_grad():
                chunk_output = reverb(chunk.to(device)).cpu()

            output[chunk_start:chunk_end] = chunk_output

            # Progress
            if (chunk_start // chunk_size) % 5 == 0:
                progress = chunk_start / len(audio) * 100
                print(f"    {progress:.0f}%... ", end='')

        print(f"100% Done!")

        return output, param_timeline

    def plot_adaptation(self, param_timeline, save_path=None):
        """Plot how parameters adapt over time."""
        times = [p['time'] for p in param_timeline]
        feedbacks = [p['params']['feedback'] for p in param_timeline]
        dampings = [p['params']['damping'] for p in param_timeline]
        wets = [p['params']['wet'] for p in param_timeline]
        energies = [p['features']['energy'] for p in param_timeline]

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Reverb parameters
        ax1 = axes[0]
        ax1.plot(times, feedbacks, label='Feedback', linewidth=2)
        ax1.plot(times, dampings, label='Damping', linewidth=2)
        ax1.plot(times, wets, label='Wet Mix', linewidth=2)
        ax1.set_ylabel('Parameter Value')
        ax1.set_title('Adaptive Reverb Parameters Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])

        # Energy (to show adaptation basis)
        ax2 = axes[1]
        ax2.plot(times, energies, label='Energy (RMS)', linewidth=2, color='red')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Energy')
        ax2.set_title('Audio Energy (Adaptation Driver)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Adaptation plot saved: {save_path}")
        else:
            plt.show()

        plt.close()


def demo_adaptive_reverb():
    """Demonstrate adaptive reverb with synthetic audio."""

    print("\n" + "#"*70)
    print("# ADAPTIVE REVERB DEMO")
    print("# Context-Aware Parameter Adaptation")
    print("#"*70)

    sr = 48000
    processor = AdaptiveReverbProcessor(sr=sr)

    # Create test audio with varying characteristics
    print("\n1. Creating test audio with varying dynamics...")

    duration = 15  # seconds
    t = torch.arange(int(duration * sr)) / sr

    audio = torch.zeros(int(duration * sr))

    # Section 1 (0-5s): Quiet, sustained tones
    t1 = t[:5*sr]
    audio[:5*sr] = 0.1 * torch.sin(2 * np.pi * 440 * t1)

    # Section 2 (5-10s): Loud, dense chords
    t2 = t[:5*sr]
    for freq in [440, 554, 659, 880, 1109]:
        start = 5 * sr
        audio[start:start+5*sr] += 0.25 * torch.sin(2 * np.pi * freq * t2)

    # Section 3 (10-15s): Percussive hits
    t3 = t[:5*sr]
    for i in range(10):
        pos = 10 * sr + i * sr // 2
        if pos + 1000 < len(audio):
            audio[pos:pos+500] += torch.exp(-torch.arange(500) / 50) * 0.8

    save_audio(audio.unsqueeze(0), sr, 'adaptive_dry.mp3')
    print(f"   Saved: adaptive_dry.mp3")
    print(f"   Section 1 (0-5s):   Quiet, sustained → expect MORE reverb")
    print(f"   Section 2 (5-10s):  Loud, dense → expect LESS reverb")
    print(f"   Section 3 (10-15s): Percussive → expect TIGHT reverb")

    # Process with different strategies
    strategies = ['smart', 'inverse']

    for strategy in strategies:
        print(f"\n2. Processing with '{strategy}' adaptation strategy...")

        output, timeline = processor.process_adaptive(
            audio,
            rules=strategy,
            smoothing=0.85
        )

        # Save
        output_file = f'adaptive_{strategy}.mp3'
        save_audio(output.unsqueeze(0), sr, output_file)
        print(f"   Saved: {output_file}")

        # Plot
        plot_file = f'adaptive_{strategy}_timeline.png'
        processor.plot_adaptation(timeline, plot_file)

    print(f"\n{'='*70}")
    print(f"ADAPTIVE REVERB DEMO COMPLETE!")
    print(f"{'='*70}")
    print(f"\nGenerated files:")
    print(f"  - adaptive_dry.mp3                    : Original audio")
    print(f"  - adaptive_smart.mp3                  : Smart adaptation")
    print(f"  - adaptive_inverse.mp3                : Inverse adaptation")
    print(f"  - adaptive_smart_timeline.png         : Parameter evolution (smart)")
    print(f"  - adaptive_inverse_timeline.png       : Parameter evolution (inverse)")
    print(f"\nKey insight:")
    print(f"  Smart strategy: Adapts reverb to audio content")
    print(f"    - Quiet sections → more reverb (spacious)")
    print(f"    - Loud sections → less reverb (clear)")
    print(f"    - Bright content → less damping (preserve highs)")
    print(f"\n  This enables intelligent, context-aware mixing!")
    print(f"{'='*70}\n")


def process_file_adaptive(input_file, output_file, rules='smart', smoothing=0.9):
    """Process audio file with adaptive reverb."""

    print(f"\n{'='*70}")
    print(f"Processing: {input_file}")
    print(f"{'='*70}")

    sr = 48000
    processor = AdaptiveReverbProcessor(sr=sr)

    # Load audio
    print(f"\nLoading audio...")
    audio, _ = load_audio(input_file, mono=True, target_sample_rate=sr)

    # Truncate if too long
    max_samples = 60 * sr  # 1 minute max
    if len(audio) > max_samples:
        audio = audio[:max_samples]
        print(f"Truncated to 60 seconds")

    # Process
    output, timeline = processor.process_adaptive(audio, rules=rules, smoothing=smoothing)

    # Save
    save_audio(output.unsqueeze(0), sr, output_file)
    print(f"\nOutput saved: {output_file}")

    # Plot
    plot_file = output_file.replace('.mp3', '_timeline.png')
    processor.plot_adaptation(timeline, plot_file)

    print(f"{'='*70}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Adaptive Reverb - Context-Aware Processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo with synthetic audio (recommended)
  python app_adaptive_reverb.py --demo

  # Process your audio file
  python app_adaptive_reverb.py \\
      --input my_song.mp3 \\
      --output my_song_adaptive.mp3

  # Different adaptation strategies
  python app_adaptive_reverb.py \\
      --input song.mp3 \\
      --output song_adaptive.mp3 \\
      --rules extreme \\
      --smoothing 0.95
        """
    )

    parser.add_argument('--demo', action='store_true',
                       help='Run demo with synthetic audio')
    parser.add_argument('--input', '-i', type=str,
                       help='Input audio file')
    parser.add_argument('--output', '-o', type=str,
                       help='Output audio file')
    parser.add_argument('--rules', type=str, default='smart',
                       choices=['smart', 'inverse', 'extreme'],
                       help='Adaptation strategy (default: smart)')
    parser.add_argument('--smoothing', type=float, default=0.9,
                       help='Parameter smoothing (0-1, default: 0.9)')

    args = parser.parse_args()

    if args.demo:
        demo_adaptive_reverb()
    elif args.input and args.output:
        process_file_adaptive(args.input, args.output, args.rules, args.smoothing)
    else:
        print("Error: Use --demo or provide --input and --output")
        parser.print_help()


if __name__ == '__main__':
    main()
