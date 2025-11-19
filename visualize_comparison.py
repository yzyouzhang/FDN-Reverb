"""
Visualization Script for Comparison Results

Creates comprehensive visualizations comparing:
1. Performance metrics (speed, throughput)
2. Audio quality metrics (spectral, temporal)
3. Waveforms and spectrograms
4. Frequency responses
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from io_utils import load_audio
from reverb_util import FDNReverb
from ddsp_reverb import LearnableFDNReverb


class ComparisonVisualizer:
    """Visualize comparison results."""

    def __init__(self, sr=48000):
        self.sr = sr

    def plot_waveforms(self, signals_dict, save_path=None):
        """
        Plot waveforms for multiple signals.

        Args:
            signals_dict: Dict of {label: audio_tensor}
            save_path: Optional path to save figure
        """
        num_signals = len(signals_dict)
        fig, axes = plt.subplots(num_signals, 1, figsize=(12, 3*num_signals))

        if num_signals == 1:
            axes = [axes]

        for ax, (label, signal) in zip(axes, signals_dict.items()):
            t = np.arange(len(signal)) / self.sr
            ax.plot(t, signal.numpy(), linewidth=0.5, alpha=0.7)
            ax.set_title(f'Waveform: {label}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, min(5, t[-1])])  # Show first 5 seconds

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Waveform plot saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_spectrograms(self, signals_dict, save_path=None):
        """
        Plot spectrograms for multiple signals.

        Args:
            signals_dict: Dict of {label: audio_tensor}
            save_path: Optional path to save figure
        """
        num_signals = len(signals_dict)
        fig, axes = plt.subplots(num_signals, 1, figsize=(14, 3*num_signals))

        if num_signals == 1:
            axes = [axes]

        for ax, (label, signal) in zip(axes, signals_dict.items()):
            # Compute STFT
            stft = torch.stft(
                signal,
                n_fft=2048,
                hop_length=512,
                window=torch.hann_window(2048),
                return_complex=True
            )
            magnitude = torch.abs(stft).numpy()
            magnitude_db = 20 * np.log10(magnitude + 1e-10)

            # Time and frequency axes
            times = np.arange(magnitude_db.shape[1]) * 512 / self.sr
            freqs = np.fft.rfftfreq(2048, 1/self.sr)

            # Plot
            im = ax.pcolormesh(times, freqs, magnitude_db,
                              shading='gouraud', cmap='magma',
                              vmin=magnitude_db.max()-80, vmax=magnitude_db.max())
            ax.set_title(f'Spectrogram: {label}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_ylim([0, 10000])
            plt.colorbar(im, ax=ax, label='Magnitude (dB)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Spectrogram plot saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_frequency_response(self, signals_dict, save_path=None):
        """
        Plot frequency responses.

        Args:
            signals_dict: Dict of {label: audio_tensor}
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        for label, signal in signals_dict.items():
            # Compute magnitude spectrum
            fft = torch.fft.rfft(signal)
            magnitude = torch.abs(fft).numpy()
            magnitude_db = 20 * np.log10(magnitude + 1e-10)
            freqs = torch.fft.rfftfreq(len(signal), 1/self.sr).numpy()

            # Smooth for visualization
            from scipy.ndimage import uniform_filter1d
            magnitude_db_smooth = uniform_filter1d(magnitude_db, size=100)

            ax.plot(freqs, magnitude_db_smooth, label=label, linewidth=2, alpha=0.8)

        ax.set_title('Frequency Response Comparison')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_xscale('log')
        ax.set_xlim([20, self.sr/2])
        ax.grid(True, alpha=0.3, which='both')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Frequency response plot saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_performance_comparison(self, results, save_path=None):
        """
        Plot performance comparison bar charts.

        Args:
            results: Dict from performance benchmark
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Extract data
        labels = []
        times = []
        throughputs = []
        memories = []

        for impl in ['traditional_fdn', 'ddsp_fdn']:
            if impl in results:
                labels.append(impl.replace('_', ' ').title())
                times.append(results[impl]['avg_time'])
                throughputs.append(results[impl]['throughput'])
                memories.append(results[impl].get('memory_delta', 0))

        x = np.arange(len(labels))

        # Processing time
        axes[0].bar(x, times, color=['#3498db', '#e74c3c'])
        axes[0].set_ylabel('Time (seconds)')
        axes[0].set_title('Processing Time\n(Lower is Better)')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=15, ha='right')
        axes[0].grid(True, alpha=0.3, axis='y')

        # Add values on bars
        for i, v in enumerate(times):
            axes[0].text(i, v, f'{v:.3f}s', ha='center', va='bottom')

        # Throughput
        axes[1].bar(x, throughputs, color=['#3498db', '#e74c3c'])
        axes[1].set_ylabel('Real-time Factor')
        axes[1].set_title('Throughput\n(Higher is Better)')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=15, ha='right')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].axhline(y=1.0, color='green', linestyle='--', label='Real-time', alpha=0.5)
        axes[1].legend()

        for i, v in enumerate(throughputs):
            axes[1].text(i, v, f'{v:.1f}x', ha='center', va='bottom')

        # Memory
        axes[2].bar(x, memories, color=['#3498db', '#e74c3c'])
        axes[2].set_ylabel('Memory (MB)')
        axes[2].set_title('Memory Usage\n(Lower is Better)')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(labels, rotation=15, ha='right')
        axes[2].grid(True, alpha=0.3, axis='y')

        for i, v in enumerate(memories):
            axes[2].text(i, v, f'{v:.1f}MB', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Performance comparison saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_scalability(self, scalability_data, save_path=None):
        """
        Plot scalability results.

        Args:
            scalability_data: Dict with scalability test results
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        durations = scalability_data['durations']
        trad_times = scalability_data['traditional_times']
        ddsp_times = scalability_data['ddsp_times']
        trad_throughput = scalability_data['traditional_throughput']
        ddsp_throughput = scalability_data['ddsp_throughput']

        # Processing time vs audio length
        axes[0].plot(durations, trad_times, 'o-', label='Traditional FDN',
                    linewidth=2, markersize=8, color='#3498db')
        axes[0].plot(durations, ddsp_times, 's-', label='DDSP FDN',
                    linewidth=2, markersize=8, color='#e74c3c')
        axes[0].set_xlabel('Audio Duration (seconds)')
        axes[0].set_ylabel('Processing Time (seconds)')
        axes[0].set_title('Scalability: Processing Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Throughput vs audio length
        axes[1].plot(durations, trad_throughput, 'o-', label='Traditional FDN',
                    linewidth=2, markersize=8, color='#3498db')
        axes[1].plot(durations, ddsp_throughput, 's-', label='DDSP FDN',
                    linewidth=2, markersize=8, color='#e74c3c')
        axes[1].axhline(y=1.0, color='green', linestyle='--',
                       label='Real-time', alpha=0.5)
        axes[1].set_xlabel('Audio Duration (seconds)')
        axes[1].set_ylabel('Throughput (real-time factor)')
        axes[1].set_title('Scalability: Throughput')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Scalability plot saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def create_full_comparison_report(self, dry_audio, output_dir='comparison_results'):
        """
        Create a complete comparison report with all visualizations.

        Args:
            dry_audio: Input audio to process
            output_dir: Directory to save results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Creating Full Comparison Report")
        print(f"{'='*60}")

        # Process with both implementations
        print(f"\nProcessing audio with both implementations...")

        # Traditional
        trad_reverb = FDNReverb(sr=self.sr)
        trad_output = trad_reverb.process(dry_audio.clone())

        # DDSP
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ddsp_reverb = LearnableFDNReverb(sr=self.sr).to(device)
        ddsp_reverb.eval()
        with torch.no_grad():
            ddsp_output = ddsp_reverb(dry_audio.to(device)).cpu()

        signals = {
            'Dry (Input)': dry_audio,
            'Traditional FDN': trad_output,
            'DDSP FDN': ddsp_output,
        }

        # Create visualizations
        print(f"\nGenerating visualizations...")

        self.plot_waveforms(signals,
                           save_path=f'{output_dir}/waveforms.png')

        self.plot_spectrograms(signals,
                              save_path=f'{output_dir}/spectrograms.png')

        self.plot_frequency_response(
            {'Traditional FDN': trad_output, 'DDSP FDN': ddsp_output},
            save_path=f'{output_dir}/frequency_response.png'
        )

        print(f"\nReport saved to: {output_dir}/")
        print(f"Files created:")
        print(f"  - waveforms.png")
        print(f"  - spectrograms.png")
        print(f"  - frequency_response.png")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Visualize Comparison Results')
    parser.add_argument('--input', '-i', type=str, default='dry.mp3',
                       help='Input audio file')
    parser.add_argument('--output_dir', '-o', type=str, default='comparison_results',
                       help='Output directory for visualizations')
    parser.add_argument('--sr', type=int, default=48000,
                       help='Sample rate')
    parser.add_argument('--duration', type=float, default=10,
                       help='Duration to analyze (seconds)')

    args = parser.parse_args()

    # Load audio
    print(f"Loading audio: {args.input}")
    dry_audio, sr = load_audio(args.input, mono=True, target_sample_rate=args.sr)

    # Truncate
    max_samples = int(args.duration * sr)
    if len(dry_audio) > max_samples:
        dry_audio = dry_audio[:max_samples]

    # Create visualizations
    visualizer = ComparisonVisualizer(sr=sr)
    visualizer.create_full_comparison_report(dry_audio, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
