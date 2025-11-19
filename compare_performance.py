"""
Performance Comparison Script

Compares processing speed between:
1. FDN-Reverb (this repo) - Traditional implementation with C++/Python
2. DDSP FDN (this repo) - Learnable version (pure PyTorch)
3. dafx25-ddsp-tutorial (FLAMO-based) - if available

Metrics:
- Processing time per second of audio
- Throughput (seconds of audio processed per second)
- Memory usage
- Scalability (different audio lengths)
"""

import torch
import numpy as np
import time
import psutil
import os
from io_utils import load_audio
from reverb_util import FDNReverb
from ddsp_reverb import LearnableFDNReverb


class PerformanceBenchmark:
    """Benchmark performance of different FDN implementations."""

    def __init__(self, sr=48000):
        self.sr = sr
        self.results = {}

    def generate_test_audio(self, duration_sec):
        """Generate test audio of specified duration."""
        num_samples = int(duration_sec * self.sr)
        # White noise + some tones for realistic signal
        noise = torch.randn(num_samples) * 0.1
        t = torch.arange(num_samples) / self.sr
        tone1 = 0.3 * torch.sin(2 * np.pi * 440 * t)
        tone2 = 0.2 * torch.sin(2 * np.pi * 880 * t)
        audio = noise + tone1 + tone2
        return audio

    def measure_memory(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB

    def benchmark_traditional_fdn(self, audio, num_runs=3):
        """Benchmark traditional FDN from reverb_util.py."""
        print(f"\n{'='*60}")
        print(f"Benchmarking Traditional FDN (reverb_util.py)")
        print(f"{'='*60}")

        reverb = FDNReverb(
            sr=self.sr,
            delays_ms=(29, 37, 43, 53, 61, 71, 79, 89),
            feedback_gain=0.9,
            damp=0.25,
            wet=0.9,
        )

        # Warmup
        _ = reverb.process(audio.clone())

        # Benchmark
        times = []
        mem_before = self.measure_memory()

        for i in range(num_runs):
            start = time.time()
            output = reverb.process(audio.clone())
            end = time.time()
            elapsed = end - start
            times.append(elapsed)
            print(f"  Run {i+1}/{num_runs}: {elapsed:.4f}s")

        mem_after = self.measure_memory()

        avg_time = np.mean(times)
        std_time = np.std(times)
        audio_duration = len(audio) / self.sr
        throughput = audio_duration / avg_time

        results = {
            'avg_time': avg_time,
            'std_time': std_time,
            'min_time': np.min(times),
            'max_time': np.max(times),
            'throughput': throughput,
            'memory_delta': mem_after - mem_before,
            'audio_duration': audio_duration,
        }

        print(f"\n  Results:")
        print(f"    Avg time: {avg_time:.4f}s ± {std_time:.4f}s")
        print(f"    Throughput: {throughput:.2f}x real-time")
        print(f"    Memory delta: {results['memory_delta']:.2f} MB")

        self.results['traditional_fdn'] = results
        return output

    def benchmark_ddsp_fdn(self, audio, num_runs=3):
        """Benchmark DDSP FDN from ddsp_reverb.py."""
        print(f"\n{'='*60}")
        print(f"Benchmarking DDSP FDN (ddsp_reverb.py)")
        print(f"{'='*60}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Device: {device}")

        reverb = LearnableFDNReverb(
            sr=self.sr,
            num_delays=8,
            init_feedback_gain=0.9,
            init_damp=0.25,
            init_wet=0.9,
        ).to(device)

        reverb.eval()
        audio_device = audio.to(device)

        # Warmup
        with torch.no_grad():
            _ = reverb(audio_device.clone())

        # Benchmark
        times = []
        mem_before = self.measure_memory()

        for i in range(num_runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            with torch.no_grad():
                output = reverb(audio_device.clone())
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.time()
            elapsed = end - start
            times.append(elapsed)
            print(f"  Run {i+1}/{num_runs}: {elapsed:.4f}s")

        mem_after = self.measure_memory()

        avg_time = np.mean(times)
        std_time = np.std(times)
        audio_duration = len(audio) / self.sr
        throughput = audio_duration / avg_time

        results = {
            'avg_time': avg_time,
            'std_time': std_time,
            'min_time': np.min(times),
            'max_time': np.max(times),
            'throughput': throughput,
            'memory_delta': mem_after - mem_before,
            'audio_duration': audio_duration,
            'device': str(device),
        }

        print(f"\n  Results:")
        print(f"    Avg time: {avg_time:.4f}s ± {std_time:.4f}s")
        print(f"    Throughput: {throughput:.2f}x real-time")
        print(f"    Memory delta: {results['memory_delta']:.2f} MB")

        self.results['ddsp_fdn'] = results
        return output.cpu()

    def benchmark_scalability(self, durations=[1, 5, 10, 30, 60]):
        """Test scalability with different audio lengths."""
        print(f"\n{'='*60}")
        print(f"Scalability Test")
        print(f"{'='*60}")

        scalability_results = {
            'durations': durations,
            'traditional_times': [],
            'traditional_throughput': [],
            'ddsp_times': [],
            'ddsp_throughput': [],
        }

        for duration in durations:
            print(f"\n  Testing with {duration}s audio...")
            audio = self.generate_test_audio(duration)

            # Traditional FDN
            reverb_trad = FDNReverb(sr=self.sr)
            start = time.time()
            _ = reverb_trad.process(audio.clone())
            trad_time = time.time() - start
            scalability_results['traditional_times'].append(trad_time)
            scalability_results['traditional_throughput'].append(duration / trad_time)
            print(f"    Traditional: {trad_time:.4f}s ({duration/trad_time:.2f}x)")

            # DDSP FDN
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            reverb_ddsp = LearnableFDNReverb(sr=self.sr).to(device)
            reverb_ddsp.eval()
            with torch.no_grad():
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start = time.time()
                _ = reverb_ddsp(audio.to(device).clone())
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                ddsp_time = time.time() - start
            scalability_results['ddsp_times'].append(ddsp_time)
            scalability_results['ddsp_throughput'].append(duration / ddsp_time)
            print(f"    DDSP: {ddsp_time:.4f}s ({duration/ddsp_time:.2f}x)")

        self.results['scalability'] = scalability_results

    def run_full_benchmark(self, audio_duration=10, num_runs=5):
        """Run complete benchmark suite."""
        print(f"\n{'#'*60}")
        print(f"# Performance Benchmark Suite")
        print(f"# Audio duration: {audio_duration}s")
        print(f"# Sample rate: {self.sr} Hz")
        print(f"# Runs per test: {num_runs}")
        print(f"{'#'*60}")

        # Generate test audio
        print(f"\nGenerating {audio_duration}s test audio...")
        audio = self.generate_test_audio(audio_duration)

        # Run benchmarks
        trad_output = self.benchmark_traditional_fdn(audio, num_runs)
        ddsp_output = self.benchmark_ddsp_fdn(audio, num_runs)

        # Scalability test
        self.benchmark_scalability()

        # Summary
        self.print_summary()

        return {
            'traditional_output': trad_output,
            'ddsp_output': ddsp_output,
        }

    def print_summary(self):
        """Print summary comparison."""
        print(f"\n{'='*60}")
        print(f"SUMMARY COMPARISON")
        print(f"{'='*60}")

        if 'traditional_fdn' in self.results and 'ddsp_fdn' in self.results:
            trad = self.results['traditional_fdn']
            ddsp = self.results['ddsp_fdn']

            speedup = ddsp['avg_time'] / trad['avg_time']

            print(f"\nProcessing Time:")
            print(f"  Traditional FDN: {trad['avg_time']:.4f}s")
            print(f"  DDSP FDN:        {ddsp['avg_time']:.4f}s")
            print(f"  Speedup:         {speedup:.2f}x {'(Traditional faster)' if speedup > 1 else '(DDSP faster)'}")

            print(f"\nThroughput (real-time factor):")
            print(f"  Traditional FDN: {trad['throughput']:.2f}x")
            print(f"  DDSP FDN:        {ddsp['throughput']:.2f}x")

            print(f"\nMemory Usage:")
            print(f"  Traditional FDN: {trad['memory_delta']:.2f} MB")
            print(f"  DDSP FDN:        {ddsp['memory_delta']:.2f} MB")

        print(f"\n{'='*60}\n")

    def save_results(self, filename='performance_results.json'):
        """Save results to JSON file."""
        import json

        # Convert numpy types to Python types
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_types(val) for key, val in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj

        results_clean = convert_types(self.results)

        with open(filename, 'w') as f:
            json.dump(results_clean, f, indent=2)

        print(f"Results saved to: {filename}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Performance Benchmark for FDN Reverb')
    parser.add_argument('--duration', type=float, default=10,
                       help='Test audio duration in seconds (default: 10)')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of runs per test (default: 5)')
    parser.add_argument('--sr', type=int, default=48000,
                       help='Sample rate (default: 48000)')
    parser.add_argument('--save', type=str, default='performance_results.json',
                       help='Save results to JSON file')
    parser.add_argument('--no_scalability', action='store_true',
                       help='Skip scalability test')

    args = parser.parse_args()

    benchmark = PerformanceBenchmark(sr=args.sr)

    if args.no_scalability:
        # Quick benchmark
        audio = benchmark.generate_test_audio(args.duration)
        benchmark.benchmark_traditional_fdn(audio, args.runs)
        benchmark.benchmark_ddsp_fdn(audio, args.runs)
        benchmark.print_summary()
    else:
        # Full benchmark
        benchmark.run_full_benchmark(audio_duration=args.duration, num_runs=args.runs)

    if args.save:
        benchmark.save_results(args.save)


if __name__ == '__main__':
    main()
