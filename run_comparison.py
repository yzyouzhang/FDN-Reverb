"""
Complete Comparison Suite

Runs comprehensive comparison between FDN implementations:
1. Performance benchmarks
2. Audio quality metrics
3. Visualizations

This is the main script to run all comparisons at once.

Usage:
    python run_comparison.py --input dry.mp3 --full
"""

import torch
import argparse
import os
import json
from io_utils import load_audio
from compare_performance import PerformanceBenchmark
from compare_quality import ReverbQualityComparison
from visualize_comparison import ComparisonVisualizer


def run_complete_comparison(
    input_file='dry.mp3',
    sr=48000,
    output_dir='comparison_results',
    performance_runs=5,
    test_duration=10,
    skip_performance=False,
    skip_quality=False,
    skip_visualization=False,
):
    """
    Run complete comparison suite.

    Args:
        input_file: Input audio file
        sr: Sample rate
        output_dir: Output directory for results
        performance_runs: Number of runs for performance benchmark
        test_duration: Duration of test audio for performance
        skip_performance: Skip performance benchmarks
        skip_quality: Skip quality metrics
        skip_visualization: Skip visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"#  COMPLETE FDN REVERB COMPARISON SUITE")
    print(f"#")
    print(f"#  Comparing:")
    print(f"#    - Traditional FDN (reverb_util.py)")
    print(f"#    - DDSP FDN (ddsp_reverb.py)")
    print(f"#")
    print(f"#  Input: {input_file}")
    print(f"#  Output: {output_dir}/")
    print(f"{'#'*70}\n")

    results = {}

    # ========================================
    # 1. PERFORMANCE BENCHMARK
    # ========================================
    if not skip_performance:
        print(f"\n{'='*70}")
        print(f"1. PERFORMANCE BENCHMARK")
        print(f"{'='*70}")

        benchmark = PerformanceBenchmark(sr=sr)
        benchmark.run_full_benchmark(
            audio_duration=test_duration,
            num_runs=performance_runs
        )

        # Save results
        perf_results_path = f'{output_dir}/performance_results.json'
        benchmark.save_results(perf_results_path)

        results['performance'] = benchmark.results

        # Visualize performance
        visualizer = ComparisonVisualizer(sr=sr)
        if 'traditional_fdn' in benchmark.results and 'ddsp_fdn' in benchmark.results:
            visualizer.plot_performance_comparison(
                benchmark.results,
                save_path=f'{output_dir}/performance_comparison.png'
            )

        if 'scalability' in benchmark.results:
            visualizer.plot_scalability(
                benchmark.results['scalability'],
                save_path=f'{output_dir}/scalability.png'
            )

    # ========================================
    # 2. AUDIO QUALITY METRICS
    # ========================================
    if not skip_quality:
        print(f"\n{'='*70}")
        print(f"2. AUDIO QUALITY METRICS")
        print(f"{'='*70}")

        # Load audio
        print(f"\nLoading audio: {input_file}")
        dry_audio, audio_sr = load_audio(input_file, mono=True, target_sample_rate=sr)

        # Truncate to reasonable length
        max_samples = 30 * sr
        if len(dry_audio) > max_samples:
            dry_audio = dry_audio[:max_samples]
            print(f"Truncated to 30 seconds for analysis")

        # Run quality comparison
        quality_comp = ReverbQualityComparison(sr=sr)
        quality_results = quality_comp.compare_implementations(dry_audio)

        results['quality'] = quality_results['comparison']

        # Save quality results
        quality_path = f'{output_dir}/quality_results.json'
        with open(quality_path, 'w') as f:
            json.dump(quality_results['comparison'], f, indent=2)
        print(f"\nQuality results saved: {quality_path}")

        # Save audio outputs
        from io_utils import save_audio
        save_audio(quality_results['traditional_output'].unsqueeze(0),
                  sr, f'{output_dir}/output_traditional.mp3')
        save_audio(quality_results['ddsp_output'].unsqueeze(0),
                  sr, f'{output_dir}/output_ddsp.mp3')
        print(f"Audio outputs saved:")
        print(f"  - {output_dir}/output_traditional.mp3")
        print(f"  - {output_dir}/output_ddsp.mp3")

    # ========================================
    # 3. VISUALIZATIONS
    # ========================================
    if not skip_visualization:
        print(f"\n{'='*70}")
        print(f"3. VISUALIZATIONS")
        print(f"{'='*70}")

        # Load audio if not already loaded
        if skip_quality:
            print(f"\nLoading audio: {input_file}")
            dry_audio, audio_sr = load_audio(input_file, mono=True, target_sample_rate=sr)
            max_samples = 30 * sr
            if len(dry_audio) > max_samples:
                dry_audio = dry_audio[:max_samples]

        # Create visualizations
        visualizer = ComparisonVisualizer(sr=sr)
        visualizer.create_full_comparison_report(dry_audio, output_dir=output_dir)

    # ========================================
    # 4. FINAL SUMMARY
    # ========================================
    print(f"\n{'='*70}")
    print(f"COMPARISON COMPLETE!")
    print(f"{'='*70}")

    print(f"\nResults saved to: {output_dir}/")
    print(f"\nGenerated files:")

    if not skip_performance:
        print(f"\n  Performance:")
        print(f"    - performance_results.json")
        print(f"    - performance_comparison.png")
        if 'scalability' in results.get('performance', {}):
            print(f"    - scalability.png")

    if not skip_quality:
        print(f"\n  Quality:")
        print(f"    - quality_results.json")
        print(f"    - output_traditional.mp3")
        print(f"    - output_ddsp.mp3")

    if not skip_visualization:
        print(f"\n  Visualizations:")
        print(f"    - waveforms.png")
        print(f"    - spectrograms.png")
        print(f"    - frequency_response.png")

    print(f"\n{'='*70}")

    # Print key insights
    print_key_insights(results)

    return results


def print_key_insights(results):
    """Print key insights from comparison."""
    print(f"\nKEY INSIGHTS:")
    print(f"{'='*70}")

    if 'performance' in results:
        perf = results['performance']
        if 'traditional_fdn' in perf and 'ddsp_fdn' in perf:
            trad = perf['traditional_fdn']
            ddsp = perf['ddsp_fdn']

            speedup = ddsp['avg_time'] / trad['avg_time']

            print(f"\n1. Performance:")
            if speedup > 1:
                print(f"   → Traditional FDN is {speedup:.1f}x FASTER than DDSP")
                print(f"     (C++ optimizations provide significant speedup)")
            else:
                print(f"   → DDSP FDN is {1/speedup:.1f}x faster than Traditional")

            print(f"\n   Real-time factors:")
            print(f"     - Traditional: {trad['throughput']:.1f}x")
            print(f"     - DDSP: {ddsp['throughput']:.1f}x")

            if trad['throughput'] > 1 and ddsp['throughput'] > 1:
                print(f"   ✓ Both implementations can run in real-time")
            elif trad['throughput'] > 1:
                print(f"   → Only Traditional can run in real-time")
            else:
                print(f"   ✗ Neither can run in real-time for this audio length")

    if 'quality' in results:
        quality = results['quality']
        spec_dist = quality.get('spectral_distance', 0)
        temp_dist = quality.get('temporal_distance', 0)

        print(f"\n2. Audio Quality:")
        if spec_dist < 1000:
            print(f"   ✓ Very similar spectral characteristics")
            print(f"     (Both implementations produce nearly identical frequency content)")
        elif spec_dist < 5000:
            print(f"   → Moderately similar spectral characteristics")
            print(f"     (Minor differences in frequency content)")
        else:
            print(f"   ✗ Different spectral characteristics")
            print(f"     (Significant differences in frequency content)")

        # RT60 comparison
        s1 = quality['signal1']
        s2 = quality['signal2']
        rt60_diff = abs(s1['rt60_full'] - s2['rt60_full'])
        rt60_rel = (rt60_diff / (s1['rt60_full'] + 1e-10)) * 100

        print(f"\n   Decay time (RT60):")
        print(f"     - {s1['label']}: {s1['rt60_full']:.3f}s")
        print(f"     - {s2['label']}: {s2['rt60_full']:.3f}s")
        print(f"     - Difference: {rt60_rel:.1f}%")

    print(f"\n3. Use Case Recommendations:")
    print(f"   - Traditional FDN: Production use, real-time processing")
    print(f"   - DDSP FDN: Research, learning from data, parameter discovery")

    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Complete FDN Reverb Comparison Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full comparison
  python run_comparison.py --input dry.mp3 --full

  # Quick comparison (skip some tests)
  python run_comparison.py --input dry.mp3 --quick

  # Performance only
  python run_comparison.py --skip_quality --skip_visualization

  # Quality and visualization only
  python run_comparison.py --skip_performance
        """
    )

    parser.add_argument('--input', '-i', type=str, default='dry.mp3',
                       help='Input audio file (default: dry.mp3)')
    parser.add_argument('--output_dir', '-o', type=str, default='comparison_results',
                       help='Output directory (default: comparison_results)')
    parser.add_argument('--sr', type=int, default=48000,
                       help='Sample rate (default: 48000)')

    # Test configuration
    parser.add_argument('--performance_runs', type=int, default=5,
                       help='Runs for performance benchmark (default: 5)')
    parser.add_argument('--test_duration', type=float, default=10,
                       help='Duration for performance test (default: 10s)')

    # Quick/full modes
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (fewer runs, shorter tests)')
    parser.add_argument('--full', action='store_true',
                       help='Full mode (more runs, comprehensive tests)')

    # Skip options
    parser.add_argument('--skip_performance', action='store_true',
                       help='Skip performance benchmarks')
    parser.add_argument('--skip_quality', action='store_true',
                       help='Skip quality metrics')
    parser.add_argument('--skip_visualization', action='store_true',
                       help='Skip visualizations')

    args = parser.parse_args()

    # Adjust settings for quick/full modes
    if args.quick:
        args.performance_runs = 3
        args.test_duration = 5
    elif args.full:
        args.performance_runs = 10
        args.test_duration = 30

    # Run comparison
    results = run_complete_comparison(
        input_file=args.input,
        sr=args.sr,
        output_dir=args.output_dir,
        performance_runs=args.performance_runs,
        test_duration=args.test_duration,
        skip_performance=args.skip_performance,
        skip_quality=args.skip_quality,
        skip_visualization=args.skip_visualization,
    )


if __name__ == '__main__':
    main()
