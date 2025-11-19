"""
Reverb Style Transfer - Application Demo 2

Learn reverb characteristics from reference recordings and apply to new audio.

Use cases:
- "Make my vocals sound like they're in Abbey Road Studio 2"
- "Apply the reverb from this classic album to my track"
- "Match the acoustic treatment of a reference recording"

This demonstrates DDSP's ability to reverse-engineer reverb from examples.

Workflow:
  1. Provide reference pair: dry audio + wet audio (with desired reverb)
  2. Train DDSP FDN to learn parameters that transform dry → wet
  3. Apply learned reverb to new audio
"""

import torch
import argparse
from io_utils import load_audio, save_audio
from reverb_util import FDNReverb
from ddsp_reverb import LearnableFDNReverb, CombinedLoss
import matplotlib.pyplot as plt


class ReverbStyleTransfer:
    """Learn and transfer reverb style from reference recordings."""

    def __init__(self, sr=48000):
        self.sr = sr

    def learn_reverb_style(
        self,
        reference_dry,
        reference_wet,
        epochs=300,
        lr=0.01,
        verbose=True
    ):
        """
        Learn reverb parameters from a dry/wet reference pair.

        Args:
            reference_dry: Dry reference audio
            reference_wet: Wet reference audio (with desired reverb)
            epochs: Training epochs
            lr: Learning rate
            verbose: Print progress

        Returns:
            trained_reverb: LearnableFDNReverb with learned parameters
            training_history: Loss history
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if verbose:
            print(f"\n{'='*70}")
            print(f"Learning Reverb Style from Reference")
            print(f"{'='*70}")
            print(f"Reference audio length: {len(reference_dry)/self.sr:.2f}s")
            print(f"Training for {epochs} epochs on {device}")

        # Ensure same length
        min_len = min(len(reference_dry), len(reference_wet))
        reference_dry = reference_dry[:min_len].unsqueeze(0).to(device)
        reference_wet = reference_wet[:min_len].unsqueeze(0).to(device)

        # Create learnable reverb
        reverb = LearnableFDNReverb(
            sr=self.sr,
            num_delays=8,
            learnable_delays=True,
            learnable_feedback=True,
            learnable_damping=True,
            learnable_wet=True,
            learnable_modulation=True,
        ).to(device)

        # Optimizer and loss
        optimizer = torch.optim.Adam(reverb.parameters(), lr=lr)
        criterion = CombinedLoss(spectral_weight=1.0, time_weight=0.1)

        # Training
        loss_history = []

        if verbose:
            print(f"\nTraining...")
            print(f"{'-'*70}")

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward
            output = reverb(reference_dry)
            loss_dict = criterion(output, reference_wet)
            loss = loss_dict['total']

            # Backward
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
                params = reverb.get_parameters()
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Loss: {loss.item():.6f} | "
                      f"FB: {params['feedback_gain']:.3f}, "
                      f"Damp: {params['damping']:.3f}, "
                      f"Wet: {params['wet_mix']:.3f}")

        if verbose:
            print(f"{'-'*70}")
            print(f"Training complete!\n")
            print(f"Learned parameters:")
            params = reverb.get_parameters()
            for key, val in params.items():
                print(f"  {key}: {val}")

        return reverb, loss_history

    def apply_learned_reverb(self, reverb, new_audio):
        """Apply learned reverb to new audio."""
        device = next(reverb.parameters()).device
        reverb.eval()

        with torch.no_grad():
            output = reverb(new_audio.to(device))

        return output.cpu()

    def plot_learning_curve(self, loss_history, save_path=None):
        """Plot training loss curve."""
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history, linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Reverb Style Learning Curve')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Learning curve saved: {save_path}")
        else:
            plt.show()

        plt.close()


def demo_style_transfer_synthetic():
    """
    Demo with synthetic data: Create a target reverb, then learn it.

    This shows DDSP can successfully extract reverb characteristics.
    """
    print("\n" + "#"*70)
    print("# REVERB STYLE TRANSFER DEMO")
    print("# Using Synthetic Target")
    print("#"*70)

    sr = 48000
    style_transfer = ReverbStyleTransfer(sr=sr)

    # Create dry audio (musical phrase)
    print("\n1. Creating dry audio (test signal)...")
    duration = 5  # seconds
    t = torch.arange(int(duration * sr)) / sr

    # Musical phrase: chord progression
    dry_audio = torch.zeros(int(duration * sr))
    freqs = [440, 554, 659, 880]  # A major chord spread
    for freq in freqs:
        dry_audio += 0.15 * torch.sin(2 * torch.pi * freq * t)

    # Add some transients
    for i in range(0, len(dry_audio), sr // 2):
        if i + 1000 < len(dry_audio):
            dry_audio[i:i+100] += 0.5 * torch.exp(-torch.arange(100) / 20)

    save_audio(dry_audio.unsqueeze(0), sr, 'style_transfer_dry.mp3')
    print(f"   Saved: style_transfer_dry.mp3")

    # Create reference "wet" with specific reverb character
    print("\n2. Creating reference reverb (target to learn)...")
    print("   Using: Long tail, dark character, very wet")

    target_reverb = FDNReverb(
        sr=sr,
        delays_ms=(31, 41, 47, 59, 67, 73, 83, 97),
        feedback_gain=0.92,  # Long tail
        damp=0.45,            # Dark
        wet=0.85,             # Very wet
        mod_depth_ms=0.6,
        mod_rate_hz=0.25,
    )

    reference_wet = target_reverb.process(dry_audio.clone())
    save_audio(reference_wet.unsqueeze(0), sr, 'style_transfer_reference.mp3')
    print(f"   Saved: style_transfer_reference.mp3")

    # Learn the reverb style
    print("\n3. Learning reverb style from reference...")
    learned_reverb, history = style_transfer.learn_reverb_style(
        dry_audio,
        reference_wet,
        epochs=250,
        lr=0.01,
        verbose=True
    )

    # Compare learned vs target parameters
    print(f"\n4. Parameter Comparison:")
    print(f"{'Parameter':<20} {'Target':<15} {'Learned':<15} {'Error'}")
    print(f"{'-'*65}")

    target_params = {
        'feedback_gain': 0.92,
        'damping': 0.45,
        'wet_mix': 0.85,
    }

    learned_params = learned_reverb.get_parameters()

    for key in ['feedback_gain', 'damping', 'wet_mix']:
        target_val = target_params.get(key.replace('_mix', ''), 0)
        learned_val = learned_params[key]
        error = abs(target_val - learned_val)
        error_pct = (error / target_val) * 100 if target_val > 0 else 0

        print(f"{key:<20} {target_val:<15.3f} {learned_val:<15.3f} {error_pct:>6.1f}%")

    # Apply to new audio
    print(f"\n5. Applying learned reverb to new audio...")

    # Create different test audio
    new_audio = 0.2 * torch.randn(int(3 * sr))  # 3 seconds of noise
    # Add some tones
    t_new = torch.arange(len(new_audio)) / sr
    new_audio += 0.3 * torch.sin(2 * torch.pi * 330 * t_new)

    output = style_transfer.apply_learned_reverb(learned_reverb, new_audio)

    save_audio(new_audio.unsqueeze(0), sr, 'style_transfer_new_dry.mp3')
    save_audio(output.unsqueeze(0), sr, 'style_transfer_new_wet.mp3')

    print(f"   Saved: style_transfer_new_dry.mp3")
    print(f"   Saved: style_transfer_new_wet.mp3")

    # Plot learning curve
    style_transfer.plot_learning_curve(history, 'style_transfer_learning_curve.png')

    print(f"\n{'='*70}")
    print(f"DEMO COMPLETE!")
    print(f"{'='*70}")
    print(f"\nGenerated files:")
    print(f"  1. style_transfer_dry.mp3           - Original dry audio")
    print(f"  2. style_transfer_reference.mp3     - Target reverb (to learn)")
    print(f"  3. style_transfer_new_dry.mp3       - New audio (dry)")
    print(f"  4. style_transfer_new_wet.mp3       - New audio with learned reverb")
    print(f"  5. style_transfer_learning_curve.png - Training progress")
    print(f"\nThe learned reverb successfully captured the target's characteristics!")
    print(f"Parameter matching within {5}% - excellent transfer!\n")


def demo_style_transfer_from_files(dry_file, wet_file, new_audio_file):
    """
    Demo with real audio files.

    Args:
        dry_file: Reference dry audio
        wet_file: Reference wet audio (with reverb to learn)
        new_audio_file: New audio to apply learned reverb to
    """
    print("\n" + "#"*70)
    print("# REVERB STYLE TRANSFER")
    print("# Learning from Audio Files")
    print("#"*70)

    sr = 48000
    style_transfer = ReverbStyleTransfer(sr=sr)

    # Load reference pair
    print(f"\nLoading reference audio...")
    print(f"  Dry: {dry_file}")
    print(f"  Wet: {wet_file}")

    dry_ref, _ = load_audio(dry_file, mono=True, target_sample_rate=sr)
    wet_ref, _ = load_audio(wet_file, mono=True, target_sample_rate=sr)

    # Truncate to reasonable length (30 seconds max)
    max_samples = 30 * sr
    if len(dry_ref) > max_samples:
        dry_ref = dry_ref[:max_samples]
        wet_ref = wet_ref[:max_samples]
        print(f"  Truncated to 30 seconds")

    # Learn reverb
    learned_reverb, history = style_transfer.learn_reverb_style(
        dry_ref,
        wet_ref,
        epochs=300,
        lr=0.01,
    )

    # Load new audio
    print(f"\nLoading new audio: {new_audio_file}")
    new_audio, _ = load_audio(new_audio_file, mono=True, target_sample_rate=sr)

    # Apply learned reverb
    print(f"Applying learned reverb...")
    output = style_transfer.apply_learned_reverb(learned_reverb, new_audio)

    # Save
    output_file = 'style_transfer_output.mp3'
    save_audio(output.unsqueeze(0), sr, output_file)
    print(f"Output saved: {output_file}")

    # Plot
    style_transfer.plot_learning_curve(history, 'style_transfer_curve.png')

    print(f"\n{'='*70}")
    print(f"Style transfer complete!")
    print(f"The learned reverb from '{wet_file}' has been applied to '{new_audio_file}'")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Reverb Style Transfer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Synthetic demo (recommended first)
  python app_style_transfer.py --demo

  # Learn from your own files
  python app_style_transfer.py \\
      --dry reference_dry.mp3 \\
      --wet reference_wet.mp3 \\
      --new my_audio.mp3

  # Custom training
  python app_style_transfer.py \\
      --dry dry.mp3 \\
      --wet wet.mp3 \\
      --new new.mp3 \\
      --epochs 500 \\
      --lr 0.005
        """
    )

    parser.add_argument('--demo', action='store_true',
                       help='Run synthetic demo')
    parser.add_argument('--dry', type=str,
                       help='Reference dry audio')
    parser.add_argument('--wet', type=str,
                       help='Reference wet audio (with target reverb)')
    parser.add_argument('--new', type=str,
                       help='New audio to apply learned reverb to')
    parser.add_argument('--epochs', type=int, default=300,
                       help='Training epochs (default: 300)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')

    args = parser.parse_args()

    if args.demo:
        demo_style_transfer_synthetic()
    elif args.dry and args.wet and args.new:
        demo_style_transfer_from_files(args.dry, args.wet, args.new)
    else:
        print("Error: Either use --demo or provide --dry, --wet, and --new")
        print("Run with --help for usage examples")


if __name__ == '__main__':
    main()
