"""
Train DDSP FDN Reverb to Match Target Audio

This script demonstrates how to use gradient descent to learn FDN parameters
that match a target reverb sound.

Usage:
    python train_ddsp.py --dry dry.mp3 --target target_reverb.mp3 --epochs 200

The script will:
1. Load dry and target audio
2. Initialize a learnable FDN
3. Train it to match the target
4. Save the trained model and output
"""

import torch
import argparse
import os
from ddsp_reverb import LearnableFDNReverb, CombinedLoss, SpectralLoss
from io_utils import load_audio, save_audio
import matplotlib.pyplot as plt


def train_reverb(
    dry_audio: torch.Tensor,
    target_audio: torch.Tensor,
    sr: int,
    num_delays: int = 8,
    epochs: int = 200,
    lr: float = 0.01,
    learnable_delays: bool = True,
    learnable_feedback: bool = True,
    learnable_damping: bool = True,
    learnable_wet: bool = True,
    learnable_modulation: bool = True,
    spectral_weight: float = 1.0,
    time_weight: float = 0.1,
):
    """
    Train FDN to match target audio.

    Args:
        dry_audio: Dry input signal (no reverb)
        target_audio: Target output (with desired reverb)
        sr: Sample rate
        num_delays: Number of FDN delay lines
        epochs: Training iterations
        lr: Learning rate
        learnable_*: Which parameters to optimize
        spectral_weight: Weight for spectral loss
        time_weight: Weight for time-domain loss

    Returns:
        trained_model, loss_history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # Ensure same length
    min_len = min(dry_audio.shape[-1], target_audio.shape[-1])
    dry_audio = dry_audio[..., :min_len].to(device)
    target_audio = target_audio[..., :min_len].to(device)

    # Handle mono/stereo
    if dry_audio.dim() == 1:
        dry_audio = dry_audio.unsqueeze(0)
    if target_audio.dim() == 1:
        target_audio = target_audio.unsqueeze(0)

    # Take first channel if stereo (for simplicity)
    if dry_audio.shape[0] > 1:
        dry_audio = dry_audio[0:1]
    if target_audio.shape[0] > 1:
        target_audio = target_audio[0:1]

    print(f"Audio shape: {dry_audio.shape}, Sample rate: {sr}")

    # Initialize learnable reverb
    reverb = LearnableFDNReverb(
        sr=sr,
        num_delays=num_delays,
        learnable_delays=learnable_delays,
        learnable_feedback=learnable_feedback,
        learnable_damping=learnable_damping,
        learnable_wet=learnable_wet,
        learnable_modulation=learnable_modulation,
    ).to(device)

    print(f"\nInitial parameters:")
    for key, val in reverb.get_parameters().items():
        print(f"  {key}: {val}")

    # Setup optimizer
    optimizer = torch.optim.Adam(reverb.parameters(), lr=lr)

    # Loss function
    criterion = CombinedLoss(spectral_weight=spectral_weight, time_weight=time_weight)

    # Training loop
    loss_history = []
    spectral_history = []
    time_history = []

    print(f"\nTraining for {epochs} epochs...")
    print("-" * 60)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        output = reverb(dry_audio)

        # Compute loss
        loss_dict = criterion(output, target_audio)
        loss = loss_dict['total']

        # Backward pass
        loss.backward()
        optimizer.step()

        # Record history
        loss_history.append(loss.item())
        spectral_history.append(loss_dict['spectral'])
        time_history.append(loss_dict['time'])

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {loss.item():.6f} | "
                  f"Spectral: {loss_dict['spectral']:.6f} | "
                  f"Time: {loss_dict['time']:.6f}")

    print("-" * 60)
    print(f"\nFinal parameters:")
    for key, val in reverb.get_parameters().items():
        print(f"  {key}: {val}")

    return reverb, {
        'total': loss_history,
        'spectral': spectral_history,
        'time': time_history,
    }


def plot_training_curves(loss_history: dict, save_path: str = None):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(loss_history['total'])
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)

    axes[1].plot(loss_history['spectral'])
    axes[1].set_title('Spectral Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True)

    axes[2].plot(loss_history['time'])
    axes[2].set_title('Time Domain Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train DDSP FDN Reverb')

    # Input/output
    parser.add_argument('--dry', '-i', type=str, default='dry.mp3',
                       help='Dry input audio file')
    parser.add_argument('--target', '-t', type=str, required=True,
                       help='Target reverb audio file (what we want to match)')
    parser.add_argument('--output', '-o', type=str, default='ddsp_trained.mp3',
                       help='Output file for trained reverb')
    parser.add_argument('--model_save', type=str, default='ddsp_reverb_model.pt',
                       help='Path to save trained model')
    parser.add_argument('--plot', type=str, default='training_curves.png',
                       help='Path to save training curve plot')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--num_delays', type=int, default=8,
                       help='Number of FDN delay lines (must be power of 2)')

    # Loss weights
    parser.add_argument('--spectral_weight', type=float, default=1.0,
                       help='Weight for spectral loss')
    parser.add_argument('--time_weight', type=float, default=0.1,
                       help='Weight for time domain loss')

    # Learnable parameters
    parser.add_argument('--freeze_delays', action='store_true',
                       help='Freeze delay times (not learnable)')
    parser.add_argument('--freeze_feedback', action='store_true',
                       help='Freeze feedback gain (not learnable)')
    parser.add_argument('--freeze_damping', action='store_true',
                       help='Freeze damping (not learnable)')
    parser.add_argument('--freeze_wet', action='store_true',
                       help='Freeze wet mix (not learnable)')
    parser.add_argument('--freeze_modulation', action='store_true',
                       help='Freeze modulation (not learnable)')

    args = parser.parse_args()

    # Load audio
    print(f"Loading dry audio: {args.dry}")
    dry_audio, sr = load_audio(args.dry, mono=True, target_sample_rate=None)

    print(f"Loading target audio: {args.target}")
    target_audio, target_sr = load_audio(args.target, mono=True, target_sample_rate=sr)

    if target_sr != sr:
        print(f"Warning: Sample rates don't match ({sr} vs {target_sr}), resampling target to {sr}")

    # Train
    trained_reverb, loss_history = train_reverb(
        dry_audio=dry_audio,
        target_audio=target_audio,
        sr=sr,
        num_delays=args.num_delays,
        epochs=args.epochs,
        lr=args.lr,
        learnable_delays=not args.freeze_delays,
        learnable_feedback=not args.freeze_feedback,
        learnable_damping=not args.freeze_damping,
        learnable_wet=not args.freeze_wet,
        learnable_modulation=not args.freeze_modulation,
        spectral_weight=args.spectral_weight,
        time_weight=args.time_weight,
    )

    # Generate output with trained reverb
    print(f"\nGenerating output audio...")
    trained_reverb.eval()
    with torch.no_grad():
        output_audio = trained_reverb(dry_audio.to(next(trained_reverb.parameters()).device))

    # Save output
    output_audio = output_audio.cpu()
    if output_audio.dim() == 1:
        output_audio = output_audio.unsqueeze(0)

    save_audio(output_audio, sample_rate=sr, file_path=args.output)
    print(f"Output saved to: {args.output}")

    # Save model
    torch.save({
        'model_state_dict': trained_reverb.state_dict(),
        'parameters': trained_reverb.get_parameters(),
        'sr': sr,
        'num_delays': args.num_delays,
    }, args.model_save)
    print(f"Model saved to: {args.model_save}")

    # Plot training curves
    if args.plot:
        plot_training_curves(loss_history, save_path=args.plot)

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
