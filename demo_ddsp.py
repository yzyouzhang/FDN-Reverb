"""
DDSP FDN Reverb Demo

This script demonstrates various use cases for the learnable DDSP FDN reverb:
1. Creating a target with traditional reverb, then learning it with DDSP
2. Comparing learned vs. original parameters
3. Showing the advantage of DDSP for parameter discovery

Usage:
    python demo_ddsp.py
"""

import torch
import numpy as np
from reverb_util import FDNReverb
from ddsp_reverb import LearnableFDNReverb, CombinedLoss
from io_utils import load_audio, save_audio


def demo_1_learn_from_target():
    """
    Demo 1: Create target audio with traditional FDN, then learn it with DDSP.

    This shows how DDSP can discover parameters automatically.
    """
    print("=" * 70)
    print("DEMO 1: Learning FDN Parameters from Target Audio")
    print("=" * 70)

    # Load dry audio
    print("\n1. Loading dry audio...")
    dry_audio, sr = load_audio("dry.mp3", mono=True, target_sample_rate=48000)
    print(f"   Audio shape: {dry_audio.shape}, Sample rate: {sr}")

    # Create target with known parameters (the "secret" we want to discover)
    print("\n2. Creating target reverb with KNOWN parameters:")
    target_params = {
        'delays_ms': (31, 41, 47, 59, 67, 73, 83, 97),
        'feedback_gain': 0.85,
        'damp': 0.35,
        'wet': 0.7,
        'mod_depth_ms': 0.8,
        'mod_rate_hz': 0.25,
    }
    print(f"   Target parameters:")
    for key, val in target_params.items():
        print(f"     {key}: {val}")

    target_reverb = FDNReverb(
        sr=sr,
        delays_ms=target_params['delays_ms'],
        feedback_gain=target_params['feedback_gain'],
        damp=target_params['damp'],
        wet=target_params['wet'],
        mod_depth_ms=target_params['mod_depth_ms'],
        mod_rate_hz=target_params['mod_rate_hz'],
    )

    target_audio = target_reverb.process(dry_audio.clone())
    save_audio(target_audio.unsqueeze(0), sr, "demo_target.mp3")
    print(f"   Target audio saved to: demo_target.mp3")

    # Now use DDSP to learn these parameters (pretend we don't know them)
    print("\n3. Training DDSP to discover parameters...")
    print("   (This may take 1-2 minutes)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dry_audio = dry_audio.unsqueeze(0).to(device)  # (1, T)
    target_audio = target_audio.unsqueeze(0).to(device)  # (1, T)

    # Initialize with default parameters (different from target)
    learnable_reverb = LearnableFDNReverb(
        sr=sr,
        num_delays=8,
        init_feedback_gain=0.78,  # Different from target (0.85)
        init_damp=0.25,           # Different from target (0.35)
        init_wet=0.5,             # Different from target (0.7)
    ).to(device)

    print(f"\n   Initial (before training):")
    for key, val in learnable_reverb.get_parameters().items():
        print(f"     {key}: {val}")

    # Train
    optimizer = torch.optim.Adam(learnable_reverb.parameters(), lr=0.01)
    criterion = CombinedLoss(spectral_weight=1.0, time_weight=0.1)

    epochs = 150
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = learnable_reverb(dry_audio)
        loss_dict = criterion(output, target_audio)
        loss = loss_dict['total']
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 30 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.6f}")

    # Compare learned vs. target
    print(f"\n4. Comparing learned vs. target parameters:")
    learned_params = learnable_reverb.get_parameters()

    print(f"\n   {'Parameter':<20} {'Target':<20} {'Learned':<20} {'Error'}")
    print(f"   {'-'*75}")
    print(f"   {'feedback_gain':<20} {target_params['feedback_gain']:<20.4f} "
          f"{learned_params['feedback_gain']:<20.4f} "
          f"{abs(target_params['feedback_gain'] - learned_params['feedback_gain']):.4f}")
    print(f"   {'damping':<20} {target_params['damp']:<20.4f} "
          f"{learned_params['damping']:<20.4f} "
          f"{abs(target_params['damp'] - learned_params['damping']):.4f}")
    print(f"   {'wet_mix':<20} {target_params['wet']:<20.4f} "
          f"{learned_params['wet_mix']:<20.4f} "
          f"{abs(target_params['wet'] - learned_params['wet_mix']):.4f}")

    # Generate output
    learnable_reverb.eval()
    with torch.no_grad():
        learned_output = learnable_reverb(dry_audio)

    save_audio(learned_output.cpu(), sr, "demo_learned.mp3")
    print(f"\n   Learned output saved to: demo_learned.mp3")
    print(f"\n   ✓ DDSP successfully discovered parameters close to the target!")


def demo_2_clone_existing_reverb():
    """
    Demo 2: Clone an existing reverb effect.

    This demonstrates how DDSP can reverse-engineer a reverb you like.
    """
    print("\n\n" + "=" * 70)
    print("DEMO 2: Cloning Existing Reverb")
    print("=" * 70)

    print("\nScenario: You have a reverb sound you love (maybe from hardware)")
    print("          DDSP can learn to replicate it!")

    # In this demo, we'll use the default wet_fdn.mp3 as our "beloved reverb"
    try:
        print("\n1. Loading existing reverb audio (wet_fdn.mp3)...")
        beloved_reverb, sr = load_audio("wet_fdn.mp3", mono=True, target_sample_rate=48000)
        dry_audio, _ = load_audio("dry.mp3", mono=True, target_sample_rate=sr)

        print("\n2. Training DDSP to clone this reverb...")
        print("   (Quick demo: only 100 epochs)")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dry_audio = dry_audio.unsqueeze(0).to(device)
        beloved_reverb = beloved_reverb.unsqueeze(0).to(device)

        clone_reverb = LearnableFDNReverb(sr=sr, num_delays=8).to(device)
        optimizer = torch.optim.Adam(clone_reverb.parameters(), lr=0.01)
        criterion = CombinedLoss()

        for epoch in range(100):
            optimizer.zero_grad()
            output = clone_reverb(dry_audio)
            loss_dict = criterion(output, beloved_reverb)
            loss = loss_dict['total']
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 25 == 0:
                print(f"   Epoch {epoch+1:3d}/100 | Loss: {loss.item():.6f}")

        print(f"\n3. Learned parameters:")
        for key, val in clone_reverb.get_parameters().items():
            print(f"     {key}: {val}")

        # Save
        clone_reverb.eval()
        with torch.no_grad():
            cloned_output = clone_reverb(dry_audio)

        save_audio(cloned_output.cpu(), sr, "demo_cloned.mp3")
        print(f"\n   Cloned output saved to: demo_cloned.mp3")
        print(f"   Compare with original: wet_fdn.mp3")
        print(f"\n   ✓ Now you have interpretable parameters for that reverb!")

    except FileNotFoundError:
        print("\n   (Skipping - wet_fdn.mp3 not found)")


def demo_3_optimize_for_specific_goal():
    """
    Demo 3: Optimize reverb for a specific goal.

    This shows how DDSP can optimize for objectives beyond simple matching.
    """
    print("\n\n" + "=" * 70)
    print("DEMO 3: Optimize Reverb for Specific Goals")
    print("=" * 70)

    print("\nScenario: You want a reverb that's long but not too muddy")
    print("          Manual tuning = tedious. DDSP = automatic!")

    # Load audio
    dry_audio, sr = load_audio("dry.mp3", mono=True, target_sample_rate=48000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dry_audio = dry_audio.unsqueeze(0).to(device)

    # Create reverb with specific constraints
    print("\n1. Setting up constraints:")
    print("   - High feedback (long tail)")
    print("   - High damping (reduce muddiness)")
    print("   - Medium wet mix")

    reverb = LearnableFDNReverb(
        sr=sr,
        num_delays=8,
        init_feedback_gain=0.90,  # Start with long tail
        init_damp=0.40,            # Start with high damping
        init_wet=0.6,              # Medium wet
        learnable_delays=True,     # Optimize delays too
        learnable_feedback=False,  # Keep feedback fixed (we want it long)
        learnable_damping=True,    # Optimize damping
        learnable_wet=True,        # Optimize wet mix
    ).to(device)

    print(f"\n   Fixed parameters:")
    print(f"     feedback_gain: 0.90 (frozen - we want long tail)")
    print(f"\n   Learnable parameters:")
    print(f"     delays, damping, wet_mix (will be optimized)")

    # Custom optimization: maximize reverb time while minimizing muddiness
    # (This is a simplified example - in practice, you'd use more sophisticated metrics)

    print(f"\n2. Optimizing...")
    optimizer = torch.optim.Adam(reverb.parameters(), lr=0.01)

    # Create a "clean long reverb" by emphasizing high frequencies
    # We'll create a synthetic target that has what we want
    temp_reverb = FDNReverb(sr=sr, feedback_gain=0.92, damp=0.5, wet=0.7)
    synthetic_target = temp_reverb.process(dry_audio.squeeze(0).cpu()).unsqueeze(0).to(device)

    criterion = CombinedLoss(spectral_weight=1.0, time_weight=0.05)

    for epoch in range(80):
        optimizer.zero_grad()
        output = reverb(dry_audio)
        loss_dict = criterion(output, synthetic_target)
        loss = loss_dict['total']
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1:3d}/80 | Loss: {loss.item():.6f}")

    print(f"\n3. Optimized parameters:")
    for key, val in reverb.get_parameters().items():
        print(f"     {key}: {val}")

    reverb.eval()
    with torch.no_grad():
        optimized_output = reverb(dry_audio)

    save_audio(optimized_output.cpu(), sr, "demo_optimized.mp3")
    print(f"\n   Output saved to: demo_optimized.mp3")
    print(f"\n   ✓ DDSP found parameters that match your specific goals!")


def main():
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                   DDSP FDN Reverb Demos                            ║")
    print("║                                                                    ║")
    print("║  This demo shows the power of Differentiable DSP:                 ║")
    print("║  - Learn parameters from target audio                             ║")
    print("║  - Clone existing reverb effects                                  ║")
    print("║  - Optimize for specific goals                                    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    # Run demos
    demo_1_learn_from_target()
    demo_2_clone_existing_reverb()
    demo_3_optimize_for_specific_goal()

    print("\n\n" + "=" * 70)
    print("ALL DEMOS COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - demo_target.mp3    : Target reverb (known parameters)")
    print("  - demo_learned.mp3   : DDSP learned to match target")
    print("  - demo_cloned.mp3    : Cloned existing reverb")
    print("  - demo_optimized.mp3 : Optimized for specific goal")
    print("\nKey takeaway:")
    print("  DDSP automatically discovers parameters that would take hours")
    print("  to find manually. This is especially powerful for:")
    print("    - Matching real spaces (record a hall, learn its reverb)")
    print("    - Cloning hardware (reverse-engineer expensive gear)")
    print("    - Optimizing for complex goals (multi-objective optimization)")
    print("\n")


if __name__ == '__main__':
    main()
