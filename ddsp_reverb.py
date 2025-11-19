"""
Differentiable FDN Reverb - DDSP Version

This module implements a learnable version of FDN reverb where parameters
can be optimized via gradient descent to match target reverb characteristics.

Key differences from reverb_util.py:
- Parameters are torch.nn.Parameter (learnable)
- All operations support backpropagation
- No C++ extensions (pure PyTorch for differentiability)
- Designed for training, not real-time processing
"""

import torch
import torch.nn as nn
import numpy as np


def hadamard_orthogonal(n: int) -> torch.Tensor:
    """Generate an n x n Hadamard orthogonal matrix (n must be a power of 2)."""
    if n & (n - 1) != 0:
        raise ValueError("Hadamard size must be power of 2.")

    H = torch.ones(1, 1, dtype=torch.float32)
    current_size = 1
    while current_size < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)
        current_size *= 2

    return H[:n, :n] / torch.sqrt(torch.tensor(n, dtype=torch.float32))


class LearnableFDNReverb(nn.Module):
    """
    Learnable FDN Reverb - All parameters are optimizable via gradient descent.

    This is a DDSP implementation where you can train the reverb to match
    target audio characteristics.

    Learnable parameters:
    - delay_times: Delay line lengths (constrained to valid range)
    - feedback_gain: Overall feedback strength
    - damping: High-frequency damping factor
    - wet_mix: Wet/dry ratio
    - mod_depth: LFO modulation depth
    - mod_rate: LFO modulation rate

    Example usage:
        # Create learnable reverb
        reverb = LearnableFDNReverb(sr=48000, num_delays=8)

        # Setup optimizer
        optimizer = torch.optim.Adam(reverb.parameters(), lr=0.01)

        # Training loop
        for epoch in range(100):
            output = reverb(dry_audio)
            loss = loss_function(output, target_audio)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    """

    def __init__(
        self,
        sr: int,
        num_delays: int = 8,
        init_delays_ms: tuple = None,
        init_feedback_gain: float = 0.78,
        init_damp: float = 0.3,
        init_wet: float = 0.25,
        init_mod_depth_ms: float = 1.2,
        init_mod_rate_hz: float = 0.2,
        learnable_delays: bool = True,
        learnable_feedback: bool = True,
        learnable_damping: bool = True,
        learnable_wet: bool = True,
        learnable_modulation: bool = True,
    ):
        """
        Args:
            sr: Sample rate
            num_delays: Number of delay lines (must be power of 2)
            init_delays_ms: Initial delay times in ms (default: coprime values)
            init_feedback_gain: Initial feedback gain (0-1)
            init_damp: Initial damping factor (0-1)
            init_wet: Initial wet/dry mix (0-1)
            init_mod_depth_ms: Initial modulation depth in ms
            init_mod_rate_hz: Initial modulation rate in Hz
            learnable_*: Whether each parameter type should be learnable
        """
        super().__init__()

        self.sr = sr
        self.num_delays = num_delays

        # Validate num_delays is power of 2
        if num_delays & (num_delays - 1) != 0:
            raise ValueError(f"num_delays must be power of 2, got {num_delays}")

        # Initialize delay times
        if init_delays_ms is None:
            # Default coprime delays
            default_delays = {
                2: (29, 37),
                4: (29, 37, 43, 53),
                8: (29, 37, 43, 53, 61, 71, 79, 89),
                16: (19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83),
            }
            init_delays_ms = default_delays.get(num_delays, tuple(range(20, 20 + num_delays * 10, 10)))

        # Convert delays to samples (learnable in log space for stability)
        delays_samples = torch.tensor([sr * d / 1000.0 for d in init_delays_ms], dtype=torch.float32)
        if learnable_delays:
            # Store in log space for better optimization (prevents negative delays)
            self.log_delays = nn.Parameter(torch.log(delays_samples))
        else:
            self.register_buffer('log_delays', torch.log(delays_samples))

        # Feedback gain (learnable, constrained to (0, 1) via sigmoid)
        feedback_init = torch.tensor(init_feedback_gain, dtype=torch.float32)
        # Store in logit space
        feedback_logit = torch.logit(feedback_init.clamp(0.01, 0.99))
        if learnable_feedback:
            self.feedback_logit = nn.Parameter(feedback_logit)
        else:
            self.register_buffer('feedback_logit', feedback_logit)

        # Damping (learnable, constrained to (0, 1) via sigmoid)
        damp_init = torch.tensor(init_damp, dtype=torch.float32)
        damp_logit = torch.logit(damp_init.clamp(0.01, 0.99))
        if learnable_damping:
            self.damp_logit = nn.Parameter(damp_logit)
        else:
            self.register_buffer('damp_logit', damp_logit)

        # Wet/dry mix (learnable, constrained to (0, 1) via sigmoid)
        wet_init = torch.tensor(init_wet, dtype=torch.float32)
        wet_logit = torch.logit(wet_init.clamp(0.01, 0.99))
        if learnable_wet:
            self.wet_logit = nn.Parameter(wet_logit)
        else:
            self.register_buffer('wet_logit', wet_logit)

        # Modulation parameters (learnable, positive via softplus)
        mod_depth_init = torch.tensor(init_mod_depth_ms * sr / 1000.0, dtype=torch.float32)
        mod_rate_init = torch.tensor(init_mod_rate_hz, dtype=torch.float32)

        if learnable_modulation:
            # Store in log space for positivity
            self.log_mod_depth = nn.Parameter(torch.log(mod_depth_init + 1e-6))
            self.log_mod_rate = nn.Parameter(torch.log(mod_rate_init + 1e-6))
        else:
            self.register_buffer('log_mod_depth', torch.log(mod_depth_init + 1e-6))
            self.register_buffer('log_mod_rate', torch.log(mod_rate_init + 1e-6))

        # Random initial phases for modulation (not learnable)
        self.register_buffer('mod_phase', 2 * np.pi * torch.rand(num_delays, dtype=torch.float32))

        # Hadamard feedback matrix (not learnable - orthogonality is critical)
        H = hadamard_orthogonal(num_delays)
        self.register_buffer('hadamard_matrix', H)

    def get_parameters(self):
        """Get current parameter values (human-readable)."""
        delays_samples = torch.exp(self.log_delays)
        delays_ms = delays_samples * 1000.0 / self.sr
        feedback = torch.sigmoid(self.feedback_logit)
        damp = torch.sigmoid(self.damp_logit)
        wet = torch.sigmoid(self.wet_logit)
        mod_depth_samples = torch.exp(self.log_mod_depth)
        mod_depth_ms = mod_depth_samples * 1000.0 / self.sr
        mod_rate = torch.exp(self.log_mod_rate)

        return {
            'delays_ms': delays_ms.detach().cpu().numpy(),
            'feedback_gain': feedback.item(),
            'damping': damp.item(),
            'wet_mix': wet.item(),
            'mod_depth_ms': mod_depth_ms.item(),
            'mod_rate_hz': mod_rate.item(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process audio through learnable FDN.

        Args:
            x: Input audio tensor
               - 1D: (T,) mono
               - 2D: (C, T) multi-channel

        Returns:
            Output audio tensor (same shape as input)
        """
        # Handle input dimensions
        original_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, T)
            is_mono = True
        else:
            is_mono = False

        device = x.device
        num_channels, length = x.shape

        # Get constrained parameters
        delays = torch.exp(self.log_delays).to(device)  # (N,)
        feedback_gain = torch.sigmoid(self.feedback_logit).to(device)
        damp = torch.sigmoid(self.damp_logit).to(device)
        wet = torch.sigmoid(self.wet_logit).to(device)
        mod_depth = torch.exp(self.log_mod_depth).to(device)
        mod_rate = torch.exp(self.log_mod_rate).to(device)

        # Feedback matrix
        A = feedback_gain * self.hadamard_matrix.to(device)  # (N, N)

        # Maximum delay for buffer size
        max_delay = int(delays.max().item() + mod_depth.item() + 100)

        # Initialize buffers
        buffers = torch.zeros(num_channels, self.num_delays, max_delay,
                            dtype=torch.float32, device=device)
        write_idx = torch.zeros(num_channels, self.num_delays,
                              dtype=torch.long, device=device)
        lp_state = torch.zeros(num_channels, self.num_delays,
                             dtype=torch.float32, device=device)

        # Pre-compute modulation (LFO)
        t = torch.arange(length, dtype=torch.float32, device=device) / self.sr
        phase_matrix = (2 * np.pi * mod_rate * t.unsqueeze(1) +
                       self.mod_phase.to(device).unsqueeze(0))
        lfo_all = mod_depth * torch.sin(phase_matrix)  # (T, N)

        # Pre-compute fractional delays
        frac_delays_all = delays.unsqueeze(0) + lfo_all  # (T, N)
        frac_delays_all = torch.clamp(frac_delays_all, 1.0, float(max_delay - 2))

        # Process sample by sample (required for feedback)
        y_out = torch.zeros_like(x)

        for n in range(length):
            frac_delays = frac_delays_all[n]  # (N,)

            # Calculate read positions
            read_pos = (write_idx.float() - frac_delays.unsqueeze(0)) % max_delay
            read_pos = torch.where(read_pos < 0.0, read_pos + max_delay, read_pos)

            # Linear interpolation
            i0 = read_pos.floor().long() % max_delay
            i1 = (i0 + 1) % max_delay
            frac = read_pos - read_pos.floor()

            # Read from buffers
            y0 = torch.gather(buffers, 2, i0.unsqueeze(-1)).squeeze(-1)
            y1 = torch.gather(buffers, 2, i1.unsqueeze(-1)).squeeze(-1)
            y_vec = torch.lerp(y0, y1, frac)  # (C, N)

            # Feedback mixing
            fb = torch.einsum('ij,cj->ci', A, y_vec)  # (C, N)

            # Damping (one-pole lowpass)
            lp_state = lp_state * (1.0 - damp) + fb * damp
            fb_damped = lp_state

            # Input distribution
            in_vec = x[:, n].unsqueeze(1) / self.num_delays  # (C, 1)

            # Write to buffers
            write_pos = (write_idx % max_delay).unsqueeze(-1)
            buffers.scatter_(2, write_pos, (fb_damped + in_vec).unsqueeze(-1))
            write_idx = (write_idx + 1) % max_delay

            # Output (wet/dry mix)
            wet_sample = y_vec.mean(dim=1)  # (C,)
            y_out[:, n] = x[:, n] * (1.0 - wet) + wet_sample * wet

        # Restore original shape
        if is_mono:
            y_out = y_out.squeeze(0)

        return y_out


class SpectralLoss(nn.Module):
    """
    Multi-scale spectral loss for audio.

    Compares magnitude spectrograms at multiple FFT sizes.
    Commonly used in audio DDSP for perceptually-relevant training.
    """

    def __init__(self, fft_sizes=[512, 1024, 2048], loss_type='L1'):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.loss_type = loss_type

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral loss between predicted and target audio.

        Args:
            pred: Predicted audio (B, T) or (T,)
            target: Target audio (same shape)

        Returns:
            Scalar loss value
        """
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        total_loss = 0.0

        for fft_size in self.fft_sizes:
            # Compute STFT
            pred_stft = torch.stft(pred, n_fft=fft_size, hop_length=fft_size//4,
                                  window=torch.hann_window(fft_size, device=pred.device),
                                  return_complex=True)
            target_stft = torch.stft(target, n_fft=fft_size, hop_length=fft_size//4,
                                    window=torch.hann_window(fft_size, device=target.device),
                                    return_complex=True)

            # Magnitude
            pred_mag = torch.abs(pred_stft)
            target_mag = torch.abs(target_stft)

            # Loss
            if self.loss_type == 'L1':
                loss = torch.mean(torch.abs(pred_mag - target_mag))
            else:  # L2
                loss = torch.mean((pred_mag - target_mag) ** 2)

            total_loss += loss

        return total_loss / len(self.fft_sizes)


class TimeDomainLoss(nn.Module):
    """Simple time-domain loss (L1 or L2)."""

    def __init__(self, loss_type='L1'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'L1':
            return torch.mean(torch.abs(pred - target))
        else:
            return torch.mean((pred - target) ** 2)


class CombinedLoss(nn.Module):
    """
    Combined time + spectral loss for audio DDSP training.

    This is a common approach in DDSP papers.
    """

    def __init__(self, spectral_weight=1.0, time_weight=1.0):
        super().__init__()
        self.spectral_loss = SpectralLoss()
        self.time_loss = TimeDomainLoss()
        self.spectral_weight = spectral_weight
        self.time_weight = time_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        spectral = self.spectral_loss(pred, target)
        time = self.time_loss(pred, target)
        total = self.spectral_weight * spectral + self.time_weight * time

        return {
            'total': total,
            'spectral': spectral.item(),
            'time': time.item(),
        }
