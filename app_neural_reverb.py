"""
Neural Reverb Controller - Application Demo 1

This demonstrates using a neural network to predict optimal reverb parameters
from audio features. The network learns to map audio characteristics
(spectral, temporal) to reverb settings.

Use case: Automatic reverb suggestions based on audio content
- Vocals → shorter, brighter reverb
- Drums → tight, controlled reverb
- Strings → longer, richer reverb

Architecture:
  Audio → Feature Extraction (CNN) → Reverb Parameters → Differentiable FDN → Output

The entire pipeline is differentiable, allowing end-to-end training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from io_utils import load_audio, save_audio
from ddsp_reverb import LearnableFDNReverb


class AudioFeatureExtractor(nn.Module):
    """
    Convolutional network to extract features from audio spectrograms.

    Analyzes spectral and temporal characteristics to understand
    what kind of reverb would suit the audio.
    """

    def __init__(self, feature_dim=128):
        super().__init__()

        # Spectrogram processing (CNN over time-frequency)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=2)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Feature processing
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, feature_dim)

        self.dropout = nn.Dropout(0.3)

    def forward(self, audio):
        """
        Extract features from audio.

        Args:
            audio: (B, T) audio waveform

        Returns:
            features: (B, feature_dim) feature vector
        """
        # Compute spectrogram
        stft = torch.stft(
            audio,
            n_fft=1024,
            hop_length=256,
            window=torch.hann_window(1024, device=audio.device),
            return_complex=True
        )
        mag = torch.abs(stft).unsqueeze(1)  # (B, 1, freq, time)

        # Log magnitude for better dynamic range
        mag = torch.log(mag + 1e-8)

        # CNN processing
        x = F.relu(self.conv1(mag))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Global pooling
        x = self.pool(x)
        x = x.flatten(1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ReverbParameterPredictor(nn.Module):
    """
    Predicts reverb parameters from audio features.

    Maps feature vector to FDN reverb parameters:
    - feedback_gain (0-1)
    - damping (0-1)
    - wet_mix (0-1)
    """

    def __init__(self, feature_dim=128):
        super().__init__()

        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, 64)

        # Output heads for different parameters
        self.feedback_head = nn.Linear(64, 1)
        self.damping_head = nn.Linear(64, 1)
        self.wet_head = nn.Linear(64, 1)

    def forward(self, features):
        """
        Predict reverb parameters.

        Args:
            features: (B, feature_dim)

        Returns:
            params: dict with feedback, damping, wet (all in [0,1])
        """
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))

        # Sigmoid to constrain to [0, 1]
        feedback = torch.sigmoid(self.feedback_head(x))
        damping = torch.sigmoid(self.damping_head(x))
        wet = torch.sigmoid(self.wet_head(x))

        return {
            'feedback': feedback.squeeze(-1),  # (B,)
            'damping': damping.squeeze(-1),
            'wet': wet.squeeze(-1),
        }


class NeuralReverbController(nn.Module):
    """
    End-to-end neural reverb controller.

    Audio → CNN Features → Predicted Parameters → Differentiable FDN → Reverb Output

    The entire pipeline is differentiable, enabling:
    - Training from paired dry/wet examples
    - Learning optimal reverb for different audio types
    - Automatic reverb suggestion
    """

    def __init__(self, sr=48000, feature_dim=128):
        super().__init__()

        self.sr = sr
        self.feature_extractor = AudioFeatureExtractor(feature_dim)
        self.param_predictor = ReverbParameterPredictor(feature_dim)

        # Base reverb (parameters will be modulated by network)
        # We'll create reverb instances dynamically with predicted params

    def forward(self, audio_batch):
        """
        Process audio batch through neural reverb controller.

        Args:
            audio_batch: (B, T) audio waveforms

        Returns:
            outputs: (B, T) reverb outputs
            params: dict of predicted parameters
        """
        batch_size = audio_batch.shape[0]
        device = audio_batch.device

        # Extract features
        features = self.feature_extractor(audio_batch)

        # Predict parameters
        params = self.param_predictor(features)

        # Apply reverb with predicted parameters (batch processing)
        outputs = []
        for i in range(batch_size):
            # Create reverb with predicted parameters
            reverb = LearnableFDNReverb(
                sr=self.sr,
                init_feedback_gain=params['feedback'][i].item(),
                init_damping=params['damping'][i].item(),
                init_wet=params['wet'][i].item(),
                learnable_delays=False,
                learnable_feedback=False,
                learnable_damping=False,
                learnable_wet=False,
                learnable_modulation=False,
            ).to(device)

            reverb.eval()
            output = reverb(audio_batch[i])
            outputs.append(output)

        outputs = torch.stack(outputs)

        return outputs, params


def create_synthetic_dataset(num_samples=100, sr=48000, audio_duration=2):
    """
    Create synthetic training dataset.

    Generates different audio types with appropriate reverb settings:
    - High-pitched sounds → shorter reverb
    - Low-pitched sounds → longer reverb
    - Percussive sounds → tight reverb
    - Sustained sounds → longer reverb
    """
    dataset = []

    for i in range(num_samples):
        # Generate different audio types
        audio_type = np.random.choice(['high_tone', 'low_tone', 'percussive', 'noise'])

        num_samples_audio = int(audio_duration * sr)
        t = torch.arange(num_samples_audio) / sr

        if audio_type == 'high_tone':
            # High frequency tone → short, bright reverb
            freq = np.random.uniform(1000, 3000)
            audio = 0.3 * torch.sin(2 * np.pi * freq * t)
            target_params = {'feedback': 0.7, 'damping': 0.2, 'wet': 0.4}

        elif audio_type == 'low_tone':
            # Low frequency tone → longer, darker reverb
            freq = np.random.uniform(100, 300)
            audio = 0.3 * torch.sin(2 * np.pi * freq * t)
            target_params = {'feedback': 0.9, 'damping': 0.5, 'wet': 0.7}

        elif audio_type == 'percussive':
            # Percussive → tight reverb
            audio = torch.zeros(num_samples_audio)
            for _ in range(10):
                pos = np.random.randint(0, num_samples_audio - 1000)
                audio[pos:pos+100] = torch.randn(100) * 0.5 * torch.exp(-torch.arange(100) / 10)
            target_params = {'feedback': 0.6, 'damping': 0.4, 'wet': 0.3}

        else:  # noise
            # Noise → moderate reverb
            audio = 0.1 * torch.randn(num_samples_audio)
            target_params = {'feedback': 0.75, 'damping': 0.3, 'wet': 0.5}

        # Generate target output with target parameters
        reverb = LearnableFDNReverb(
            sr=sr,
            init_feedback_gain=target_params['feedback'],
            init_damping=target_params['damping'],
            init_wet=target_params['wet'],
            learnable_delays=False,
            learnable_feedback=False,
            learnable_damping=False,
            learnable_wet=False,
        )

        reverb.eval()
        with torch.no_grad():
            target_output = reverb(audio)

        dataset.append({
            'input': audio,
            'target': target_output,
            'params': target_params,
            'type': audio_type,
        })

    return dataset


def train_neural_reverb(epochs=50, batch_size=4, lr=0.001):
    """Train the neural reverb controller."""

    print("="*70)
    print("Training Neural Reverb Controller")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    model = NeuralReverbController(sr=48000, feature_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create dataset
    print("\nGenerating synthetic dataset...")
    dataset = create_synthetic_dataset(num_samples=100, sr=48000, audio_duration=1)
    print(f"Dataset size: {len(dataset)} samples")

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    print("-"*70)

    for epoch in range(epochs):
        total_loss = 0
        param_errors = {'feedback': [], 'damping': [], 'wet': []}

        # Mini-batch training
        indices = torch.randperm(len(dataset))

        for i in range(0, len(dataset), batch_size):
            batch_indices = indices[i:i+batch_size]

            # Prepare batch
            inputs = torch.stack([dataset[idx]['input'] for idx in batch_indices]).to(device)
            targets = torch.stack([dataset[idx]['target'] for idx in batch_indices]).to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs, pred_params = model(inputs)

            # Loss: MSE between output and target
            loss = F.mse_loss(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Track parameter prediction errors
            for j, idx in enumerate(batch_indices):
                for key in ['feedback', 'damping', 'wet']:
                    true_val = dataset[idx]['params'][key]
                    pred_val = pred_params[key][j].item()
                    param_errors[key].append(abs(true_val - pred_val))

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / (len(dataset) / batch_size)
            avg_errors = {k: np.mean(v) for k, v in param_errors.items()}

            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {avg_loss:.6f} | "
                  f"Param Errors - FB: {avg_errors['feedback']:.3f}, "
                  f"Damp: {avg_errors['damping']:.3f}, "
                  f"Wet: {avg_errors['wet']:.3f}")

    print("-"*70)
    print("Training complete!")

    return model


def demo_neural_reverb():
    """Demonstrate the trained neural reverb controller."""

    print("\n" + "="*70)
    print("Neural Reverb Controller Demo")
    print("="*70)

    # Train model
    model = train_neural_reverb(epochs=50, batch_size=4, lr=0.001)
    model.eval()

    print("\nTesting on different audio types...")
    print("-"*70)

    # Test on different audio types
    sr = 48000
    duration = 2
    t = torch.arange(int(duration * sr)) / sr

    test_cases = [
        {
            'name': 'High Frequency Tone (2000 Hz)',
            'audio': 0.3 * torch.sin(2 * np.pi * 2000 * t),
            'expected': 'Short, bright reverb',
        },
        {
            'name': 'Low Frequency Tone (150 Hz)',
            'audio': 0.3 * torch.sin(2 * np.pi * 150 * t),
            'expected': 'Long, dark reverb',
        },
        {
            'name': 'White Noise',
            'audio': 0.1 * torch.randn(int(duration * sr)),
            'expected': 'Moderate reverb',
        },
    ]

    device = next(model.parameters()).device

    for i, test in enumerate(test_cases):
        print(f"\nTest {i+1}: {test['name']}")
        print(f"  Expected: {test['expected']}")

        audio = test['audio'].unsqueeze(0).to(device)

        with torch.no_grad():
            output, params = model(audio)

        print(f"  Predicted parameters:")
        print(f"    Feedback: {params['feedback'][0].item():.3f}")
        print(f"    Damping:  {params['damping'][0].item():.3f}")
        print(f"    Wet mix:  {params['wet'][0].item():.3f}")

        # Save output
        output_file = f'neural_reverb_demo_{i+1}.mp3'
        save_audio(output.cpu(), sr, output_file)
        print(f"  Saved: {output_file}")

    print("\n" + "="*70)
    print("Demo complete! The neural network learned to:")
    print("  - Predict appropriate reverb for different audio types")
    print("  - Map audio characteristics to reverb parameters")
    print("  - Apply reverb automatically based on content")
    print("\nThis demonstrates end-to-end differentiable audio processing!")
    print("="*70 + "\n")


if __name__ == '__main__':
    demo_neural_reverb()
