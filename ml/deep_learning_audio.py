"""Optional deep-learning extension for communication-signal classification.

This module is intentionally isolated from the core app so the application
runs without PyTorch. For research, feed consented audio clips through
librosa to obtain log-mel spectrograms and train a small CNN classifier.
Do not use the model to infer protected traits, personality, or mental state.
"""
try:
    import torch
    import torch.nn as nn
except ImportError:  # optional dependency
    torch = None
    nn = None


if nn is not None:
    class AudioSignalCNN(nn.Module):
        def __init__(self, classes: int = 3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = nn.Linear(32, classes)

        def forward(self, x):
            x = self.net(x).flatten(1)
            return self.head(x)
else:
    AudioSignalCNN = None
