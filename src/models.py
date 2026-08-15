import torch
import torch.nn as nn

class Sensor1DCNN(nn.Module):
    """1D-CNN baseline model for IoT anomaly detection."""
    def __init__(self, n_channels=4, window_size=128):
        super(Sensor1DCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * (window_size // 4), 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features_flat = features.view(features.size(0), -1)
        return self.classifier(features_flat)