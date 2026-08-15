import numpy as np
import torch

def generate_iot_data(n_samples=800, window_size=128, n_channels=4, seed=42):
    """Generates synthetic multi-channel IoT sensor signals with transient anomalies."""
    np.random.seed(seed)
    t = np.linspace(0, 4 * np.pi, window_size)
    X, y = [], []
    
    for _ in range(n_samples):
        label = np.random.choice([0, 1])
        channels = []
        for c in range(n_channels):
            base_freq = 1.0 + 0.2 * c
            signal = np.sin(base_freq * t) + 0.05 * np.random.randn(window_size)
            if label == 1 and c == 0:
                start_idx = 50
                signal[start_idx:start_idx+15] += 1.2 * np.sin(5 * t[start_idx:start_idx+15])
            channels.append(signal)
        X.append(channels)
        y.append(label)
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def apply_ood_perturbation(X_samples, alpha, mode="drift"):
    """Applies OOD perturbation shifts (Linear Drift, Gaussian Noise, or Packet Loss)."""
    N, C, T = X_samples.shape
    X_pert = X_samples.copy()
    
    if mode == "drift":
        drift = alpha * (np.linspace(0, 1, T) ** 2)
        X_pert += drift
    elif mode == "noise":
        noise = np.random.normal(0, alpha, (N, C, T))
        X_pert += noise
    elif mode == "packet_loss":
        mask = np.random.choice([1, 0], size=(N, C, T), p=[1 - alpha, alpha])
        X_pert *= mask
        
    return X_pert