import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from src.dataset import generate_iot_data, apply_ood_perturbation
from src.models import Sensor1DCNN
from src.evaluate import compute_attribution, compute_jaccard, compute_esens

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Dataset generation
    X_raw, y_raw = generate_iot_data()
    split = int(0.8 * len(X_raw))
    X_train, X_test = X_raw[:split], X_raw[split:]
    y_train, y_test = y_raw[:split], y_raw[split:]

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=32, shuffle=True
    )

    # 2. Model Training
    model = Sensor1DCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    model.eval()
    print("Training Complete. Clean Test Accuracy:", 
          (model(torch.tensor(X_test)).argmax(dim=1) == torch.tensor(y_test)).float().mean().item())

    # 3. Attribution Robustness Evaluation
    anomaly_idx = np.where(y_test == 1)[0][:10]
    X_anomaly = X_test[anomaly_idx]
    
    attr_clean = compute_attribution(model, torch.tensor(X_anomaly, dtype=torch.float32))
    
    alphas = [0.0, 0.2, 0.5, 0.8, 1.2]
    jaccard_results = []
    
    for alpha in alphas:
        if alpha == 0.0:
            attr_shifted = attr_clean
        else:
            X_shifted = apply_ood_perturbation(X_anomaly, alpha, mode="drift")
            attr_shifted = compute_attribution(model, torch.tensor(X_shifted, dtype=torch.float32))
        
        j_score = np.mean([compute_jaccard(attr_clean[i], attr_shifted[i], k=15) for i in range(len(X_anomaly))])
        jaccard_results.append(j_score)
        print(f"Alpha={alpha:.1f} | Jaccard Similarity J_15: {j_score:.4f}")

    # 4. Save Plot
    plt.figure(figsize=(7, 4))
    plt.plot(alphas, jaccard_results, marker='o', color='#d95f02', linewidth=2)
    plt.title("Attribution Stability Collapse under Sensor Drift", fontsize=11, fontweight='bold')
    plt.xlabel("Sensor Drift Intensity (Alpha)")
    plt.ylabel("Top-k Jaccard Similarity ($J_k$)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("shap_degradation_curve.png", dpi=300)
    print("Plot saved as shap_degradation_curve.png")

if __name__ == "__main__":
    main()