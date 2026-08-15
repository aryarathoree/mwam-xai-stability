# Robust XAI Defense for IoT Time-Series Anomaly Detection Under Sensor Perturbations

Official PyTorch implementation of the experimental framework and evaluation metrics ($J_k$, $E_{\text{sens}}$) presented in the manuscript.

---

## Abstract
Machine learning models deployed in Internet of Things (IoT) environments are vulnerable to out-of-distribution (OOD) perturbations such as linear sensor drift, Gaussian noise, and packet loss. While black-box post-hoc explainability methods (e.g., SHAP, LIME) are used to interpret anomaly detection decisions, their stability degrades severely under operational noise. This repository contains the code to reproduce attribution stability degradation curves and quantify explanation robustness.

---

## Repository Structure

```text
mwam-xai-stability/
├── src/
│   ├── dataset.py      # Synthetic IoT signal generation & perturbation functions
│   ├── models.py       # PyTorch 1D-CNN baseline architecture
│   └── evaluate.py     # Attribution extraction, Top-k Jaccard (J_k), and E_sens metrics
├── main.py             # Pipeline execution and plot generation
├── requirements.txt    # Python dependencies
└── README.md