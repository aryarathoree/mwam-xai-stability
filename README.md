# Robust XAI Defense for IoT Time-Series Anomaly Detection Under Sensor Perturbations

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of the experimental framework and evaluation metrics ($J_k$, $E_{\text{sens}}$) presented in the paper *"Robust Explanation Frameworks for IoT Time-Series Anomaly Detection under Operational Perturbations"*.

---

## Abstract

Machine learning models deployed in Internet of Things (IoT) environments face severe out-of-distribution (OOD) perturbations such as linear sensor drift, Gaussian noise, and packet loss. While post-hoc explainability methods (e.g., SHAP, LIME) are standard for interpreting anomaly detection decisions, their attributions degrade under operational noise. This repository provides an end-to-end framework to evaluate and quantify attribution stability collapse under sensor perturbations.

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
