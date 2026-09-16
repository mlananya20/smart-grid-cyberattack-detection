# Privacy-Preserving Smart Grid Cyberattack Detection

**Author:** Ananya
**Department:** Computer Science and Engineering, VIT-AP University

## Overview
This project implements a privacy-preserving federated learning framework for
multi-attack detection in smart-grid smart-meter data. It combines a
Transformer-based temporal classifier with FedProx-based federated
optimization, targeting five classes: Normal, False Data Injection (FDI),
Energy Theft, Meter Tampering, and Replay.

## Dataset
State Grid Corporation of China (SGCC) Electricity Theft Detection Dataset
(daily consumption, 42,372 customers, 1,035 days). FDI, tampering, and
replay attacks are synthetically injected on top of the real theft labels.

## Models Compared
- LSTM (baseline)
- CNN-LSTM (baseline)
- Autoencoder + GAN (baseline)
- Transformer + FedAvg (baseline)
- **Transformer + FedProx (proposed)**

## Project Structure
- `data/` — raw and processed dataset files
- `preprocessing/` — cleaning, normalization, windowing
- `models/` — model architectures
- `federated/` — FedAvg and FedProx implementations
- `experiments/` — training and evaluation scripts
- `results/` — metrics, logs, plots

## Status
🚧 Implementation in progress.