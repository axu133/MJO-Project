# MJO-Project
MJO Project for Lu Group

Project began June 2025

## Overview
Source code for a deep learning framework designed to forecast the Madden-Julian Oscillation (MJO) using multimodal atmospheric data. The model utilizes attention-based architectures and operator learning to predict weather patterns up to 40 days in advance.

## Architectures

### Multimodal Vision Transformer (ViT)
* **Conditioning Mechanism:** Implements **FiLM (Feature-wise Linear Modulation)** to dynamically modulate attention weights based on auxiliary scalar inputs.
* **Forecasting Tasks:**
    * **Index Prediction (Regression):** Predicts MJO indices with **0.70 correlation** at a 15-day lead time (Benchmark: Shin et al., 2021).
    * **Spatio-Temporal Forecasting (Img2Img):** Supports autoregressive image-to-image prediction to generate future states (t+2 days) from current atmospheric snapshots.

### Fourier Neural Operator (FNO)
* *Status: Under active development.*
* Implementation of resolution-invariant operator learning for PDE-based weather modeling.

## Scalability & Infrastructure
Current experiments utilize a reduced dataset with an in-memory pipeline to maximize iteration speed. High-dimensional optimization was validated in previous iterations:

* **Memory Management:** Custom PyTorch Dataset implementation supports lazy loading to decouple RAM usage from dataset size.
* **Performance:** Benchmarks demonstrated a **>50% reduction** in peak VRAM footprint, enabling training on datasets 5x larger than standard approaches.
