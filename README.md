# C3NN-SBI: Physics-Informed Simulation-Based Inference with Interpretable NPCF Summaries

[![arXiv:2402.09526](https://img.shields.io/badge/arXiv-2402.09526-b31b1b.svg)](https://arxiv.org/abs/2402.09526)
[![arXiv:2602.16768](https://img.shields.io/badge/arXiv-2602.16768-b31b1b.svg)](https://arxiv.org/abs/2602.16768)

This repository contains code for **C3NN-SBI**, a simulation-based inference pipeline built on the Cosmological Correlator Convolutional Neural Network (C3NN). The learned summaries have an explicit correspondence to different orders of cosmological N-point correlation functions (NPCFs), combining ML efficiency with physical interpretability.

The current project paper is available on arXiv: https://arxiv.org/abs/2602.16768

## Architecture

The model architecture used in the original C3NN paper is shown below:

![C3NN architecture](repo_images/ccnn_architecture.png)

The model has two conceptual parts:

1. A CNN-based N-point moment map generator that builds moment maps recursively from an initial convolution.
2. A downstream network (classification or inference) that consumes these summaries.

## Interpretability

C3NN interpretability mainly comes from:

1. The output moment of order `N` can be linked analytically to an `N`-point correlation function.
2. The contribution of different orders can be compared explicitly in downstream tasks.
3. Learned filter weights can be inspected to understand which configurations drive performance.

## Citation Request

If you use this codebase, please cite the following papers:

1. https://arxiv.org/abs/2011.03474
2. https://arxiv.org/abs/2402.09526
3. https://arxiv.org/abs/2602.16768

## Usage

Set hyperparameters in [`hyperparameters.py`](hyperparameters.py).

1. Run [`main.py`](main.py) when you want to train/evaluate a **single** configuration.
2. Run [`hyperparams_optimization.py`](hyperparams_optimization.py) when you want to do **automated hyperparameter optimization** (Optuna loop over many trials).
