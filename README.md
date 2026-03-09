# C3NN-SBI: Physics-Informed Simulation-Based Inference with Interpretable NPCF Summaries

[![arXiv:2402.09526](https://img.shields.io/badge/arXiv-2402.09526-b31b1b.svg)](https://arxiv.org/abs/2402.09526)
[![arXiv:2602.16768](https://img.shields.io/badge/arXiv-2602.16768-b31b1b.svg)](https://arxiv.org/abs/2602.16768)

This repository contains code for **C3NN-SBI**, a simulation-based inference pipeline that uses the Cosmological Correlator Convolutional Neural Network (C3NN) as an interpretable summary extractor for weak-lensing convergence maps.

The core idea is to learn summaries that are explicitly tied to N-point statistics while still training end-to-end with neural density estimators.

Paper (current project): **arXiv:2602.16768v1** (February 18, 2026), included as [`2602.16768v1.pdf`](2602.16768v1.pdf).

## Citation Request

If you use this codebase, please cite the following papers:

1. https://arxiv.org/abs/2011.03474
2. https://arxiv.org/abs/2402.09526
3. https://arxiv.org/abs/2602.16768

## Main Results (from the paper)

For the 3-parameter inference setting `(Omega_m, sigma_8, w_0)`:

1. Information improves monotonically from 2nd to 4th order C3NN summaries.
2. Relative improvements (single withheld realization):
   - `{c1,c2,c3}` vs `{c1,c2}`: `Omega_m +27.92%`, `sigma_8 +28.44%`, `w_0 +19.11%`, `FoM +96.59%`
   - `{c1,c2,c3,c4}` vs `{c1,c2,c3}`: `Omega_m +12.89%`, `sigma_8 +19.99%`, `w_0 +3.68%`, `FoM +34.38%`
3. Over 200 fiducial realizations, median FoM gains are:
   - `+75.30%` when adding `c3` on top of `c2`
   - `+30.69%` when adding `c4` on top of `c3`
4. Hybrid and NLE extensions:
   - Hybrid C3NN vs 2PCF: `FoM +7.29%`
   - NLE (larger simulation budget) vs NPE: `FoM +347.73%`
5. Validation with phase-reshuffled maps shows higher-order model gains collapse to the 2nd-order baseline, consistent with genuine higher-order information extraction.

## Repository Structure

- `main.py`: main C3NN + SNPE training entrypoint.
- `hyperparams_optimization.py`: Optuna loop around `main.py`.
- `main_cnn.py`: baseline CNN embedding + SNPE comparison.
- `C3NN_NLE.py`: retrain likelihood model (NLE) on learned summaries.
- `C3NN_NLE_sample_only.py`: posterior sampling/plotting from trained NLE.
- `C3NN_NLE_coverage.py`: coverage/SBC-style diagnostics.
- `src/C3NN_extensions.py`: C3NN model variants used by `main.py`.
- `src/load_data.py`: data loading helpers.
- `hyperparameters.py`: default C3NN and CNN hyperparameter settings.

## Data Assumptions

The current scripts use project-specific absolute paths (e.g. `/project/ag-weller/...`) for map data and model output.

Before running, update paths in:

- `src/load_data.py`
- `main.py`
- `main_cnn.py`
- `C3NN_NLE*.py`

## Usage Examples

### 1) Quick smoke test (short training)

```bash
python main.py --test
```

### 2) Train C3NN + SNPE (default config)

```bash
python main.py
```

### 3) Run Optuna hyperparameter scan

```bash
python hyperparams_optimization.py
```

Short scan for code-path testing:

```bash
python hyperparams_optimization.py --test
```

### 4) Train CNN baseline + SNPE

```bash
python main_cnn.py
```

### 5) Train NLE on learned summaries

```bash
python C3NN_NLE.py
```

### 6) Sample and plot posteriors from NLE outputs

```bash
python C3NN_NLE_sample_only.py
```

### 7) Coverage / SBC-like checks

```bash
python C3NN_NLE_coverage.py
```

### 8) Measure 2PCF products used for comparisons

```bash
python measure_2pcf.py
```

## Notes

1. Default model selection is configured in `hyperparameters.py` (`hparams.model`).
2. Most training scripts expect CUDA and large memory footprints.
3. Some scripts are analysis-stage and require previously saved checkpoints/chains.
