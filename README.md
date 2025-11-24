<div align="center">

# 🏏 LLM‑Based Automatic Cricket Commentary  
Using Umpire Wearable Sensor + UWB Distance Data

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](notebooks/train_cricket_classifier.ipynb)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-WIP-orange)](#-roadmap)

<br/>
<i>From raw multisensor umpire signals ➜ event classification ➜ (planned) natural‑language commentary.</i>

</div>

---

## ✨ What This Project Does
- **Ingests** wearable IMU + UWB distance data from cricket umpires  
- **Classifies** the signal label (e.g., boundary, wide, no‑ball) with a simple 1D CNN baseline  
- **Planned**: detect events in continuous streams and **generate commentary** via an LLM  

We’ve built a clean data pipeline and a first baseline classifier with solid sanity‑check performance and low complexity.

---

## 🗂️ Table of Contents
- [✨ What This Project Does](#-what-this-project-does)
- [📦 What’s Implemented So Far](#-whats-implemented-so-far)
- [📊 Data Format (Quick Intuition)](#-data-format-quick-intuition)
- [🧪 Quickstart](#-quickstart)
- [🧱 Model Overview](#-model-overview)
- [📈 Results Snapshot](#-results-snapshot)
- [🏗️ Suggested Repository Layout](#️-suggested-repository-layout)
- [🗺️ Roadmap](#️-roadmap)

---

## 📦 What’s Implemented So Far

### 1) ViSig Data Loader
File: `src/data_loading/load_visig.py`

**Capabilities**
- Loads all `.mat` files containing umpire signal trials  
- Parses:
  - `acc_mat` – accelerometer data  
  - `gyro_mat` – gyroscope data  
  - `dist_mat` – pairwise UWB distances between sensor nodes  
  - `rawt` – timestamps  
- Normalizes shapes and converts to time‑major format  
- Extracts:
  - `label` (e.g., boundary4, wide, out, etc.)  
  - `participant_id` from filename suffix (e.g., `_1`, `_2`, …)  
- Provides:
  - `ViSigSample` dataclass to hold one trial  
  - `load_visig_mat(...)` to load a single file  
  - `load_visig_dataset(...)` to load all `.mat` under a directory  
  - `get_label_distribution(...)` for quick label counts  
  - `to_flat_sequence(...)` to convert each trial into a `(T, F)` matrix  

**Current notes (validated on our dataset)**
- Samples: **80**  
- Labels: **10** (`boundary4`, `boundary6`, `cancelcall`, `deadball`, `legbye`, `noball`, `out`, `penaltyrun`, `shortrun`, `wide`)  
- Balanced: **8 samples per class**  
- Each sample is a multivariate time series with feature dimension **F = 195**  
  - 90 from `acc_mat`, 90 from `gyro_mat`, 15 from upper‑triangular `dist_mat`

---

### 2) PyTorch Dataset Wrapper
File: `src/data_loading/cricket_dataset.py`

**Capabilities**
- `build_label_mapping(...)`: deterministic label→index mapping (lexicographically sorted)  
- `CricketSignalsDataset`: wraps a list of `ViSigSample` for PyTorch
  - Internally uses `to_flat_sequence(...)`
  - Outputs:
    - `x`: tensor of shape `(max_len, feature_dim)`
      - Center‑cropped if longer than `max_len`
      - Padded at the end with `pad_value` if shorter
    - `y`: scalar class index
  - Exposes: `num_classes`, `feature_dim`  
- `create_cricket_datasets(...)`: loads all samples, builds a shared label mapping, and splits into train/val/test via `random_split` (70% / 15% / 15% by default)

---

### 3) Baseline Sequence Classifier (1D CNN)
Files:  
`src/models/seq_cnn.py`  
`src/training/train_seq_classifier.py`

**Key ideas**
- Treat each signal as a time series (not an image)  
- 1D CNN over time:
  - Input: `(batch, seq_len, feature_dim)`
  - `Conv1d` over time to learn local temporal patterns
  - Global max pooling over time for shift invariance
  - Linear layer → class logits

**Reasoning**
- Small dataset (~80 sequences) → prefer simple, low‑parameter model  
- 1D CNNs are standard for multivariate time‑series classification  
- Intentional, interpretable baseline (not the final architecture)

**Training script**
- Loads data via `create_cricket_datasets`
- Builds `SimpleCricketCNN` with:
  - `input_dim = feature_dim`
  - `num_classes = dataset.num_classes`
- Optimizer/Loss/Early‑stopping:
  - Adam (lr = 1e‑3 default)
  - Cross‑entropy
  - Early stopping on validation accuracy
- Saves best checkpoint to `models/checkpoints/visig_simple_cnn.pt`
- Logs train/val metrics per epoch and final test accuracy

---

### 4) LSTM Sequence Classifier (New)
Files:  
`src/models/seq_lstm.py`  
`src/training/train_seq_classifier.py`

**Key ideas**
- Input `(B, T, F)` → `nn.LSTM(batch_first=True)` → masked/pooled over valid timesteps (ignores padding) → MLP head → logits.
- Shines with windowed training (short fixed‑length segments).

**What’s implemented**
- `SimpleCricketLSTM(input_dim, num_classes, hidden_size=128, num_layers=1, dropout=0.2, head_hidden=128)`.
- Masked max‑pooling over time (no reliance on padded timesteps).
- Selectable via `--model lstm` (default remains `cnn`).
- Optional loss mixing: `--loss ce+onehot_mse` (default `ce`).

---

## 📊 Data Format (Quick Intuition)

Each `.mat` file ≈ one labeled umpire gesture.

Inside (original shapes):
- `acc_mat`: `(6, 15, N)`
- `gyro_mat`: `(6, 15, N)`
- `dist_mat`: `(6, 6, N)`
- `rawt`: `(N,)`

Where:
- `6` = number of sensor nodes  
- `15` = 5 consecutive IMU readings × 3 axes (pre‑grouped)  
- `N` = timesteps in this trial  

Converted:
```
ViSigSample:
  acc  -> (T, 6, 15)
  gyro -> (T, 6, 15)
  dist -> (T, 6, 6)
  t    -> (T,)
```

Then `to_flat_sequence(sample)` → `(T, 195)`  
From there, models only see a clean multivariate sequence.

> Snapshot
>
> - Sequences: 80 • Classes: 10 • Feature dim: 195  
> - Balanced: 8 samples per class

<details>
  <summary>📷 Signal reference (cricket)</summary>

  <p align="center">
    <img src="visig_body_signal_data/signal_codebooks/cricket.jpg" alt="Cricket signal reference" width="600"/>
  </p>
</details>

---

## 🧪 Quickstart

### Option A) Run in Jupyter Notebook (recommended for exploration)
- Activate your environment and set `VISIG_ROOT` (see steps below)
- Launch Jupyter and open the notebook:
  ```bash
  jupyter lab  # or: jupyter notebook
  ```
  Then open: `notebooks/train_cricket_classifier.ipynb`
- Run all cells to:
  - Load and summarize the dataset
  - Build train/val/test splits
  - Train the baseline 1D‑CNN and view metrics
- Tweak hyperparameters in the top “Config” cell (`max_len`, `batch_size`, `lr`, `num_epochs`, etc.)

### 1) Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Minimal requirements (if `requirements.txt` is missing):
```
numpy
scipy
torch
torchvision
torchaudio
tqdm
```

### 2) Point to the data
We use `VISIG_ROOT` so paths aren’t hardcoded.
```bash
export VISIG_ROOT="/path/to/visig/data/cricket"
```
Or in `.env`:
```
VISIG_ROOT=/path/to/visig/data/cricket
```
Each `.mat` file should live somewhere under this directory.

### 3) Sanity check: loader
```bash
python -m src.data_loading.load_visig
```
What it prints:
- number of samples
- label distribution
- shapes for the first sample

### 4) Create datasets & inspect shapes
```bash
python -m src.data_loading.cricket_dataset
```
What it prints:
- split sizes
- example `x` shape (should be `(max_len, feature_dim)`)
- example label index

### 5) Train the baseline CNN
```bash
python -m src.training.train_seq_classifier
```
Defaults:
- `max_len = 400`
- `batch_size = 8`
- `lr = 1e-3`
- `num_epochs = 50`
- `patience = 8`

During training you’ll see logs like:
```
Using device: cuda
Input dim: 195, num_classes: 10
Dataset sizes -> train: 56, val: 12, test: 12
Epoch 001: train_loss=..., val_acc=..., best_val_acc=...
...
Saved best model to models/checkpoints/visig_simple_cnn.pt
Test accuracy: 0.67
```

---

## 🧱 Model Overview

| Component | Purpose |
|---|---|
| 1D CNN over time | Learn local temporal patterns |
| Global max pool | Shift invariance across time |
| Linear classifier | Map pooled features to class logits |
| Early stopping | Prevent overfitting on small dataset |

> Why 1D CNN?  
> Simple, data‑efficient, and standard for multivariate time‑series classification — a strong baseline before heavier models.

---

## 📈 Results Snapshot
- Test accuracy typically: **~60–70%** with a 2‑layer CNN on this tiny dataset  
- Sanity check: random baseline would be **~10%**

If you see ~0.1 accuracy, re‑check:
- `VISIG_ROOT` path
- tensor shapes
- label mapping consistency

---

## 🔬 Evaluation & Windowed Training (New)

- We now support training/evaluation on fixed‑length windows with optional preprocessing:
  - Windowing: `use_windows=True`, `window_size=32`, `stride=8`
  - Preprocessing flags: `standardize`, `imu_zero_offset`, `lowpass_k=5`, `uwb_correct`
- Preprocessing transforms (`src/data_processing/preprocess.py`):
  - `StandardizeFeatures`: fit per‑feature mean/std on train windows; saved to `models/checkpoints/scaler_windowed.pt` (and under `results/...` when using the eval runner).
  - `IMUZeroOffset`: subtract per‑window mean on acc+gyro channels.
  - `LowPassSmooth(k)`: moving‑average smoothing on IMU channels (k odd).
  - `UWBCorrectStaticClamp`: when accel variance is low, clamp UWB spikes to robust caps.
- Splits:
  - Random/stratified trial‑level split (so every class appears in each split).
  - LOPO (Leave‑One‑Participant‑Out) using `participant_id` from filenames.
- Metrics:
  - Window‑level: each `(W,F)` slice is scored → good for diagnostics.
  - Trial‑level: aggregate window probabilities per trial → main accuracy for gesture classification.

### Windowed dataset (New)
- `src/data_loading/windowed_cricket_dataset.py` generates overlapping windows from each trial after the split.
- Exposes:
  - `window_to_trial`: maps each window to its originating trial index.
  - `trial_labels`: per‑trial true labels.
- Enables trial‑level aggregation from window predictions.

### How to run from the notebook
- Set these in `HYPERPARAMS` and run the existing training cell:
  - `use_windows=True`, `window_size=32`, `stride=8`
  - `standardize=True`, `imu_zero_offset=True`, `lowpass_k=5`, `uwb_correct=True`
  - `stratified_split=True`
- The training cell prints both window‑ and trial‑level results for the test split.

### Training script flags (New)
The training entrypoint supports:
- Model/Loss: `--model {cnn,lstm}`, `--loss {ce,ce+onehot_mse}`, `--mse-weight`
- Windowing: `--use-windows`, `--window-size`, `--stride`
- Preprocessing: `--standardize`, `--imu-zero-offset`, `--lowpass-k`, `--uwb-correct`
- Splits: `--stratified-split` (label‑stratified trials). LOPO via the eval runner below.
- Checkpoints:
  - Notebook/regular runs: `models/checkpoints/visig_simple_{model}.pt`
  - Eval runner: per‑fold checkpoints and scaler saved under `results/...`

### LOPO (CLI, recommended for full eval)

```bash
python -m src.eval.run_eval \
  --data-root "$VISIG_ROOT" \
  --protocol lopo \
  --model lstm \
  --batch-size 32 --window-size 32 --stride 8 \
  --standardize --imu-zero-offset --lowpass-k 5 --uwb-correct \
  --num-epochs 40 --patience 8
```

- Outputs JSON summaries under `results/<timestamp>/`.
- Saves per‑fold checkpoints to `results/<timestamp>/lopo_pid_<PID>/checkpoint_lstm.pt` and the fitted scaler when used.

Random/stratified evaluation is also supported:

```bash
python -m src.eval.run_eval \
  --data-root "$VISIG_ROOT" \
  --protocol random \
  --model lstm \
  --batch-size 32 --window-size 32 --stride 8 \
  --standardize --imu-zero-offset --lowpass-k 5 --uwb-correct \
  --num-epochs 40 --patience 8
```

> Note on metrics
>
> - Window‑level accuracy counts windows (longer trials yield more windows).
> - Trial‑level aggregates windows per trial (one prediction per gesture) and is the primary headline number.

---

## ✅ Current Results

### LOPO (Leave‑One‑Participant‑Out)

pid  test_acc  window_acc  trial_acc  
1    0.941     0.941       1.000  
2    0.928     0.928       1.000  
3    1.000     1.000       1.000  
4    0.962     0.962       1.000  
5    0.991     0.991       1.000  
6    0.995     0.995       1.000  
7    0.985     0.985       1.000  
8    0.967     0.967       1.000  

Summary (mean ± std):
- **Test Acc**: 0.969 ± 0.024 (n=8)
- **Window Acc**: 0.969 ± 0.024 (n=8)
- **Trial Acc**: 1.000 ± 0.000 (n=8)

### Stratified split (no LOPO) – Per‑class (Window‑level)

- boundary4: 0.968 (30/31)  
- boundary6: 1.000 (38/38)  
- cancelcall: 1.000 (32/32)  
- deadball: 1.000 (32/32)  
- legbye: 1.000 (38/38)  
- noball: 0.757 (28/37)  
- out: 0.946 (35/37)  
- penaltyrun: 1.000 (35/35)  
- shortrun: 0.765 (26/34)  
- wide: 1.000 (34/34)  

Overall window‑level accuracy: **0.943**

---

## 🏗️ Suggested Repository Layout

```
project-root/
  README.md
  .env                         # optional, can store VISIG_ROOT here
  models/
    checkpoints/
  src/
    __init__.py
    data_loading/
      __init__.py
      load_visig.py
      cricket_dataset.py
    models/
      __init__.py
      seq_cnn.py
    training/
      __init__.py
      train_seq_classifier.py
  notebooks/
    00_eda.ipynb               # lengths, label dist, sanity plots, etc.
  data/
    # (we do NOT commit .mat here; stored locally)
```

---

## 🗺️ Roadmap
- ✅ Clean ingestion of `.mat` files into a typed Python representation  
- ✅ Standardized `(T, F)` feature representation for each trial  
- ✅ PyTorch `Dataset` + random/stratified trial‑level splitting  
- ✅ Simple 1D CNN baseline with early stopping  
- ✅ LSTM classifier with masked pooling (windowed training)  
- ✅ Windowed dataset and preprocessing (standardize, IMU zero‑offset, low‑pass, UWB clamp)  
- ✅ Trial‑level aggregation from windows (window → trial mapping)  
- ✅ LOPO evaluation protocol (per‑participant)  
- ✅ Evaluation runner CLI with saved checkpoints/scalers and JSON summaries  
- ✅ Verified that the sensor data is predictive of the signal labels  
- ⏭️ Continuous stream simulation (concatenate trials + idle; sliding window + detector)  
- ⏭️ Sensor fusion variant (two‑branch IMU+UWB) and ablations  
- ⏭️ Augmentation tuning and robustness analysis  
- ⏭️ Unify standalone tester with windowed/preprocessing flags  
- ⏭️ LLM‑based commentary generation (templates as robust fallback)  
- ⏭️ End‑to‑end demo: “Given this sensor stream, show detected events + auto‑generated commentary.”

