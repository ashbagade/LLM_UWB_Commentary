<div align="center">

# 🏏 LLM-Based Automatic Cricket Commentary

Using Umpire Wearable Sensor + UWB Distance Data

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter\&logoColor=white)](notebooks/train_cricket_classifier.ipynb)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Prototype%20Demo%20Ready-brightgreen)](#-roadmap)

<br/>
<i>End-to-end: raw multisensor umpire signals ➜ ML event classification ➜ event detection ➜ match state ➜ LLM-generated cricket commentary.</i>

</div>

---

## ✨ What This Project Does

**High-level pipeline:**

> Umpire IMU+UWB signals (per gesture trial, `.mat`)
> → windowed 1D CNN classifier (per time window)
> → window-to-event post-processing (clean `UmpireEvent`s)
> → running `MatchState` (scoreboard, overs, wickets, 4s/6s, etc.)
> → LLM (Gemini) that generates broadcast-style cricket commentary.

This repo:

* **Ingests** wearable IMU + UWB distance data from cricket umpires (`.mat` files).
* **Classifies** each window of sensor data into gesture labels (boundary4, boundary6, wide, out, etc.) with a simple 1D CNN baseline (and optional LSTM).
* **Groups noisy window predictions into clean events** (`UmpireEvent` objects with timestamps, event types, and confidences).
* **Maintains a synthetic match scoreboard** (`MatchState`) as events stream in.
* **Generates natural-language commentary** for each ball using **Google Gemini**, with personality and context that builds over the innings.
* **Provides a pseudo-match simulator** that samples random per-gesture `.mat` files and runs through the full pipeline: detection → match state → commentary.

---

## 🗂️ Table of Contents

* [✨ What This Project Does](#-what-this-project-does)
* [🏗️ End-to-End Architecture](#️-end-to-end-architecture)
* [🧪 Quickstart (for Teammates)](#-quickstart-for-teammates)

  * [1. Clone & Environment](#1-clone--environment)
  * [2. Install Dependencies](#2-install-dependencies)
  * [3. Point to Cricket Data](#3-point-to-cricket-data)
  * [4. Train the CNN Classifier](#4-train-the-cnn-classifier)
  * [5. Run Inference on a Single Trial](#5-run-inference-on-a-single-trial)
  * [6. Convert a Trial to Discrete Events](#6-convert-a-trial-to-discrete-events)
  * [7. Configure Gemini API Key](#7-configure-gemini-api-key)
  * [8. Test Commentary for a Single Event](#8-test-commentary-for-a-single-event)
  * [9. Run a Full Pseudo-Match with Commentary](#9-run-a-full-pseudo-match-with-commentary)
* [📦 Components & Files](#-components--files)

  * [Data Loading](#data-loading)
  * [Modeling & Training](#modeling--training)
  * [Inference & Event Detection](#inference--event-detection)
  * [Match State & Cricket Logic](#match-state--cricket-logic)
  * [LLM Commentary Integration](#llm-commentary-integration)
  * [Pseudo-Match Simulation](#pseudo-match-simulation)
* [📊 Data Format (Quick Intuition)](#-data-format-quick-intuition)
* [📈 Results Snapshot (ML Classifier)](#-results-snapshot-ml-classifier)
* [🔑 LLM / Gemini Usage](#-llm--gemini-usage)
* [🗺️ Roadmap](#️-roadmap)

---

## 🏗️ End-to-End Architecture

**1. Sensor → Sequence**

* `.mat` files (one umpire gesture trial per file) are loaded by `src/data_loading/load_visig.py`.
* Each trial is converted into a time-major multivariate sequence `(T, F)` with `F = 195` features (IMU + UWB distances).

**2. Sequence → Window Predictions**

* `src/training/train_seq_classifier.py` trains a 1D CNN (and optionally an LSTM) on fixed-length windows (`window_size`, `stride`) to predict one of 10 cricket gesture classes.
* `src/inference/run_on_trial.py` applies the trained model to a single `.mat` file and outputs window-level predictions (label + probability per window).

**3. Windows → Discrete Events**

* `src/events/detector.py` groups consecutive windows with the same label into runs and converts them into clean `UmpireEvent`s (timestamp, label, confidence).
* `src/inference/run_trial_to_events.py` wraps the workflow: `.mat` → window predictions → `UmpireEvent`s → JSON.

**4. Events → Match State**

* `src/state/cricket_state.py` defines `MatchState`, which keeps track of:

  * `innings`, `total_runs`, `wickets`, `balls_bowled`, `fours`, `sixes`, and the list of events.
* `apply_event_to_state(...)` updates the scoreboard as each event arrives.
* `build_state_from_events(...)` reconstructs state from a list of events (useful for offline experiments).

**5. Match State + Event → Commentary**

* `src/commentary/llm_client.py` wraps **Google Gemini** as an LLM client.
* `src/commentary/prompting.py` builds prompts describing current score, recent events, and the new event.
* `src/commentary/engine.py` maintains LLM chat history and generates commentary text for each event.

**6. Pseudo-Match Simulation**

* `src/simulation/fake_match.py`:

  * Samples random `.mat` files from the cricket folder (each treated as one ball).
  * Runs detection + `MatchState` updates for all balls.
  * Calls Gemini once with all ball-level data and asks for JSON commentary for each ball.
  * Prints a ball-by-ball “broadcast” and writes a JSON log.

---

## 🧪 Quickstart (for Teammates)

This is the recommended path for someone new to the repo to get to a working demo.

### 1. Clone & Environment

```bash
git clone https://github.com/<your-org-or-user>/LLM-Auto-Commentary.git
cd LLM-Auto-Commentary-main

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If needed, you can manually install the core dependencies:

```bash
pip install numpy scipy torch torchvision torchaudio tqdm google-genai
```

### 3. Point to Cricket Data

We **do not** commit `.mat` data. Each person should have the VISIG cricket dataset locally, e.g.:

```text
/Users/you/path/to/visig_body_signal_data/data/cricket
```

You can define a convenience environment variable:

```bash
export VISIG_ROOT="/Users/you/path/to/visig_body_signal_data/data/cricket"
```

Most scripts take `--data-root`, so `VISIG_ROOT` is optional.

### 4. Train the CNN Classifier

From the repo root:

```bash
python -m src.training.train_seq_classifier \
  --data-root /Users/you/path/to/visig_body_signal_data/data/cricket \
  --model cnn \
  --use-windows \
  --window-size 64 \
  --stride 16 \
  --save-name visig_simple_cnn.pt \
  --model-dir models
```

This will:

* Load all cricket `.mat` files.
* Build train/val/test splits.
* Train a 1D CNN on 64-length windows with stride 16.
* Save the best model to: `models/visig_simple_cnn.pt`.
* Print train/val metrics, and final test accuracy + confusion matrix.

You can skip this step if a trained checkpoint is already present in `models/visig_simple_cnn.pt`.

### 5. Run Inference on a Single Trial

To get window-level predictions for one `.mat` file:

```bash
python -m src.inference.run_on_trial \
  --mat-path /Users/you/path/to/visig_body_signal_data/data/cricket/boundary6_1.mat \
  --checkpoint models/visig_simple_cnn.pt \
  --model-type cnn \
  --num-classes 10 \
  --window-size 64 \
  --stride 16 \
  --label-map-json metadata/cricket_label_map.json \
  --output-json outputs/boundary6_1_windows.json
```

Example output (truncated):

```json
[
  {
    "window_index": 0,
    "time": 3.0966,
    "pred_idx": 1,
    "pred_label": "boundary6",
    "prob": 1.0
  },
  ...
]
```

### 6. Convert a Trial to Discrete Events

To convert noisy windows into clean `UmpireEvent`s:

```bash
python -m src.inference.run_trial_to_events \
  --mat-path /Users/you/path/to/visig_body_signal_data/data/cricket/boundary6_1.mat \
  --checkpoint models/visig_simple_cnn.pt \
  --window-size 64 \
  --stride 16 \
  --label-map-json metadata/cricket_label_map.json \
  --min-conf 0.8 \
  --min-run-length 2 \
  --merge-gap-sec 0.5 \
  --output-json outputs/boundary6_1_events.json
```

Example output:

```json
[
  {
    "event_type": "boundary6",
    "confidence": 1.0,
    "t": 13.1173
  },
  {
    "event_type": "wide",
    "confidence": 1.0,
    "t": 24.9679
  }
]
```

These events are the symbolic input to the match state and commentary system.

### 7. Configure Gemini API Key

We use **Google Gemini** via the `google-genai` package.

Each user must create their own API key in [Google AI Studio](https://ai.google.dev/gemini-api/docs/quickstart) and set it locally.

Recommended environment variable:

```bash
export GEMINI_API_KEY="your-key-here"
```

(Alternatively, `GOOGLE_API_KEY` is also supported.)

Keys are not stored in the repository.

### 8. Test Commentary for a Single Event

There is a simple example script that simulates one event and generates commentary:

```python
# test_commentary.py

from src.commentary.llm_client import LLMClient
from src.commentary.engine import CommentaryEngine
from src.state.cricket_state import MatchState, apply_event_to_state
from src.events.schema import UmpireEvent, EventType

system_prompt = (
    "You are a lively, knowledgeable cricket commentator calling a live match. "
    "You speak concisely in 1–2 sentences per event."
)

print("Initializing commentary engine...\n")
llm = LLMClient(system_prompt=system_prompt)
engine = CommentaryEngine(llm_client=llm, system_prompt=system_prompt)

state = MatchState()
print("Simulating a boundary six event...\n")

event = UmpireEvent(timestamp=15.0, event_type=EventType.BOUNDARY6, confidence=0.99)
apply_event_to_state(state, event)

print("Generating commentary...\n")
commentary = engine.generate_for_event(state, event)

print("============================================================")
print("COMMENTARY:")
print("============================================================")
print(commentary)
print("============================================================")
print(f"\nMatch State: {state.total_runs} runs, {state.balls_bowled} balls, {state.wickets} wickets")
```

Run:

```bash
python test_commentary.py
```

You should see something like:

```text
COMMENTARY:
============================================================
SIX! What a start! The batsman sends the very first ball soaring over the boundary for a massive six!
============================================================

Match State: 6 runs, 1 balls, 0 wickets
```

### 9. Run a Full Pseudo-Match with Commentary

The main end-to-end demo is:

```bash
python -m src.simulation.fake_match \
  --data-root /Users/you/path/to/visig_body_signal_data/data/cricket \
  --checkpoint models/visig_simple_cnn.pt \
  --label-map-json metadata/cricket_label_map.json \
  --num-trials 30 \
  --log-json outputs/fake_match_log.json
```

What this does:

1. Randomly samples `num-trials` `.mat` files from the cricket folder (each treated as one ball).
2. For each sampled file:

   * Runs `.mat → windows → events`.
   * Keeps the best event per trial (highest confidence).
   * Updates a global `MatchState` (runs, wickets, balls, fours, sixes).
3. Builds a ball-by-ball log (`match_log`) that includes:

   * `ball_index`, `event_type`, `confidence`,
   * `score_before` and `score_after` for each ball.
4. Calls Gemini **once** with this data, asking it to return a strict JSON array:

   * `[{ "ball_index": i, "commentary": "..." }, ...]`.
5. Prints a ball-by-ball broadcast to the terminal.
6. Writes the full log (including commentary) to `outputs/fake_match_log.json`.

This demonstrates the full pipeline from offline `.mat` gesture data to narrative commentary over a synthetic innings.

---

## 📦 Components & Files

### Data Loading

**File:** `src/data_loading/load_visig.py`

* Loads `.mat` files containing umpire trials.
* Reads:

  * `acc_mat`, `gyro_mat`, `dist_mat`, `rawt`
* Provides:

  * `ViSigSample` dataclass
  * `load_visig_mat(path)` – loads a single trial
  * `load_visig_dataset(root)` – loads all `.mat` under a directory
  * `to_flat_sequence(sample)` – converts trial to `(T, 195)` feature matrix

**File:** `src/data_loading/cricket_dataset.py`

* `build_label_mapping(...)` – label ↔ index mapping (lexicographically sorted).
* `CricketSignalsDataset` – PyTorch dataset for trials:

  * Pads or crops to `max_len`.
  * Returns tensors suitable for CNN/LSTM models.
* `create_cricket_datasets(...)` – creates train/val/test splits (e.g., 70/15/15).

(If present in your version of the repo) additional files like `windowed_cricket_dataset.py` and evaluation helpers support advanced windowed/LOPO setups described below.

---

### Modeling & Training

**CNN model**
**File:** `src/models/seq_cnn.py`

* Simple baseline for multivariate time-series classification:

  * Input: `(batch, T, F)`
  * 1D convolution layers along time.
  * Global max pooling over time.
  * Fully connected classifier → logits over 10 classes.

**LSTM model** (if present)
**File:** `src/models/seq_lstm.py`

* `SimpleCricketLSTM(input_dim, num_classes, hidden_size=128, num_layers=1, dropout=0.2, head_hidden=128)`:

  * Uses `nn.LSTM(batch_first=True)`.
  * Masked pooling over valid (non-padded) timesteps.
  * Outputs class logits.

**Training script**
**File:** `src/training/train_seq_classifier.py`

Key arguments (subset):

* `--data-root`: path to cricket `.mat` folder.
* `--model {cnn,lstm}`: choose 1D CNN or LSTM.
* `--max-len`: max sequence length for non-windowed training.
* `--use-windows`: enable windowed training.
* `--window-size`, `--stride`: window configuration.
* Preprocessing (if present): `--standardize`, `--imu-zero-offset`, `--lowpass-k`, `--uwb-correct`.
* Optimization: `--num-epochs`, `--patience`, `--lr`.
* Saving: `--model-dir`, `--save-name`.

Example (CNN + windows):

```bash
python -m src.training.train_seq_classifier \
  --data-root /path/to/cricket \
  --model cnn \
  --use-windows \
  --window-size 64 \
  --stride 16 \
  --save-name visig_simple_cnn.pt \
  --model-dir models
```

The script prints epoch-wise metrics and finally a confusion matrix and per-class accuracy.

---

### Inference & Event Detection

**Window-level inference**
**File:** `src/inference/run_on_trial.py`

* Loads a `.mat` file.
* Builds windows with the given `window_size` and `stride`.
* Applies the trained model.
* Outputs a JSON list, where each entry contains:

  * `window_index`
  * `time` (approximate timestamp of window center or end)
  * `pred_idx` (class index)
  * `pred_label` (class name)
  * `prob` (softmax confidence)

**Event schema**
**File:** `src/events/schema.py`

* `EventType` enum with values like:

  * `boundary4`, `boundary6`, `cancelcall`, `deadball`, `legbye`, `noball`, `out`, `penaltyrun`, `shortrun`, `wide`.
* `@dataclass UmpireEvent`:

  * `timestamp: float`
  * `event_type: EventType`
  * `confidence: float`

**Window → Event detection**
**File:** `src/events/detector.py`

* `detect_events(window_preds, min_conf=0.8, min_run_length=2, merge_gap_sec=0.5) -> list[UmpireEvent]`:

  * Filters out windows where `prob < min_conf` or label is not meaningful.
  * Groups consecutive windows with the same label into runs.
  * Discards runs shorter than `min_run_length`.
  * For each remaining run:

    * `timestamp` = average of window timestamps in the run.
    * `confidence` = average or max of window probabilities.
    * `event_type` = run label.
  * Optionally merges events of the same type that are within `merge_gap_sec` seconds.

**Trial → Events CLI**
**File:** `src/inference/run_trial_to_events.py`

* Command-line entrypoint that:

  * Loads `.mat` and model.
  * Runs window-level inference.
  * Calls `detect_events(...)`.
  * Writes events to JSON: `[{ "event_type": "...", "confidence": ..., "t": ... }, ...]`.
  * Prints a human-readable summary to the console.

---

### Match State & Cricket Logic

**File:** `src/state/cricket_state.py`

Defines the match state and scoring logic:

```python
from dataclasses import dataclass, field
from typing import List
from src.events.schema import UmpireEvent, EventType

@dataclass
class MatchState:
    innings: int = 1
    total_runs: int = 0
    wickets: int = 0
    balls_bowled: int = 0
    sixes: int = 0
    fours: int = 0
    events: List[UmpireEvent] = field(default_factory=list)
```

* `RUN_VALUES: Dict[EventType, int]` maps each event type to runs scored (e.g., `boundary4 → 4`, `boundary6 → 6`, `wide → 1` etc.).

* `apply_event_to_state(state: MatchState, event: UmpireEvent) -> MatchState`:

  * Updates `total_runs` based on `RUN_VALUES`.
  * Increments `balls_bowled` for legal deliveries.
  * Increments `wickets` (for `out` events).
  * Tracks `fours` and `sixes`.
  * Appends the event to `state.events`.

* `build_state_from_events(events: list[UmpireEvent]) -> MatchState`:

  * Sorts events by `timestamp`.
  * Applies `apply_event_to_state` in order to reconstruct the state.

Self-test (sanity check) can be run via:

```bash
python -m src.state.cricket_state
```

You should see something like:

```text
Self-test state: MatchState(innings=1, runs=12, wickets=1, balls=3, 4s=0, 6s=2)
Simple self-test passed.
```

---

### LLM Commentary Integration

**LLM client (Gemini wrapper)**
**File:** `src/commentary/llm_client.py`

* Wraps the Google Gemini API via `google-genai`.
* Reads API key from:

  * `GEMINI_API_KEY` (recommended), or
  * `GOOGLE_API_KEY`.
* Default model: `"gemini-2.0-flash"` (can be overridden).
* Key method:

```python
class LLMClient:
    def __init__(self, system_prompt: str, model_name: str = "gemini-2.0-flash", api_key: Optional[str] = None, debug: bool = False):
        ...

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 80, temperature: float = 0.8) -> str:
        ...
```

* Flattens a list of `{"role": ..., "content": ...}` into a prompt for Gemini.
* Robustly extracts response text, handling `response.text` and `response.candidates` cases.

**Prompt builder**
**File:** `src/commentary/prompting.py`

* `build_commentary_prompt(state: MatchState, event: UmpireEvent) -> list[dict]`:

  * Builds a minimal set of chat-style messages for the LLM:

    * System: “you are a lively cricket commentator…”
    * User: includes current score, brief history (last N events), and the new event, with instructions for 1–2 sentences of commentary.

**Commentary engine**
**File:** `src/commentary/engine.py`

```python
class CommentaryEngine:
    def __init__(self, llm_client: LLMClient, system_prompt: str, history_limit: int = 20):
        self.llm_client = llm_client
        self.history = [{"role": "system", "content": system_prompt}]
        self.history_limit = history_limit

    def generate_for_event(self, state: MatchState, event: UmpireEvent) -> str:
        messages = self.history + build_commentary_prompt(state, event)
        text = self.llm_client.generate(messages)
        self.history.append({"role": "assistant", "content": text})
        self.history = self.history[-self.history_limit:]
        return text
```

* Maintains a short rolling history of assistant messages to give the LLM context over the match.
* Used in `test_commentary.py` for single-event commentary and could be used for streaming use cases.

---

### Pseudo-Match Simulation

**File:** `src/simulation/fake_match.py`

Purpose: Build a **pseudo T20 innings** from offline per-gesture trials and generate commentary over it in one batched Gemini call (efficient and avoids rate limits).

Core flow:

1. Gather all `.mat` files under `--data-root`.
2. Randomly sample `num-trials` of them (each = one “ball”).
3. For each sampled file:

   * Run `.mat → windows → events` via `mat_to_events(...)`.
   * Choose the best event (highest confidence).
   * Create a synthetic global timestamp (e.g., every 30 seconds).
   * Apply `apply_event_to_state` to update `MatchState`.
   * Record `score_before`, `score_after`, `event_type`, `confidence` in `match_log`.
4. After all balls are processed:

   * Print final state (runs, wickets, balls, fours, sixes).
   * Call `generate_batch_commentary(llm_client, match_log)`:

     * Builds instructions: “You are a lively TV cricket commentator…”
     * Provides ball-by-ball data with scores before/after each ball.
     * Asks for strict JSON of the form:
       `[{ "ball_index": 1, "commentary": "..." }, ...]`.
   * Map the returned commentary back to each ball.
   * Print ball-by-ball commentary and write an optional JSON log.

Example run:

```bash
python -m src.simulation.fake_match \
  --data-root /path/to/visig_body_signal_data/data/cricket \
  --checkpoint models/visig_simple_cnn.pt \
  --label-map-json metadata/cricket_label_map.json \
  --num-trials 30 \
  --log-json outputs/fake_match_log.json
```

Configuration flags:

* `--data-root`: cricket `.mat` directory.
* `--checkpoint`: trained model (e.g., `models/visig_simple_cnn.pt`).
* `--label-map-json`: label mapping (e.g., `metadata/cricket_label_map.json`).
* `--num-trials`: number of balls in the pseudo-match.
* `--window-size`, `--stride`: must match training/inference.
* `--min-conf`, `--min-run-length`, `--merge-gap-sec`: event detection parameters.
* `--time-step-sec`: synthetic time gap between balls.
* `--model-name`: Gemini model name, default `"gemini-2.0-flash"`.
* `--seed`: random seed for reproducible sampling.
* `--log-json`: optional JSON file to write a detailed match log.

---

## 📊 Data Format (Quick Intuition)

Each `.mat` file ≈ one labeled umpire gesture.

Original shapes (per trial):

* `acc_mat`: `(6, 15, N)`
* `gyro_mat`: `(6, 15, N)`
* `dist_mat`: `(6, 6, N)`
* `rawt`: `(N,)`

Where:

* `6` = number of sensor nodes.
* `15` = 5 consecutive IMU readings × 3 axes (pre-grouped).
* `N` = timesteps in this trial.

`to_flat_sequence(sample)` flattens this into `(T, 195)` where:

* 90 dims from accelerometer,
* 90 dims from gyroscope,
* 15 dims from upper-triangular UWB distances.

Label set (10 classes):

* `boundary4`, `boundary6`, `cancelcall`, `deadball`, `legbye`, `noball`, `out`, `penaltyrun`, `shortrun`, `wide`

The dataset is reasonably balanced in the standard ViSig cricket split (e.g., 8 trials per class in the original version).

---

## 📈 Results Snapshot (ML Classifier)

From earlier experiments on the ViSig cricket data:

* 80 sequences (trials).
* 10 classes, roughly balanced.
* Feature dimension: 195.

Example performance (windowed training + LOPO, where available):

* **Test accuracy**: ~0.97 ± 0.02 across participants.
* **Trial-level accuracy**: near-perfect (1.00) in many folds.
* Per-class window-level accuracy:

  * boundary4: ~0.97
  * boundary6: 1.00
  * cancelcall: 1.00
  * deadball: 1.00
  * legbye: 1.00
  * noball: ~0.76
  * out: ~0.95
  * penaltyrun: 1.00
  * shortrun: ~0.77
  * wide: 1.00

This indicates that the signal classification step is strong enough for the downstream event detection and commentary generation tasks.

---

## 🔑 LLM / Gemini Usage

* LLM backend: **Google Gemini** via `google-genai`.
* Default model: `"gemini-2.0-flash"` (fast and inexpensive).
* API key:

  * Expected in `GEMINI_API_KEY` (recommended) or `GOOGLE_API_KEY`.
  * Keys are not committed to the repository; each user must configure their own environment.

Example (macOS/Linux):

```bash
export GEMINI_API_KEY="your-key-here"
```

On Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="your-key-here"
```

Once set, you can run:

* `python test_commentary.py`
* `python -m src.simulation.fake_match ...`

to exercise the commentary generation.

---

## 🗺️ Roadmap

**Completed:**

* ✅ Clean ingestion of `.mat` files into a typed Python representation.
* ✅ Standard `(T, F)` feature representation for each trial.
* ✅ PyTorch `Dataset` for trials, with train/val/test splits.
* ✅ 1D CNN baseline with early stopping.
* ✅ Optional LSTM baseline with masked pooling.
* ✅ Windowed dataset and preprocessing (standardize, IMU zero-offset, low-pass, UWB clamp) where applicable.
* ✅ Trial-level aggregation from window predictions.
* ✅ LOPO evaluation protocols and scripts (where included).
* ✅ Strong evidence that sensor data predicts umpire signals well.
* ✅ Window → event conversion via `UmpireEvent` and `detect_events`.
* ✅ `MatchState` with scoring logic and basic cricket stats.
* ✅ Gemini-based commentary integration (single-event and batched).
* ✅ Pseudo-match simulator: `.mat` files → events → scoreboard → commentary.

**Future / possible extensions:**

* ⏭️ Real-time streaming version (instead of offline per-trial).
* ⏭️ Multi-sport support (using sport-aware prompts).
* ⏭️ More sophisticated event detectors (e.g., HMM/CRF, class-specific thresholds).
* ⏭️ Multiple commentator “personalities” (analyst vs hype caster).
* ⏭️ Simple web UI to visualize the match timeline and commentary.
