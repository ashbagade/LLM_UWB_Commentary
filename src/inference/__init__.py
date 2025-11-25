"""
Inference utilities for running trained sequence classifiers on ViSig data.

Modules:
- load_model: helpers to build and load CNN/LSTM models from checkpoints.
- run_on_trial: CLI entrypoint to run a trained model on a single .mat trial.

Typical usage (from repo root):

    python -m inference.run_on_trial \\
        --mat-path path/to/trial.mat \\
        --checkpoint path/to/model_state.pt \\
        --model-type cnn \\
        --num-classes 10
"""
