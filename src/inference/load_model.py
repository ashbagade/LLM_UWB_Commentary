from __future__ import annotations

from typing import Literal, Optional, Any, Dict

import torch
from torch import nn

from src.models.seq_cnn import SimpleCricketCNN
from src.models.seq_lstm import SimpleCricketLSTM

ModelType = Literal["cnn", "lstm"]


def get_device(prefer: Optional[str] = None) -> torch.device:
    """
    Choose a reasonable default device.

    prefer:
        - "cuda", "mps", or "cpu" to force a specific device if available.
        - None (default): prefer cuda > mps > cpu.
    """
    if prefer is not None:
        if prefer == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        if prefer == "mps":
            mps = getattr(torch.backends, "mps", None)
            if not (mps and torch.backends.mps.is_available()):
                return torch.device("cpu")
        return torch.device(prefer)

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(
    model_type: ModelType,
    input_dim: int,
    num_classes: int,
    **kwargs: Any,
) -> nn.Module:
    """
    Construct a sequence classifier model matching the training code.
    """
    if model_type == "cnn":
        model = SimpleCricketCNN(input_dim, num_classes, **kwargs)
    elif model_type == "lstm":
        model = SimpleCricketLSTM(input_dim, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Expected 'cnn' or 'lstm'.")
    return model


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_type: Optional[ModelType] = None,
    input_dim: Optional[int] = None,
    num_classes: Optional[int] = None,
    device: Optional[torch.device] = None,
    **model_kwargs: Any,
) -> nn.Module:
    """
    Build a model and load weights from a checkpoint produced by train_seq_classifier.

    Understands the format:
        {
            "model_state": ...,
            "input_dim": ...,
            "num_classes": ...,
            "model_type": ...,
            "epoch": ...,
            "val_acc": ...,
        }

    It can also fall back to a plain state_dict or {state_dict=..., ...} formats.
    """
    if device is None:
        device = get_device()

    ckpt = torch.load(checkpoint_path, map_location=device)

    state_dict: Dict[str, torch.Tensor]

    # Preferred: format saved by train_seq_classifier.py
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        if input_dim is None:
            input_dim = int(ckpt["input_dim"])
        if num_classes is None:
            num_classes = int(ckpt["num_classes"])
        if model_type is None:
            model_type = ckpt.get("model_type", "cnn")  # type: ignore[assignment]
    else:
        # Fallbacks: plain state_dict / {state_dict:...} / {model_state_dict:...}
        if not isinstance(ckpt, dict):
            raise ValueError("Unexpected checkpoint format (not a dict).")

        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict = ckpt  # type: ignore[assignment]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            state_dict = ckpt["model_state_dict"]
        else:
            raise ValueError(
                "Could not interpret checkpoint format. Expected a 'model_state' key "
                "as saved by train_seq_classifier, or a plain state_dict."
            )

    if input_dim is None or num_classes is None or model_type is None:
        raise ValueError(
            f"Missing model metadata: input_dim={input_dim}, "
            f"num_classes={num_classes}, model_type={model_type}"
        )

    model = build_model(model_type, input_dim, num_classes, **model_kwargs)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
