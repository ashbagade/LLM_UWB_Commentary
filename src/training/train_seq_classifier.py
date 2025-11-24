from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data_loading.cricket_dataset import create_cricket_datasets, create_cricket_datasets_stratified, CricketSignalsDataset
from src.data_loading.windowed_cricket_dataset import WindowedCricketDataset
from src.models.seq_cnn import SimpleCricketCNN
from src.models.seq_lstm import SimpleCricketLSTM
from src.data_processing.preprocess import (
    Compose,
    IMUZeroOffset,
    LowPassSmooth,
    UWBCorrectStaticClamp,
    StandardizeFeatures,
)


def get_device() -> torch.device:
    """Return the appropriate device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(
    train_ds,
    val_ds,
    test_ds,
    batch_size: int = 8,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train, validation, and test datasets."""
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train the model for one epoch and return (average loss, accuracy)."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    total = 0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        # Diagnostics: report mask/length coverage on the first batch each epoch
        if batch_idx == 0:
            with torch.no_grad():
                valid_mask = (x.abs().sum(dim=2) > 0)
                lengths = valid_mask.sum(dim=1).float()
                frac_valid = (valid_mask.float().mean()).item()
                min_len = lengths.min().item()
                max_len = lengths.max().item()
                mean_len = lengths.mean().item()
                print(
                    f"[dbg] batch0 valid_frac={frac_valid:.3f}, "
                    f"len(min/mean/max)={min_len:.0f}/{mean_len:.1f}/{max_len:.0f}"
                )
                if (lengths == 0).any().item():
                    print("[dbg] WARNING: at least one sequence has zero valid timesteps.")

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate the model and return accuracy."""
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / max(1, total)


@torch.no_grad()
def compute_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> np.ndarray:
    """Compute confusion matrix and return as numpy array."""
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        
        for true_label, pred_label in zip(y.cpu().numpy(), preds.cpu().numpy()):
            confusion_matrix[true_label, pred_label] += 1

    return confusion_matrix


@torch.no_grad()
def evaluate_window_and_trial_level(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_names: Dict[int, str],
    use_trial_level: bool = True,
) -> Dict:
    """
    Compute window-level accuracy and (optionally) trial-level metrics by averaging window probs.
    Assumes loader.dataset is a WindowedCricketDataset for trial-level evaluation and shuffle=False.
    """
    model.eval()
    all_window_preds = []
    all_window_labels = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        all_window_preds.append(preds.cpu().numpy())
        all_window_labels.append(y.cpu().numpy())

    import numpy as _np
    all_window_preds = _np.concatenate(all_window_preds, axis=0)
    all_window_labels = _np.concatenate(all_window_labels, axis=0)
    window_acc = float((all_window_preds == all_window_labels).mean())

    results: Dict = {"window_accuracy": window_acc}

    if use_trial_level and hasattr(loader.dataset, "window_to_trial") and hasattr(loader.dataset, "trial_labels"):
        all_window_probs = []
        for x, _ in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            all_window_probs.append(probs.cpu().numpy())
        all_window_probs = _np.concatenate(all_window_probs, axis=0)

        window_to_trial = loader.dataset.window_to_trial
        trial_labels = loader.dataset.trial_labels
        num_classes = len(label_names)

        from collections import defaultdict as _dd
        trial_probs_sum: Dict[int, _np.ndarray] = _dd(lambda: _np.zeros((num_classes,), dtype=_np.float64))
        trial_counts: Dict[int, int] = _dd(int)

        for w_idx, prob in enumerate(all_window_probs):
            t_idx = window_to_trial[w_idx]
            trial_probs_sum[t_idx] += prob
            trial_counts[t_idx] += 1

        trial_true = []
        trial_pred = []
        for t_idx, true_label in enumerate(trial_labels):
            if trial_counts[t_idx] == 0:
                continue
            avg_prob = trial_probs_sum[t_idx] / max(1, trial_counts[t_idx])
            pred_label = int(avg_prob.argmax())
            trial_true.append(true_label)
            trial_pred.append(pred_label)

        trial_true = _np.array(trial_true, dtype=int)
        trial_pred = _np.array(trial_pred, dtype=int)
        trial_acc = float((trial_true == trial_pred).mean())

        cm = _np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(trial_true, trial_pred):
            cm[t, p] += 1

        results.update(
            {
                "trial_accuracy": trial_acc,
                "trial_confusion_matrix": cm,
            }
        )

    return results


def train_seq_classifier(
    data_root: str,
    max_len: int = 400,
    batch_size: int = 8,
    lr: float = 1e-3,
    num_epochs: int = 50,
    patience: int = 8,
    model_dir: str = "models/checkpoints",
    model_type: str = "cnn",
    loss_type: str = "ce",
    onehot_mse_weight: float = 0.1,
    save_name: Optional[str] = None,
    use_windows: bool = False,
    window_size: int = 32,
    stride: int = 8,
    standardize: bool = False,
    imu_zero_offset: bool = False,
    lowpass_k: int = 1,
    uwb_correct: bool = False,
    stratified_split: bool = False,
    return_history: bool = False,
) -> Tuple[Dict, List[Dict]]:
    """Train a CNN or LSTM classifier on ViSig cricket umpire signals."""
    device = get_device()
    print(f"Using device: {device}")

    if stratified_split:
        train_ds, val_ds, test_ds = create_cricket_datasets_stratified(
            root=data_root,
            max_len=max_len,
        )
    else:
        train_ds, val_ds, test_ds = create_cricket_datasets(
            root=data_root,
            max_len=max_len,
        )

    base_ds_candidate = getattr(train_ds, "dataset", None)
    base_ds: CricketSignalsDataset = base_ds_candidate if isinstance(base_ds_candidate, CricketSignalsDataset) else train_ds  # type: ignore
    input_dim = base_ds.feature_dim
    num_classes = base_ds.num_classes

    print(f"Input dim: {input_dim}, num_classes: {num_classes}")
    print(f"Dataset sizes -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    if use_windows:
        base_ds: CricketSignalsDataset = getattr(train_ds, "dataset", None) or train_ds  # type: ignore
        label_to_idx = base_ds.label_to_idx

        def subset_samples(split):
            if hasattr(split, "indices"):
                return [base_ds.samples[i] for i in split.indices]
            if hasattr(split, "samples"):
                return split.samples
            raise ValueError("Unsupported split type for subset_samples")

        pre_list = []
        if imu_zero_offset:
            pre_list.append(IMUZeroOffset())
        if lowpass_k and lowpass_k > 1:
            pre_list.append(LowPassSmooth(kernel_size=lowpass_k))
        if uwb_correct:
            pre_list.append(UWBCorrectStaticClamp())
        pre_transform = Compose(pre_list) if pre_list else None

        w_train_temp = WindowedCricketDataset(
            subset_samples(train_ds), label_to_idx,
            window_size=window_size, stride=stride,
            pad_value=0.0, use_upper_tri_dist=True,
            transform=pre_transform if pre_transform else None,
        )

        std_transform = None
        if standardize:
            mean, std = StandardizeFeatures.compute_mean_std_over_dataset(
                w_train_temp, pre_transform=None
            )
            std_transform = StandardizeFeatures(mean=mean, std=std)
            scaler_path = Path(model_dir) / "scaler_windowed.pt"
            std_transform.save(scaler_path)
            print(f"Saved windowed standardizer to {scaler_path}")

        def make_full_transform():
            parts = []
            if pre_transform is not None and len(pre_transform) > 0:
                parts.extend(pre_transform.transforms)
            if std_transform is not None:
                parts.append(std_transform)
            return Compose(parts) if parts else None

        full_transform = make_full_transform()

        w_train = WindowedCricketDataset(
            subset_samples(train_ds), label_to_idx,
            window_size=window_size, stride=stride,
            pad_value=0.0, use_upper_tri_dist=True,
            transform=full_transform,
        )
        w_val = WindowedCricketDataset(
            subset_samples(val_ds), label_to_idx,
            window_size=window_size, stride=stride,
            pad_value=0.0, use_upper_tri_dist=True,
            transform=full_transform,
        )
        w_test = WindowedCricketDataset(
            subset_samples(test_ds), label_to_idx,
            window_size=window_size, stride=stride,
            pad_value=0.0, use_upper_tri_dist=True,
            transform=full_transform,
        )
        print(f"Using windowed datasets -> "
              f"train_windows: {len(w_train)}, val_windows: {len(w_val)}, test_windows: {len(w_test)}")

        train_loader, val_loader, test_loader = make_dataloaders(
            w_train, w_val, w_test, batch_size=batch_size
        )
        input_dim = w_train.feature_dim
        num_classes = w_train.num_classes
    else:
        if standardize or imu_zero_offset or (lowpass_k and lowpass_k > 1) or uwb_correct:
            print("[warn] Preprocessing flags are set but --use-windows is False. "
                  "Preprocessing is currently supported only for windowed datasets.")
        train_loader, val_loader, test_loader = make_dataloaders(
            train_ds, val_ds, test_ds, batch_size=batch_size
        )

    model_type = (model_type or "cnn").lower()
    if model_type == "cnn":
        model = SimpleCricketCNN(
            input_dim=input_dim,
            num_classes=num_classes,
            num_channels=128,
            num_layers=2,
            kernel_size=5,
            dropout=0.3,
        ).to(device)
    elif model_type == "lstm":
        model = SimpleCricketLSTM(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_size=128,
            num_layers=1,
            dropout=0.2,
            head_hidden=128,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Expected one of ['cnn', 'lstm'].")

    # Loss configuration
    loss_type = (loss_type or "ce").lower()
    if loss_type == "ce":
        criterion: nn.Module = nn.CrossEntropyLoss()
    elif loss_type in ("ce+onehot_mse", "ce_mse"):
        class CEPlusOneHotMSE(nn.Module):
            def __init__(self, num_classes: int, mse_weight: float = 0.1) -> None:
                super().__init__()
                self.ce = nn.CrossEntropyLoss()
                self.mse = nn.MSELoss()
                self.num_classes = num_classes
                self.mse_weight = float(mse_weight)

            def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                ce_loss = self.ce(logits, y)
                y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
                probs = F.softmax(logits, dim=1)
                mse_loss = self.mse(probs, y_onehot)
                return ce_loss + self.mse_weight * mse_loss

        criterion = CEPlusOneHotMSE(num_classes=num_classes, mse_weight=onehot_mse_weight)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Expected 'ce' or 'ce+onehot_mse'.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0
    history = []  # Track training history

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate_accuracy(model, val_loader, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "model_type": model_type,
            "loss_type": loss_type,
        })

        improved = val_acc > best_val_acc + 1e-4
        if improved:
            best_val_acc = val_acc
            best_state = {
                "model_state": model.state_dict(),
                "input_dim": input_dim,
                "num_classes": num_classes,
                "model_type": model_type,
                "epoch": epoch,
                "val_acc": val_acc,
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, "
            f"best_val_acc={best_val_acc:.3f}, no_improve={epochs_no_improve}"
        )

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    if best_state is None:
        print("Warning: no improvement recorded; using final model state.")
        best_state = {
            "model_state": model.state_dict(),
            "input_dim": input_dim,
            "num_classes": num_classes,
            "model_type": model_type,
            "epoch": epoch,
            "val_acc": best_val_acc,
        }

    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    ckpt_name = save_name or f"visig_simple_{model_type}.pt"
    ckpt_file = model_path / ckpt_name
    torch.save(best_state, ckpt_file)
    print(f"Saved best model to {ckpt_file} (val_acc={best_state['val_acc']:.3f})")

    model.load_state_dict(best_state["model_state"])
    test_acc = evaluate_accuracy(model, test_loader, device)
    print(f"\nTest accuracy: {test_acc:.3f}")

    print("\nConfusion Matrix (rows=true, cols=predicted):")
    confusion_matrix = compute_confusion_matrix(model, test_loader, device, num_classes)
    
    idx_to_label = base_ds.idx_to_label
    label_names = [idx_to_label[i] for i in range(num_classes)]
    
    max_label_len = max(len(name) for name in label_names) if label_names else 8
    header = f"{'True\\Pred':<{max_label_len + 2}}"
    for name in label_names:
        header += f"{name[:8]:>10}"  
    print(header)
    print("-" * len(header))
    
    for i, true_label in enumerate(label_names):
        row = f"{true_label:<{max_label_len + 2}}"
        for j in range(num_classes):
            count = confusion_matrix[i, j]
            row += f"{count:>10}"
        print(row)
    
    print("\nPer-class accuracy:")
    for i, label_name in enumerate(label_names):
        class_correct = confusion_matrix[i, i]
        class_total = confusion_matrix[i, :].sum()
        class_acc = class_correct / max(1, class_total)
        print(f"  {label_name}: {class_acc:.3f} ({class_correct}/{class_total})")
    
    results = {
        "test_acc": test_acc,
        "best_val_acc": best_val_acc,
        "confusion_matrix": confusion_matrix,
        "label_names": label_names,
        "checkpoint_path": str(ckpt_file),
        "num_classes": num_classes,
    }

    if use_windows:
        label_names_map = {i: n for i, n in enumerate(label_names)}
        wt = evaluate_window_and_trial_level(
            model, test_loader, device, label_names=label_names_map, use_trial_level=True
        )
        print("\nWindow-/Trial-level evaluation summary:")
        print(f"  Window-level accuracy: {wt['window_accuracy']:.3f}")
        if 'trial_accuracy' in wt:
            print(f"  Trial-level accuracy:  {wt['trial_accuracy']:.3f}")
        results["window_level_accuracy"] = wt["window_accuracy"]
        if "trial_accuracy" in wt:
            results["trial_level_accuracy"] = wt["trial_accuracy"]
            results["trial_level_confusion_matrix"] = wt["trial_confusion_matrix"]
    
    if return_history:
        return results, history
    return results, []


def test_model(
    checkpoint_path: str,
    data_root: str,
    max_len: int = 400,
    batch_size: int = 8,
) -> Dict:
    """
    Load a saved model and evaluate it on the test set.
    
    Args:
        checkpoint_path: Path to the saved model checkpoint
        data_root: Root directory containing .mat files
        max_len: Maximum sequence length (should match training)
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary with test results including accuracy, confusion matrix, etc.
    """
    device = get_device()
    print(f"Using device: {device}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Validation accuracy: {checkpoint['val_acc']:.3f}")
    
    input_dim = checkpoint['input_dim']
    num_classes = checkpoint['num_classes']
    model_type = checkpoint.get('model_type', 'cnn')
    
    train_ds, val_ds, test_ds = create_cricket_datasets(
        root=data_root,
        max_len=max_len,
    )
    
    base_ds: CricketSignalsDataset = test_ds.dataset
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    if model_type == "cnn":
        model = SimpleCricketCNN(
            input_dim=input_dim,
            num_classes=num_classes,
            num_channels=128,
            num_layers=2,
            kernel_size=5,
            dropout=0.3,
        ).to(device)
    elif model_type == "lstm":
        model = SimpleCricketLSTM(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_size=128,
            num_layers=1,
            dropout=0.2,
            head_hidden=128,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type in checkpoint: {model_type}")
    
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    test_acc = evaluate_accuracy(model, test_loader, device)
    confusion_matrix = compute_confusion_matrix(model, test_loader, device, num_classes)
    
    idx_to_label = base_ds.idx_to_label
    label_names = [idx_to_label[i] for i in range(num_classes)]
    
    print(f"\nTest accuracy: {test_acc:.3f}")
    print(f"\nConfusion Matrix (rows=true, cols=predicted):")
    
    max_label_len = max(len(name) for name in label_names) if label_names else 8
    header = f"{'True\\Pred':<{max_label_len + 2}}"
    for name in label_names:
        header += f"{name[:8]:>10}"
    print(header)
    print("-" * len(header))
    
    for i, true_label in enumerate(label_names):
        row = f"{true_label:<{max_label_len + 2}}"
        for j in range(num_classes):
            count = confusion_matrix[i, j]
            row += f"{count:>10}"
        print(row)
    
    print("\nPer-class accuracy:")
    for i, label_name in enumerate(label_names):
        class_correct = confusion_matrix[i, i]
        class_total = confusion_matrix[i, :].sum()
        class_acc = class_correct / max(1, class_total)
        print(f"  {label_name}: {class_acc:.3f} ({class_correct}/{class_total})")
    
    return {
        "test_acc": test_acc,
        "confusion_matrix": confusion_matrix,
        "label_names": label_names,
        "num_classes": num_classes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViSig cricket classifier (CNN/LSTM).")
    parser.add_argument("--data-root", type=str, default=os.getenv("VISIG_ROOT"), help="Root directory with .mat files")
    parser.add_argument("--max-len", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--model-dir", type=str, default="models/checkpoints")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "lstm"])
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "ce+onehot_mse"])
    parser.add_argument("--mse-weight", type=float, default=0.1)
    parser.add_argument("--save-name", type=str, default=None, help="Optional checkpoint filename (e.g., custom.pt)")
    parser.add_argument("--use-windows", action="store_true", help="Train on fixed-length windows instead of whole trials")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--standardize", action="store_true", help="Per-feature standardization (fit on train windows)")
    parser.add_argument("--imu-zero-offset", action="store_true", help="Subtract per-window mean from IMU channels")
    parser.add_argument("--lowpass-k", type=int, default=1, help="Moving average kernel for IMU smoothing (odd, >1)")
    parser.add_argument("--uwb-correct", action="store_true", help="Clamp UWB spikes for static windows")
    parser.add_argument("--stratified-split", action="store_true", help="Use label-stratified trial-level split")
    args = parser.parse_args()

    if not args.data_root:
        raise SystemExit(
            "Please set VISIG_ROOT (or pass --data-root) to the directory containing the ViSig .mat files."
        )

    train_seq_classifier(
        data_root=args.data_root,
        max_len=args.max_len,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        patience=args.patience,
        model_dir=args.model_dir,
        model_type=args.model,
        loss_type=args.loss,
        onehot_mse_weight=args.mse_weight,
        save_name=args.save_name,
        use_windows=args.use_windows,
        window_size=args.window_size,
        stride=args.stride,
        standardize=args.standardize,
        imu_zero_offset=args.imu_zero_offset,
        lowpass_k=args.lowpass_k,
        uwb_correct=args.uwb_correct,
        stratified_split=args.stratified_split,
        return_history=False,
    )

