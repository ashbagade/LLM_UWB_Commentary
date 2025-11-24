from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.eval.eval_protocols import make_random_split, make_lopo_splits
from src.eval.metrics import confusion_from_preds, precision_recall_f1, macro_micro_f1
from src.data_loading.cricket_dataset import CricketSignalsDataset, build_label_mapping, stratified_trial_split
from src.data_loading.windowed_cricket_dataset import WindowedCricketDataset
from src.data_processing.preprocess import Compose, IMUZeroOffset, LowPassSmooth, UWBCorrectStaticClamp, StandardizeFeatures
from src.models.seq_cnn import SimpleCricketCNN
from src.models.seq_lstm import SimpleCricketLSTM
from src.training.train_seq_classifier import (
    get_device,
    make_dataloaders,
    train_one_epoch,
    evaluate_accuracy,
    evaluate_window_and_trial_level,
)


def build_windowed_loaders(
    train_samples, val_samples, test_samples,
    batch_size: int,
    window_size: int,
    stride: int,
    standardize: bool,
    imu_zero_offset: bool,
    lowpass_k: int,
    uwb_correct: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict, Optional[StandardizeFeatures]]:
    # Build label mapping
    label_to_idx = build_label_mapping(train_samples + val_samples + test_samples)

    # Pre-transform
    pre_list = []
    if imu_zero_offset:
        pre_list.append(IMUZeroOffset())
    if lowpass_k and lowpass_k > 1:
        pre_list.append(LowPassSmooth(kernel_size=lowpass_k))
    if uwb_correct:
        pre_list.append(UWBCorrectStaticClamp())
    pre_transform = Compose(pre_list) if pre_list else None

    # Train temp for scaler fit
    w_train_temp = WindowedCricketDataset(
        train_samples, label_to_idx,
        window_size=window_size, stride=stride, pad_value=0.0,
        use_upper_tri_dist=True, transform=pre_transform
    )
    std_transform = None
    if standardize:
        mean, std = StandardizeFeatures.compute_mean_std_over_dataset(w_train_temp, pre_transform=None)
        std_transform = StandardizeFeatures(mean=mean, std=std)

    def make_full_transform():
        parts = []
        if pre_transform is not None and len(pre_transform) > 0:
            parts.extend(pre_transform.transforms)
        if std_transform is not None:
            parts.append(std_transform)
        return Compose(parts) if parts else None

    tfm = make_full_transform()
    w_train = WindowedCricketDataset(train_samples, label_to_idx, window_size, stride, 0.0, True, transform=tfm)
    w_val   = WindowedCricketDataset(val_samples,   label_to_idx, window_size, stride, 0.0, True, transform=tfm)
    w_test  = WindowedCricketDataset(test_samples,  label_to_idx, window_size, stride, 0.0, True, transform=tfm)

    train_loader = DataLoader(w_train, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(w_val,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(w_test,  batch_size=batch_size, shuffle=False)

    meta = {
        "input_dim": w_train.feature_dim,
        "num_classes": w_train.num_classes,
        "label_names": sorted(label_to_idx, key=lambda k: label_to_idx[k]),
    }
    return train_loader, val_loader, test_loader, meta, std_transform


def build_model(model_type: str, input_dim: int, num_classes: int) -> torch.nn.Module:
    if model_type == "cnn":
        return SimpleCricketCNN(
            input_dim=input_dim,
            num_classes=num_classes,
            num_channels=128,
            num_layers=2,
            kernel_size=5,
            dropout=0.3,
        )
    elif model_type == "lstm":
        return SimpleCricketLSTM(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_size=128,
            num_layers=1,
            dropout=0.2,
            head_hidden=128,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def run_one_training(
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: int,
    num_classes: int,
    lr: float,
    num_epochs: int,
    patience: int,
    save_dir: Optional[Path] = '../models/checkpoints',
) -> Dict:
    device = get_device()
    model = build_model(model_type, input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val = 0.0
    best_state = None
    epochs_no_improve = 0
    for epoch in range(1, num_epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate_accuracy(model, val_loader, device)
        if val_acc > best_val + 1e-4:
            best_val = val_acc
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        # Save checkpoint if requested
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            ckpt = {
                "model_state": best_state,
                "input_dim": input_dim,
                "num_classes": num_classes,
                "model_type": model_type,
                "best_val_acc": best_val,
            }
            ckpt_path = save_dir / f"checkpoint_{model_type}.pt"
            torch.save(ckpt, ckpt_path)
    else:
        ckpt_path = None
    test_acc = evaluate_accuracy(model, test_loader, device)
    # also window/trial-level
    label_names = {i: n for i, n in enumerate(test_loader.dataset.label_to_idx.keys())}
    wt = evaluate_window_and_trial_level(model, test_loader, device, label_names=label_names, use_trial_level=True)
    out = {
        "test_acc": float(test_acc),
        "window_level_accuracy": wt.get("window_accuracy"),
    }
    if "trial_accuracy" in wt:
        out["trial_level_accuracy"] = wt["trial_accuracy"]
    if save_dir is not None:
        out["checkpoint_path"] = str(save_dir / f"checkpoint_{model_type}.pt")
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluation runner (random/LOPO)")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--protocol", type=str, choices=["random", "lopo"], default="random")
    parser.add_argument("--model", type=str, choices=["cnn", "lstm"], default="lstm")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--imu-zero-offset", action="store_true")
    parser.add_argument("--lowpass-k", type=int, default=1)
    parser.add_argument("--uwb-correct", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.protocol == "random":
        train_s, val_s, test_s = make_random_split(args.data_root)
        run_subdir = out_dir / "random"
        train_loader, val_loader, test_loader, meta, std_tfm = build_windowed_loaders(
            train_s, val_s, test_s,
            batch_size=args.batch_size,
            window_size=args.window_size,
            stride=args.stride,
            standardize=args.standardize,
            imu_zero_offset=args.imu_zero_offset,
            lowpass_k=args.lowpass_k,
            uwb_correct=args.uwb_correct,
        )
        # Save scaler if used
        if std_tfm is not None:
            (run_subdir).mkdir(parents=True, exist_ok=True)
            std_path = run_subdir / "scaler_windowed.pt"
            std_tfm.save(std_path)
        res = run_one_training(
            args.model, train_loader, val_loader, test_loader,
            input_dim=meta["input_dim"], num_classes=meta["num_classes"],
            lr=args.lr, num_epochs=args.num_epochs, patience=args.patience,
            save_dir=run_subdir,
        )
        (out_dir / "random_results.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
    else:
        folds = make_lopo_splits(args.data_root)
        fold_results: List[Dict] = []
        for train_s_all, test_s, pid in folds:
            # split off a small validation set from train_s_all (stratified)
            tr_s, va_s, _ = stratified_trial_split(train_s_all, train_ratio=0.85, val_ratio=0.15, seed=pid)
            fold_dir = out_dir / f"lopo_pid_{pid}"
            train_loader, val_loader, test_loader, meta, std_tfm = build_windowed_loaders(
                tr_s, va_s, test_s,
                batch_size=args.batch_size,
                window_size=args.window_size,
                stride=args.stride,
                standardize=args.standardize,
                imu_zero_offset=args.imu_zero_offset,
                lowpass_k=args.lowpass_k,
                uwb_correct=args.uwb_correct,
            )
            # Save scaler if used
            if std_tfm is not None:
                fold_dir.mkdir(parents=True, exist_ok=True)
                std_path = fold_dir / "scaler_windowed.pt"
                std_tfm.save(std_path)
            res = run_one_training(
                args.model, train_loader, val_loader, test_loader,
                input_dim=meta["input_dim"], num_classes=meta["num_classes"],
                lr=args.lr, num_epochs=args.num_epochs, patience=args.patience,
                save_dir=fold_dir,
            )
            res["heldout_pid"] = pid
            fold_results.append(res)
            print(f"PID {pid} -> {res}")

        # aggregate
        def agg(key: str):
            vals = [r[key] for r in fold_results if key in r]
            return float(np.mean(vals)), float(np.std(vals))

        summary = {
            "lopo_num_folds": len(fold_results),
            "test_acc_mean_std": agg("test_acc"),
            "window_acc_mean_std": agg("window_level_accuracy"),
            "trial_acc_mean_std": agg("trial_level_accuracy"),
            "folds": fold_results,
        }
        (out_dir / "lopo_results.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


