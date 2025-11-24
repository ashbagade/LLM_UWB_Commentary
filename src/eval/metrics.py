from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


def confusion_from_preds(true: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    return cm


def precision_recall_f1(cm: np.ndarray) -> Dict[str, np.ndarray]:
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
        rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) != 0)
    return {"precision": prec, "recall": rec, "f1": f1}


def macro_micro_f1(cm: np.ndarray) -> Dict[str, float]:
    cls = precision_recall_f1(cm)
    macro_f1 = float(np.nanmean(cls["f1"]))
    tp = np.diag(cm).sum()
    fp = cm.sum(axis=0).sum() - tp
    fn = cm.sum(axis=1).sum() - tp
    micro_prec = tp / max(1.0, tp + fp)
    micro_rec = tp / max(1.0, tp + fn)
    micro_f1 = 0.0 if (micro_prec + micro_rec) == 0 else 2 * micro_prec * micro_rec / (micro_prec + micro_rec)
    return {"macro_f1": macro_f1, "micro_f1": float(micro_f1)}


