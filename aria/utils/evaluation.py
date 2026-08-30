"""
evaluation.py
=============
Benchmark evaluation module for ARIA.
Computes ROC-AUC, PR-AUC, F1, Precision, Recall, Confusion Matrix,
and Latency profiles for all three pillars + ARIA Score Fusion.
Generates metrics exportable to JSON and Markdown for Writeup & PPT.
"""

import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

def compute_detector_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 1.0
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 1.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "false_positive_rate": round(fpr, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }

def benchmark_latency(score_func, sample_inputs: List[Any], n_runs: int = 100) -> Dict[str, float]:
    """Measure P50, P95, P99 and Mean inference latency in milliseconds."""
    latencies = []
    for _ in range(n_runs):
        inp = sample_inputs[_ % len(sample_inputs)]
        start = time.perf_counter()
        score_func(inp)
        duration_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(duration_ms)

    latencies.sort()
    return {
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "mean_ms": round(float(np.mean(latencies)), 2)
    }
