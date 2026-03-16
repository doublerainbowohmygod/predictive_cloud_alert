import numpy as np
from scipy.stats import median_abs_deviation
from .config import MAD_SCALE
from .evaluation import compute_pr_auc_f1


def mad_baseline_scores(X_test):
    """
    Rolling MAD (Median Absolute Deviation) baseline.

    Returns anomaly scores (higher = more anomalous).
    """
    context = X_test[:, :-1]
    last_vals = X_test[:, -1]
    medians = np.median(context, axis=1)
    mads = median_abs_deviation(context, axis=1) + 1e-8
    scores = np.abs(last_vals - medians) / (mads / MAD_SCALE)
    return scores


def eval_mad_baseline(name, X_test, y_test):
    """Evaluate MAD baseline and print results."""
    scores = mad_baseline_scores(X_test)
    pr_auc, best_f1 = compute_pr_auc_f1(y_test, scores)
    print(f"  {name:45s} PR-AUC={pr_auc:.4f} | F1={best_f1:.2f}")
    return pr_auc, best_f1