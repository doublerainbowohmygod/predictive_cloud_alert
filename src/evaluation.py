import numpy as np
from sklearn.metrics import precision_recall_curve, auc as sk_auc


def compute_pr_auc_f1(y_true, y_scores):
    """Compute PR-AUC and F1"""
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = sk_auc(rec, prec)
    f1 = (2 * prec * rec / (prec + rec + 1e-8)).max()
    return pr_auc, f1


def find_threshold_on_train(y_train, y_proba_train):
    """ Find optimal F1 threshold using training data."""
    prec, rec, thr = precision_recall_curve(y_train, y_proba_train)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return thr[min(f1.argmax(), len(thr) - 1)]


def print_fold_summary(folds):
    """Print mean ± std across folds."""
    if not folds:
        return
    for metric in ['pr_auc', 'precision', 'recall', 'f1']:
        vals = [f[metric] for f in folds]
        m, s = np.mean(vals), np.std(vals)
        if metric == 'pr_auc':
            print(f"\n  MEAN±STD: PR-AUC={m:.4f}±{s:.4f} | ", end="")
        elif metric == 'f1':
            print(f"F1={m:.3f}±{s:.3f}")
        else:
            print(f"{metric[0].upper()}={m:.3f}±{s:.3f} ", end="")
