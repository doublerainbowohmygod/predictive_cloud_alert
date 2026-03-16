"""
Deep Learning models for time series incident prediction.

Architecture references:
- Conv1d for time series: Fawaz et al., "InceptionTime: Finding AlexNet
  for Time Series Classification", 2020 (demonstrates that 1D CNNs match
  or exceed RNNs on time series benchmarks)
- Temporal Attention: simplified from Bahdanau et al., "Neural Machine
  Translation by Jointly Learning to Align and Translate", 2015
  (attention mechanism adapted from NLP to time series)
- Focal Loss: Lin et al., "Focal Loss for Dense Object Detection", 2017
  (addresses class imbalance by down-weighting easy examples)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_curve, auc as sk_auc
from sklearn.model_selection import TimeSeriesSplit
from .config import *
from .evaluation import find_threshold_on_train, print_fold_summary


# === Loss ===

class FocalLoss(nn.Module):
    """Focal Loss for sparse anomaly detection (Lin et al., 2017)."""
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


# === Architectures ===

class CNN1D(nn.Module):
    """
    1D CNN for raw time series windows.
    Based on standard Conv1d approach for time series classification
    (Fawaz et al., 2020). Two conv layers extract local patterns
    (spikes, trends), adaptive pooling reduces to fixed-size vector.
    """
    def __init__(self, window_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x).squeeze(-1)
        return self.head(x).squeeze(-1)


class AttentionCNN(nn.Module):
    """
    1D CNN + Temporal Attention.
    Conv layers extract local patterns, attention mechanism learns
    which time positions are most predictive of incidents
    (Bahdanau-style, adapted for time series).
    """
    def __init__(self, window_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.attention = nn.Sequential(
            nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        conv_out = self.conv(x).transpose(1, 2)     # (batch, W, 32)
        attn_w = torch.softmax(self.attention(conv_out), dim=1)
        context = (conv_out * attn_w).sum(dim=1)     # (batch, 32)
        return self.head(context).squeeze(-1)


# === Training + Evaluation ===

def _normalize(X_tr, X_te):
    """Normalize using train stats only."""
    mu, sigma = X_tr.mean(), X_tr.std() + 1e-8
    return (X_tr - mu) / sigma, (X_te - mu) / sigma


def eval_dl_tscv(name, X, y, model_class, n_splits=N_SPLITS,
                 epochs=CNN_EPOCHS, lr=CNN_LR, use_focal=False):
    """
    DL evaluation with TimeSeriesSplit.
    Normalizes per fold, threshold on train, reports mean±std.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []

    for fold_i, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        if y_tr.sum() == 0 or y_te.sum() == 0:
            print(f"  Fold {fold_i+1}: skipped (no anomalies)")
            continue

        X_tr_n, X_te_n = _normalize(X_tr, X_te)
        X_tr_t = torch.FloatTensor(X_tr_n)
        y_tr_t = torch.FloatTensor(y_tr.astype(np.float32))
        X_te_t = torch.FloatTensor(X_te_n)

        model = model_class(window_size=X_tr.shape[1])

        if use_focal:
            criterion = FocalLoss()
        else:
            pos_w = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)])
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                            batch_size=CNN_BATCH_SIZE, shuffle=True)

        model.train()
        for _ in range(epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()

        # Threshold on train (uses shared function from evaluation.py)
        model.eval()
        with torch.no_grad():
            proba_tr = torch.sigmoid(model(X_tr_t)).numpy()
            proba_te = torch.sigmoid(model(X_te_t)).numpy()

        thr = find_threshold_on_train(y_tr, proba_tr)
        pred_te = (proba_te >= thr).astype(int)

        prec_pts, rec_pts, _ = precision_recall_curve(y_te, proba_te)
        pr_auc = sk_auc(rec_pts, prec_pts)
        recall = pred_te[y_te == 1].sum() / max(y_te.sum(), 1)
        precision = y_te[pred_te == 1].sum() / max(pred_te.sum(), 1)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        folds.append({
            'pr_auc': pr_auc, 'precision': precision,
            'recall': recall, 'f1': f1, 'threshold': thr,
            'train_size': len(X_tr), 'test_size': len(X_te),
            'test_incidents': int(y_te.sum()),
        })

        print(f"  Fold {fold_i+1}: PR-AUC={pr_auc:.4f} | "
              f"P={precision:.3f} R={recall:.3f} F1={f1:.3f} | "
              f"thr={thr:.3f} | "
              f"train={len(X_tr)} test={len(X_te)} incidents={int(y_te.sum())}")

    print_fold_summary(folds)
    return folds