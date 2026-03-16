import numpy as np
import pandas as pd
from .config import WINDOW_SIZE, HORIZON


def make_raw_windows(values, labels, W=WINDOW_SIZE, H=HORIZON):
    """
    Create sliding windows from time series for binary classification.

    Returns:
        X: np.float32 array (n_samples, W)
        y: np.int32 array (n_samples,)
    """
    n = len(values)
    n_samples = n - W - H

    # Vectorized window creation
    idx = np.arange(W)[None, :] + np.arange(n_samples)[:, None]
    X = values[idx].astype(np.float32)

    # Vectorized labels
    lbl_idx = np.arange(H)[None, :] + np.arange(W, W + n_samples)[:, None]
    y = labels[lbl_idx].any(axis=1).astype(np.int32)

    return X, y


def make_feature_windows(values, timestamps, labels, W=WINDOW_SIZE, H=HORIZON):
    """
    Extract engineered features from each sliding window.

    Returns:
        X: np.float32 array (n_samples, 9)
        y: np.int32 array (n_samples,)
        feature_names: list of str
    """
    n_samples = len(values) - W - H
    feature_names = ['mean', 'std', 'min', 'max', 'last',
                     'slope', 'diff_mean', 'sin_hour', 'cos_hour']

    # X = np.empty((n_samples, len(feature_names)), dtype=np.float32)
    # x_axis = np.arange(W, dtype=np.float32)
    # ts = pd.to_datetime(timestamps)

    # for i in range(n_samples):
    #     t = W + i
    #     w = values[t - W: t]
    #     X[i, 0] = w.mean()
    #     X[i, 1] = w.std()
    #     X[i, 2] = w.min()
    #     X[i, 3] = w.max()
    #     X[i, 4] = w[-1]
    #     X[i, 5] = np.polyfit(x_axis, w, 1)[0]
    #     X[i, 6] = np.abs(np.diff(w)).mean()
    #     hour = ts[t].hour
    #     X[i, 7] = np.sin(2 * np.pi * hour / 24)
    #     X[i, 8] = np.cos(2 * np.pi * hour / 24)
    win_idx = np.arange(W)[None, :] + np.arange(n_samples)[:, None]
    windows = values[win_idx]  # (n_samples, W)

    X = np.empty((n_samples, len(feature_names)), dtype=np.float32)
    X[:, 0] = windows.mean(axis=1)
    X[:, 1] = windows.std(axis=1)
    X[:, 2] = windows.min(axis=1)
    X[:, 3] = windows.max(axis=1)
    X[:, 4] = windows[:, -1]
    # Vectorized slope: cov(x, y) / var(x) — same as polyfit degree 1
    x_axis = np.arange(W, dtype=np.float32)
    x_mean = x_axis.mean()
    x_var = ((x_axis - x_mean) ** 2).sum()
    y_centered = windows - windows.mean(axis=1, keepdims=True)
    X[:, 5] = (y_centered * (x_axis - x_mean)).sum(axis=1) / x_var
    X[:, 6] = np.abs(np.diff(windows, axis=1)).mean(axis=1)
    ts = pd.to_datetime(timestamps)
    hours = np.array([ts[W + i].hour for i in range(n_samples)])
    X[:, 7] = np.sin(2 * np.pi * hours / 24)
    X[:, 8] = np.cos(2 * np.pi * hours / 24)

    lbl_idx = np.arange(H)[None, :] + np.arange(W, W + n_samples)[:, None]
    y = labels[lbl_idx].any(axis=1).astype(np.int32)

    return X, y, feature_names