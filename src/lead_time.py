import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier
from .config import N_ESTIMATORS, MAX_DEPTH, RANDOM_STATE, N_SPLITS
from .data_loader import get_interval_minutes
from .evaluation import find_threshold_on_train


def compute_lead_time(name, X, y, timestamps, is_anomaly,
                      W, H, n_splits=N_SPLITS):
    """
    Lead time = minutes before incident start when first alert fires.

    For each incident start (0→1 transition in test):
    - Search backward for the first alert (y_pred=1)
    - Stop if we hit another anomaly region (alerts there belong to it)
    - Report lead time in minutes (computed from data interval, not hardcoded)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    interval = get_interval_minutes(timestamps)
    all_lead = []

    for tr_idx, te_idx in tscv.split(X):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        if y_tr.sum() == 0 or y_te.sum() == 0:
            continue

        weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        model = XGBClassifier(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
            scale_pos_weight=weight, eval_metric='aucpr',
            random_state=RANDOM_STATE,
        )
        model.fit(X_tr, y_tr, verbose=False)

        proba_tr = model.predict_proba(X_tr)[:, 1]
        thr = find_threshold_on_train(y_tr, proba_tr)

        proba_te = model.predict_proba(X_te)[:, 1]
        pred = (proba_te >= thr).astype(int)

        # Find incident starts (0→1 transitions)
        for i in range(1, len(y_te)):
            if y_te[i] == 1 and y_te[i - 1] == 0:
                found = False
                for j in range(i - 1, -1, -1):
                    if y_te[j] == 1:
                        break  # another anomaly region
                    if pred[j] == 1:
                        all_lead.append((i - j) * interval)
                        found = True
                        break
                if not found:
                    all_lead.append(0)

    if all_lead:
        lt = np.array(all_lead)
        detected = lt[lt > 0]
        missed = (lt == 0).sum()
        print(f"\n=== Lead Time: {name} ===")
        print(f"  Interval: {interval:.0f} min")
        print(f"  Incidents: {len(lt)} | "
              f"Detected: {len(detected)} ({100*len(detected)/len(lt):.0f}%) | "
              f"Missed: {missed}")
        if len(detected) > 0:
            print(f"  Mean: {detected.mean():.0f} min | "
                  f"Median: {np.median(detected):.0f} min | "
                  f"Min: {detected.min():.0f} | Max: {detected.max():.0f}")
    else:
        print(f"\n=== Lead Time: {name} === no incidents in test folds")

    return all_lead