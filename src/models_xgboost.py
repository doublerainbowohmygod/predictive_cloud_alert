import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_recall_curve, auc as sk_auc
from xgboost import XGBClassifier
from .config import N_ESTIMATORS, MAX_DEPTH, RANDOM_STATE, N_SPLITS
from .evaluation import find_threshold_on_train, print_fold_summary


def eval_xgboost_tscv(name, X, y, n_splits=N_SPLITS):
    """
    XGBoost with TimeSeriesSplit cross-validation.
    Threshold tuned on train, applied to test (no data leakage).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []

    for fold_i, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        if y_tr.sum() == 0 or y_te.sum() == 0:
            print(f"  Fold {fold_i + 1}: skipped (no anomalies)")
            continue

        weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        model = XGBClassifier(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
            scale_pos_weight=weight, eval_metric='aucpr',
            random_state=RANDOM_STATE,
        )
        model.fit(X_tr, y_tr, verbose=False)

        # Threshold on train (no leakage)
        proba_tr = model.predict_proba(X_tr)[:, 1]
        thr = find_threshold_on_train(y_tr, proba_tr)

        # Evaluate on test
        proba_te = model.predict_proba(X_te)[:, 1]
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

        print(f"  Fold {fold_i + 1}: PR-AUC={pr_auc:.4f} | "
              f"P={precision:.3f} R={recall:.3f} F1={f1:.3f} | "
              f"thr={thr:.3f} | "
              f"train={len(X_tr)} test={len(X_te)} incidents={int(y_te.sum())}")

    print_fold_summary(folds)
    return folds