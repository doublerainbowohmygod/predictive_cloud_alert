# Predictive Alerting for Cloud Metrics

Incident prediction system for AWS CloudWatch metrics using sliding-window
binary classification. Compares statistical baselines, gradient-boosted trees,
and deep learning (1D CNN with temporal attention and focal loss).

Trained and evaluated on the [NAB](https://github.com/numenta/NAB) (Numenta Anomaly Benchmark)
real-world CloudWatch dataset.

## Project structure

```
src/
├── config.py              # All hyperparameters and paths
├── data_loader.py         # Download NAB data + attach anomaly labels
├── sliding_windows.py     # Sliding window creation (numpy vectorized)
├── baselines.py           # MAD (Median Absolute Deviation) baseline
├── evaluation.py          # PR-AUC, F1, threshold tuning (no data leakage)
├── models_xgboost.py      # XGBoost with TimeSeriesSplit
├── models_dl.py           # 1D CNN, Attention CNN, Focal Loss (PyTorch)
└── lead_time.py           # Detection lead time analysis
01_eda.ipynb               # Exploratory data analysis
02_experiments.ipynb        # All experiments, comparison, discussion
```

## Requirements
README.md
git@github.com:doublerainbowohmygod/predictive_cloud_alert.gitgit@github.com:doublerainbowohmygod/predictive_cloud_alert.git

```bash
pip install -r requirements.txt
```

If running in Jupyter Notebook (not JupyterLab or Colab):

```bash
pip install notebook
```

## Quick start

```bash
git clone https://github.com/doublerainbowohmygod/predictive_cloud_alert.git
cd predictive_cloud_alert
pip install -r requirements.txt
pip install notebook        # if Jupyter is not installed
jupyter notebook
jupyter notebook 01_eda.ipynb
```

Open `01_eda.ipynb` first — it downloads NAB data automatically.
Then open `02_experiments.ipynb` for all models and results.

All functions are in `src/`, notebooks import them.
Data downloads automatically on first run (NAB CloudWatch from GitHub API, Pryshlyak from HuggingFace, internet required).
## Methods

| Method | Type | Input |
|---|---|---|
| MAD baseline | Statistical | Rolling median + MAD per window |
| XGBoost raw | ML (trees) | 12 raw values |
| XGBoost features | ML (trees) | 9 features: mean, std, slope, sin/cos hour, ... |
| Prophet | Forecasting | Full time series → residual → threshold |
| 1D CNN | Deep Learning | 12 raw values (normalized) |
| Attention CNN + Focal Loss | Deep Learning | 12 raw values (normalized) |

## Evaluation

- **TimeSeriesSplit** (5 folds) — no future leakage
- **PR-AUC** as primary metric (robust to class imbalance)
- **Threshold tuned on train only** — no data leakage
- **Lead time** — minutes before incident when first alert fires

## Key results

Best PR-AUC per method (mean across TSCV folds):

| File | MAD | XGB-feat | CNN | A-CNN |
|---|---|---|---|---|
| iio_NetworkIn | 0.72 | 0.89 | 0.86 | 0.86 |
| rds_cpu | 0.32 | 0.60 | 0.76 | 0.73 |
| ec2_cpu_fe7f93 | 0.31 | 0.50 | 0.56 | 0.47 |

See `02_experiments.ipynb` for full comparison table and analysis.

## References

- Fawaz et al., 2019 — *Deep Learning for Time Series Classification: A Review*
- Lin et al., 2017 — *Focal Loss for Dense Object Detection*
- Bahdanau et al., 2015 — *Neural Machine Translation by Jointly Learning to Align and Translate*
- Lavin & Ahmad, 2015 — *Evaluating Real-Time Anomaly Detection Algorithms — the Numenta Anomaly Benchmark*
- Chen & Guestrin, 2016 — *XGBoost: A Scalable Tree Boosting System*
- NAB dataset: https://github.com/numenta/NAB
- Pryshlyak dataset: https://huggingface.co/datasets/pryshlyak/seasonal_time_series_for_anomaly_detection
