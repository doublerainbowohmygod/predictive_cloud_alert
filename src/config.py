# === Paths ===
DATA_DIR = 'data/realAWSCloudwatch'
LABELS_PATH = 'labels/combined_windows.json'
NAB_API_URL = "https://api.github.com/repos/numenta/NAB/contents/data/realAWSCloudwatch"
NAB_LABELS_URL = "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json"
PRYSHLYAK_BASE_URL = "https://huggingface.co/datasets/pryshlyak/seasonal_time_series_for_anomaly_detection/resolve/main/"

# === Sliding Window ===
# W=12: 12 points × 5 min = 1 hour context
# H=6:  6 points × 5 min = 30 min prediction horizon
WINDOW_SIZE = 12
HORIZON = 6

# === XGBoost ===
N_ESTIMATORS = 100
MAX_DEPTH = 4
RANDOM_STATE = 42

# === Evaluation ===
N_SPLITS = 5
TRAIN_RATIO = 0.7

# === Deep Learning ===
CNN_EPOCHS = 50
CNN_LR = 0.001
CNN_BATCH_SIZE = 64
FOCAL_ALPHA = 1.0
FOCAL_GAMMA = 2.0  # Lin et al., 2017

# === MAD Baseline ===
MAD_THRESHOLD = 3.0
MAD_SCALE = 0.6745