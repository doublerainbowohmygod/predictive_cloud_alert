import os
import json
import requests
import pandas as pd
from .config import *


def download_nab_data():
    """Download NAB CloudWatch CSV and anomaly labels."""
    os.makedirs(DATA_DIR, exist_ok=True)
    response = requests.get(NAB_API_URL)
    if response.status_code != 200:
        print("Error: GitHub API unavailable")
        return
    for info in response.json():
        if not info['name'].endswith('.csv'):
            continue
        target = os.path.join(DATA_DIR, info['name'])
        if os.path.exists(target):
            continue
        r = requests.get(info['download_url'])
        if r.status_code == 200:
            with open(target, 'wb') as f:
                f.write(r.content)
            print(f"  Downloaded: {info['name']}")

    os.makedirs('labels', exist_ok=True)
    if not os.path.exists(LABELS_PATH):
        r = requests.get(NAB_LABELS_URL)
        with open(LABELS_PATH, 'wb') as f:
            f.write(r.content)
        print(f"  Downloaded: {LABELS_PATH}")


def load_labeled_series(csv_path):
    """Load CSV + attach binary anomaly labels from combined_windows.json."""
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    with open(LABELS_PATH) as f:
        all_labels = json.load(f)
    file_key = f"realAWSCloudwatch/{os.path.basename(csv_path)}"
    df['is_anomaly'] = 0
    for start, end in all_labels.get(file_key, []):
        mask = (df['timestamp'] >= pd.to_datetime(start)) & \
               (df['timestamp'] <= pd.to_datetime(end))
        df.loc[mask, 'is_anomaly'] = 1
    return df


def get_interval_minutes(timestamps):
    """Compute actual sampling interval from data."""
    ts = pd.to_datetime(timestamps)
    return ts.diff().median().total_seconds() / 60