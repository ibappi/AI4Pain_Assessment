import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

# === Configuration ===
SPO2_DATA_DIR = '../dataset/train/Spo2'
OUTPUT_FILE = 'SpO2_features.csv'

# === Feature Extraction Function ===
def extract_features_from_spo2(signal):
    features = {}

    # Basic stats
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['min'] = np.min(signal)
    features['max'] = np.max(signal)
    features['range'] = np.max(signal) - np.min(signal)
    features['skewness'] = skew(signal)
    features['kurtosis'] = kurtosis(signal)

    # Energy
    features['energy'] = np.sum(signal ** 2)

    # First derivative
    derivative = np.diff(signal)
    features['derivative_mean'] = np.mean(derivative)
    features['derivative_std'] = np.std(derivative)

    # Dip detection (minima)
    dips, _ = find_peaks(-signal, distance=10)
    features['dip_count'] = len(dips)
    features['dip_mean_depth'] = np.mean(np.abs(signal[dips] - np.mean(signal))) if len(dips) > 0 else 0

    return features

# === Label Extraction Function ===
def extract_label(session_name):
    session_name = session_name.upper()
    if 'LOW' in session_name:
        return 'Low_Pain'
    elif 'HIGH' in session_name:
        return 'High_Pain'
    elif 'NO' in session_name or 'BASELINE' in session_name:
        return 'No_Pain'
    elif 'REST' in session_name:
        return 'No_Pain'
    else:
        return 'Unknown'

# === Main Preprocessing Loop ===
def preprocess_spo2_folder(data_dir):
    all_features = []

    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            patient_id = os.path.splitext(filename)[0]
            file_path = os.path.join(data_dir, filename)
            df = pd.read_csv(file_path)

            for col in df.columns:
                signal = df[col].dropna().values
                if len(signal) < 10:
                    continue  # skip short/noisy sessions

                features = extract_features_from_spo2(signal)
                label = extract_label(col)

                # Add metadata
                features['patient_id'] = patient_id
                features['session'] = col
                features['label'] = label

                all_features.append(features)

    return pd.DataFrame(all_features)

# === Run Script ===
if __name__ == '__main__':
    features_df = preprocess_spo2_folder(SPO2_DATA_DIR)
    features_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Done. SpO₂ features saved to: {OUTPUT_FILE}")
