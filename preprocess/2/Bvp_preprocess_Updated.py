import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

# === Configuration ===
BVP_DATA_DIR = '../dataset/train/Bvp'
OUTPUT_FILE = 'BVP_features.csv'

# === Feature Extraction Function ===
def extract_features_from_bvp(signal, sampling_rate=64):
    features = {}

    # Basic stats
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['min'] = np.min(signal)
    features['max'] = np.max(signal)
    features['range'] = np.max(signal) - np.min(signal)
    features['skewness'] = skew(signal)
    features['kurtosis'] = kurtosis(signal)

    # Signal energy
    features['energy'] = np.sum(signal ** 2)

    # Derivative
    derivative = np.diff(signal)
    features['derivative_mean'] = np.mean(derivative)
    features['derivative_std'] = np.std(derivative)

    # Peak analysis (heartbeats)
    peaks, _ = find_peaks(signal, distance=sampling_rate/2)  # assume ~0.5 sec between peaks
    features['peak_count'] = len(peaks)

    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / sampling_rate  # in seconds
        features['hr_mean'] = 60 / np.mean(rr_intervals)
        features['hr_std'] = np.std(60 / rr_intervals)
        features['rr_mean'] = np.mean(rr_intervals)
        features['rr_std'] = np.std(rr_intervals)
    else:
        features['hr_mean'] = 0
        features['hr_std'] = 0
        features['rr_mean'] = 0
        features['rr_std'] = 0

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
def preprocess_bvp_folder(data_dir):
    all_features = []

    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            patient_id = os.path.splitext(filename)[0]
            file_path = os.path.join(data_dir, filename)
            df = pd.read_csv(file_path)

            for col in df.columns:
                signal = df[col].dropna().values
                if len(signal) < 10:
                    continue  # skip short segments

                features = extract_features_from_bvp(signal)
                label = extract_label(col)

                # Add metadata
                features['patient_id'] = patient_id
                features['session'] = col
                features['label'] = label

                all_features.append(features)

    return pd.DataFrame(all_features)

# === Run ===
if __name__ == '__main__':
    features_df = preprocess_bvp_folder(BVP_DATA_DIR)
    features_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Done. Extracted BVP features saved to: {OUTPUT_FILE}")
