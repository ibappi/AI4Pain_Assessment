# preprocess the Electrodermal Activity (EDA) data
import os
import pandas as pd
import numpy as np


Eda_data_dir = '../dataset/train/Eda'
segment_length = 5094  # <--- Updated to match your test set

def extract_label(session_name):
    if 'LOW' in session_name:
        return 'Low_Pain'
    elif 'HIGH' in session_name:
        return 'High_Pain'
    else:
        return 'No_Pain'

def preprocess_patients_file(file_path, segment_length=5094):
    df = pd.read_csv(file_path)
    segments = []
    for col in df.columns:
        label = extract_label(col)
        signal = df[col].dropna().values
        # Normalize
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
        # Only one segment if signal == segment_length, or multiple if longer
        for i in range(0, len(signal) - segment_length + 1, segment_length):
            segment = signal[i:i + segment_length]
            segments.append({
                'signal': segment.tolist(),  # Ensure list, not numpy array
                'label': label,
                'session': col,
                'patient': os.path.basename(file_path).split('.')[0]
            })
    return segments

all_segments = []
for filename in os.listdir(Eda_data_dir):
    if filename.endswith('.csv'):
        filepath = os.path.join(Eda_data_dir, filename)
        patient_segments = preprocess_patients_file(filepath, segment_length)
        all_segments.extend(patient_segments)

print(f"Total segments collected: {len(all_segments)}")

df_segments = pd.DataFrame(all_segments)
output_path = f'preprocessed_eda_segments_{segment_length}.csv'
df_segments.to_csv(output_path, index=False)
print(f"Saved preprocessed data to {output_path}")
