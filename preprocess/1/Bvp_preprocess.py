import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BVP_data_dir = './dataset/train/Bvp'
segment_length = 5094  # <--- Updated to match your test set

# === HELPER FUNCTIONS ===

def plot_raw_signals(file_path, n_sessions=3):
    """Visualize first few raw signals for sanity check."""
    df = pd.read_csv(file_path)
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(df.columns):
        if i >= n_sessions:
            break
        signal = df[col].dropna().values
        plt.plot(signal, label=f"{col} (len={len(signal)})")
    plt.title(f"Raw signals from {os.path.basename(file_path)}")
    plt.xlabel("Sample Index")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

example_file = os.path.join(BVP_data_dir, os.listdir(BVP_data_dir)[0])
plot_raw_signals(example_file, n_sessions=3)


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

def preprocess_patients_file(file_path, segment_length=5094):
    df = pd.read_csv(file_path)
    segments = []
    for col in df.columns:
        label = extract_label(col)
        signal = df[col].dropna().values
        print(f"{os.path.basename(file_path)}, Column {col}: Length = {len(signal)}")
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
for filename in os.listdir(BVP_data_dir):
    if filename.endswith('.csv'):
        filepath = os.path.join(BVP_data_dir, filename)
        patient_segments = preprocess_patients_file(filepath, segment_length)
        all_segments.extend(patient_segments)

print(f"Total segments collected: {len(all_segments)}")

df_segments = pd.DataFrame(all_segments)
output_path = f'preprocessed_bvp_segments_{segment_length}.csv'
df_segments.to_csv(output_path, index=False)
print(f"Saved preprocessed data to {output_path}")
