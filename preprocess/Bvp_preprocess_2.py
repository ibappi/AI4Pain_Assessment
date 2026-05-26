import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BVP_data_dir = '../dataset/train/Bvp'
segment_length = 320  # <--- Updated to match your test set

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

def preprocess_patients_file(file_path, segment_length=320):
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

# ----- CLASS BALANCE ------------------------
ax = df_segments['label'].value_counts().plot(kind='bar', rot=0, title='Class balance (segments)')
ax.set_xlabel('Class'); ax.set_ylabel('#Segments'); plt.tight_layout(); plt.show()

# ----- PER-PATIENT SEGMENT COUNTS -----
counts = df_segments.groupby('patient').size()
counts.plot(kind='box', title='Segments per patient (distribution)'); plt.tight_layout(); plt.show()

# Per-subject vs Global Normalization Check----------------------
def seg_stat(x):
    return float(np.mean(x)) if len(x)>0 else np.nan

df_tmp = df_segments.copy()
df_tmp['mean_amp'] = df_tmp['signal'].apply(lambda s: np.mean(s))

# Compare per-patient z-norm vs global z-norm on the summary stat
gmu, gsd = df_tmp['mean_amp'].mean(), df_tmp['mean_amp'].std() + 1e-6
df_tmp['mean_amp_gz'] = (df_tmp['mean_amp'] - gmu) / gsd
df_tmp['mean_amp_pz'] = df_tmp.groupby('patient')['mean_amp'].transform(lambda v: (v - v.mean())/(v.std()+1e-6))

plt.figure(figsize=(8,4))
df_tmp['mean_amp_gz'].plot(kind='kde', label='Global z-norm');
df_tmp['mean_amp_pz'].plot(kind='kde', label='Per-patient z-norm');
plt.title('Effect of normalization on a simple segment stat'); plt.legend(); plt.tight_layout(); plt.show()

# 5) Stationarity & autocorrelation (justifies 320-sample window)----------------------
def plot_stationarity(sig, w=64):
    sig = np.asarray(sig, dtype=np.float32)
    rm = pd.Series(sig).rolling(w, min_periods=1).mean()
    rv = pd.Series(sig).rolling(w, min_periods=1).var()
    fig, ax = plt.subplots(1,2, figsize=(10,3))
    ax[0].plot(sig, alpha=0.7, label='signal'); ax[0].plot(rm, label='roll mean')
    ax[0].plot(rv, label='roll var'); ax[0].legend(); ax[0].set_title('Rolling stats')

    # ACF
    acf = np.correlate(sig - sig.mean(), sig - sig.mean(), mode='full')
    acf = acf[acf.size//2:]; acf /= acf[0] + 1e-9
    ax[1].plot(acf[:200]); ax[1].set_title('Autocorr (first 200 lags)')
    plt.tight_layout(); plt.show()

example_sig = np.array(df_segments.iloc[0]['signal'], dtype=np.float32)
plot_stationarity(example_sig, w=64)

# Spectral profile (Welch PSD) by class------------------------
from scipy.signal import welch

def segment_psd(sig, fs=64):
    f, pxx = welch(sig, fs=fs, nperseg=min(256, len(sig)))
    return f, pxx

class_psd = {}
for lab, grp in df_segments.groupby('label'):
    psds = []
    for s in grp.sample(min(100, len(grp)), random_state=0)['signal']:
        f, p = segment_psd(np.array(s, dtype=np.float32))
        psds.append(p)
    psds = np.vstack(psds) if len(psds)>0 else None
    class_psd[lab] = (f, psds)

plt.figure(figsize=(8,5))
for lab,(f,psds) in class_psd.items():
    if psds is None: continue
    m = psds.mean(axis=0); se = psds.std(axis=0)/np.sqrt(psds.shape[0]+1e-9)
    plt.plot(f, m, label=lab); plt.fill_between(f, m-se, m+se, alpha=0.2)
plt.xlabel('Hz'); plt.ylabel('PSD'); plt.title('BVP PSD by class'); plt.legend(); plt.tight_layout(); plt.show()

output_path = f'preprocessed_bvp_segments_analysis_{segment_length}.csv'

df_segments.to_csv(output_path, index=False)
print(f"Saved preprocessed data to {output_path}")
