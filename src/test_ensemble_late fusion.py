import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score

# === Load preprocessed feature files ===
eda_df = pd.read_csv('../preprocess/EDA_features.csv')
bvp_df = pd.read_csv('../preprocess/BVP_features.csv')
resp_df = pd.read_csv('../preprocess/RESP_features.csv')
spo2_df = pd.read_csv('../preprocess/SpO2_features.csv')

# === Encode labels ===
label_map = {'No_Pain': 0, 'Low_Pain': 1, 'High_Pain': 2}
for df in [eda_df, bvp_df, resp_df, spo2_df]:
    df['label_encoded'] = df['label'].map(label_map)

# === Define helper to split and impute ===
def split_and_impute(df):
    X = df.drop(columns=['patient_id', 'session', 'label', 'label_encoded'], errors='ignore')
    y = df['label_encoded']
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# === Train and evaluate per-modality models ===
results = {}
modality_data = {'EDA': eda_df, 'BVP': bvp_df, 'RESP': resp_df, 'SpO2': spo2_df}

for name, df in modality_data.items():
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_impute(df)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }

# === Prepare merged dataset for multimodal ===
merged_df = eda_df[['patient_id', 'session', 'label_encoded']].copy()
merged_df = merged_df.merge(eda_df.drop(columns=['label', 'label_encoded']), on=['patient_id', 'session'])
merged_df = merged_df.merge(bvp_df.drop(columns=['label', 'label_encoded']), on=['patient_id', 'session'], suffixes=('', '_BVP'))
merged_df = merged_df.merge(resp_df.drop(columns=['label', 'label_encoded']), on=['patient_id', 'session'], suffixes=('', '_RESP'))
merged_df = merged_df.merge(spo2_df.drop(columns=['label', 'label_encoded']), on=['patient_id', 'session'], suffixes=('', '_SpO2'))

X_all = merged_df.drop(columns=['patient_id', 'session', 'label_encoded'])
y_all = merged_df['label_encoded']

# === Impute and split for multimodal model ===
imputer = SimpleImputer(strategy='mean')
X_all = imputer.fit_transform(X_all)
X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# === Track training and validation loss ===
train_losses = []
val_losses = []

for n in range(10, 110, 10):
    model = RandomForestClassifier(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    train_loss = 1 - accuracy_score(y_train, model.predict(X_train))
    val_loss = 1 - accuracy_score(y_val, model.predict(X_val))
    train_losses.append(train_loss)
    val_losses.append(val_loss)

# === Final multimodal model ===
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)
results['Multimodal'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred, average='weighted')
}

# === Save model after training ===
joblib.dump(final_model, 'multimodal_rf_model.pkl')
print("Multimodal Random Forest model saved as 'multimodal_rf_model.pkl'")

# === Plot training vs validation loss ===
plt.figure(figsize=(8, 5))
plt.plot(range(10, 110, 10), train_losses, label='Training Loss')
plt.plot(range(10, 110, 10), val_losses, label='Validation Loss')
plt.xlabel('Number of Trees')
plt.ylabel('Loss (1 - Accuracy)')
plt.title('Training vs Validation Loss (Multimodal Random Forest)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Print results ===
print("\n📊 Classification Results (Accuracy & F1-Score):\n")
print(pd.DataFrame(results).T.round(4))
