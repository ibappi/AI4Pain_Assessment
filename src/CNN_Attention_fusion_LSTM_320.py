# Full dataset : Train + Test + Validation with all
# Architecture: Preprocess + 1D CNN encoder + Attention Fusion + LSTM + Classification
import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========== STEP 1: DATA PREPARATION ==========
def load_and_preprocess(path):
    df = pd.read_csv(path)
    df['signal'] = df['signal'].apply(ast.literal_eval).apply(np.array)
    df['label'] = LabelEncoder().fit_transform(df['label'])  # NP=0, LP=1, HP=2
    return df

eda = load_and_preprocess("../dataset/train/preprocessed_eda_segments.csv")
bvp = load_and_preprocess("../dataset/train/preprocessed_bvp_segments.csv")
resp = load_and_preprocess("../dataset/train/preprocessed_resp_segments.csv")
spo2 = load_and_preprocess("../dataset/train/preprocessed_spo2_segments.csv")

# ========== STEP 2: DATASET CLASS ==========
class PainDataset(Dataset):
    def __init__(self, eda, bvp, resp, spo2):
        self.eda = eda['signal'].tolist()
        self.bvp = bvp['signal'].tolist()
        self.resp = resp['signal'].tolist()
        self.spo2 = spo2['signal'].tolist()
        self.labels = eda['label'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.eda[idx], dtype=torch.float32),
            torch.tensor(self.bvp[idx], dtype=torch.float32),
            torch.tensor(self.resp[idx], dtype=torch.float32),
            torch.tensor(self.spo2[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

full_dataset = PainDataset(eda, bvp, resp, spo2)
train_len = int(0.8 * len(full_dataset))
val_len = int(0.1 * len(full_dataset))
test_len  = len(full_dataset) - train_len - val_len
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_len, val_len, test_len])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# ========== STEP 3: MODEL ==========
class ModalityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        return x.squeeze(-1)

class AttentionFusion(nn.Module):
    def __init__(self, input_dim=32, output_dim=128):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(4 * input_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, *features):
        x = torch.cat(features, dim=1)
        return self.fusion(x)

class TemporalModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_classes=3):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out, _ = self.rnn(x.unsqueeze(1))
        return self.fc(out[:, -1, :])

class MultimodalPainClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_eda = ModalityEncoder()
        self.encoder_bvp = ModalityEncoder()
        self.encoder_resp = ModalityEncoder()
        self.encoder_spo2 = ModalityEncoder()
        self.fusion = AttentionFusion()
        self.temporal = TemporalModel()

    def forward(self, eda, bvp, resp, spo2):
        eda_feat = self.encoder_eda(eda)
        bvp_feat = self.encoder_bvp(bvp)
        resp_feat = self.encoder_resp(resp)
        spo2_feat = self.encoder_spo2(spo2)
        fused = self.fusion(eda_feat, bvp_feat, resp_feat, spo2_feat)
        output = self.temporal(fused)
        return output

# ========== STEP 4: TRAINING ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalPainClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

train_losses, val_losses, val_accuracies = [], [], []

best_val_f1 = 0.0  # <-- before the epoch loop

for epoch in range(150):
    model.train()
    running_loss = 0
    for eda, bvp, resp, spo2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/150"):
        eda, bvp, resp, spo2, labels = eda.to(device), bvp.to(device), resp.to(device), spo2.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(eda, bvp, resp, spo2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    model.eval()
    correct, total, val_loss = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for eda, bvp, resp, spo2, labels in val_loader:
            eda, bvp, resp, spo2, labels = eda.to(device), bvp.to(device), resp.to(device), spo2.to(device), labels.to(device)
            outputs = model(eda, bvp, resp, spo2)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(correct / total)
    val_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_accuracies[-1]*100:.2f}% | F1 Score: {val_f1:.4f}")

    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "best_model_cnn_lstm.pth")
        print(f"Saved model with new best F1: {best_val_f1:.4f}")

# ========== STEP 5: EVALUATION FUNCTION ==========
def evaluate_model_all(model, loader, device, set_name="Validation"):
    print(f"{set_name} Evaluation:")
    print(f"Set\t\tModality\tAccuracy\tF1-Score")

    def run_eval(forward_fn, name):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for eda, bvp, resp, spo2, labels in loader:
                eda, bvp, resp, spo2, labels = eda.to(device), bvp.to(device), resp.to(device), spo2.to(
                    device), labels.to(device)
                outputs = forward_fn(eda, bvp, resp, spo2)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"{set_name}\t{name:<10}\t{acc * 100:.2f}%\t\t{f1:.4f}")
        return acc, f1

    # 1. Multimodal
    run_eval(lambda eda, bvp, resp, spo2: model(eda, bvp, resp, spo2), "Multimodal")
    # 2. EDA only
    run_eval(
        lambda eda, bvp, resp, spo2: model(eda, torch.zeros_like(bvp), torch.zeros_like(resp), torch.zeros_like(spo2)),
        "EDA")
    # 3. BVP only
    run_eval(
        lambda eda, bvp, resp, spo2: model(torch.zeros_like(eda), bvp, torch.zeros_like(resp), torch.zeros_like(spo2)),
        "BVP")
    # 4. RESP only
    run_eval(
        lambda eda, bvp, resp, spo2: model(torch.zeros_like(eda), torch.zeros_like(bvp), resp, torch.zeros_like(spo2)),
        "RESP")
    # 5. SpO₂ only
    run_eval(
        lambda eda, bvp, resp, spo2: model(torch.zeros_like(eda), torch.zeros_like(bvp), torch.zeros_like(resp), spo2),
        "SpO₂")

# ========== STEP 6: VISUALIZATION ==========
plt.figure(figsize=(6,4))
# plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss Curves")
plt.legend()
plt.show()

# ========== STEP 7: FINAL TEST AND EVALUATION ==========
evaluate_model_all(model, val_loader, device, set_name="Validation")
evaluate_model_all(model, test_loader, device, set_name="Test")

# Epoch 50/50: 100%|██████████| 376/376 [00:03<00:00, 118.89it/s]
# Train Loss: 0.4580 | Val Loss: 0.4579 | Val Acc: 82.73% | F1 Score: 0.4946
# Validation Evaluation:
# Set		Modality	Accuracy	F1-Score
# Validation	Multimodal	82.73%		0.4946
# Validation	EDA       	81.27%		0.3706
# Validation	BVP       	80.33%		0.2970
# Validation	RESP      	80.33%		0.2970
# Validation	SpO₂      	80.33%		0.2970
# Test Evaluation:
# Set		Modality	Accuracy	F1-Score
# Test	Multimodal	81.16%		0.4271
# Test	EDA       	81.09%		0.3639
# Test	BVP       	80.36%		0.2970
# Test	RESP      	80.36%		0.2970
# Test	SpO₂      	80.36%		0.2970

# Epoch 100/100: 100%|██████████| 376/376 [00:03<00:00, 121.24it/s]
# Train Loss: 0.4194 | Val Loss: 0.4653 | Val Acc: 82.53% | F1 Score: 0.5662
# Saved model with new best F1: 0.5662
# Validation Evaluation:
# Set		Modality	Accuracy	F1-Score
# Validation	Multimodal	82.53%		0.5662
# Validation	EDA       	81.33%		0.3876
# Validation	BVP       	80.73%		0.2978
# Validation	RESP      	49.87%		0.2977
# Validation	SpO₂      	80.73%		0.2978
# Test Evaluation:
# Set		Modality	Accuracy	F1-Score
# Test	Multimodal	81.29%		0.5618
# Test	EDA       	79.89%		0.3742
# Test	BVP       	79.16%		0.2946
# Test	RESP      	52.13%		0.3123
# Test	SpO₂      	79.16%		0.2946
#
# Process finished with exit code 0

