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

# ========= STEP 1: DEVICE ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= STEP 2: DATA ==========
def load_and_preprocess(path):
    df = pd.read_csv(path)
    df['signal'] = df['signal'].apply(ast.literal_eval).apply(np.array)
    df['label'] = LabelEncoder().fit_transform(df['label'])
    return df

eda = load_and_preprocess("preprocessed_eda_segments.csv")
bvp = load_and_preprocess("preprocessed_bvp_segments.csv")
resp = load_and_preprocess("preprocessed_resp_segments.csv")
spo2 = load_and_preprocess("preprocessed_spo2_segments.csv")
# test data
test_eda = load_and_preprocess("preprocessed_eda_segments_test.csv")
test_bvp = load_and_preprocess("preprocessed_bvp_segments_test.csv")
test_resp = load_and_preprocess("preprocessed_resp_segments_test.csv")
test_spo2 = load_and_preprocess("preprocessed_spo2_segments_test.csv")

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
train_len = int(0.9 * len(full_dataset))
val_len = len(full_dataset) - train_len
train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])
test_dataset = PainDataset(test_eda, test_bvp, test_resp, test_spo2)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# ========= STEP 3: MODEL ==========
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=320, model_dim=64, n_heads=4, n_layers=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 320)
        x = self.proj(x)    # (B, 1, model_dim)
        x = self.encoder(x) # (B, 1, model_dim)
        return x.squeeze(1) # (B, model_dim)

class AttentionFusion(nn.Module):
    def __init__(self, input_dim=64, output_dim=128):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(4 * input_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, *features):
        x = torch.stack(features, dim=1)  # (B, 4, D)
        attn_out, _ = self.attn(x, x, x)
        x_flat = attn_out.reshape(attn_out.size(0), -1)
        return self.proj(x_flat)

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
        out, _ = self.rnn(x.unsqueeze(1))  # (B, 1, 128)
        return self.fc(out[:, -1, :])      # (B, num_classes)

class MultimodalPainClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_eda = TransformerEncoder()
        self.encoder_bvp = TransformerEncoder()
        self.encoder_resp = TransformerEncoder()
        self.encoder_spo2 = TransformerEncoder()
        self.fusion = AttentionFusion()
        self.temporal = TemporalModel()

    def forward(self, eda, bvp, resp, spo2):
        eda_feat = self.encoder_eda(eda)
        bvp_feat = self.encoder_bvp(bvp)
        resp_feat = self.encoder_resp(resp)
        spo2_feat = self.encoder_spo2(spo2)
        fused = self.fusion(eda_feat, bvp_feat, resp_feat, spo2_feat)
        return self.temporal(fused)

# ========= STEP 4: TRAIN ==========
model = MultimodalPainClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

train_losses, val_losses, val_accuracies = [], [], []
best_val_f1 = 0.0

for epoch in range(50):
    model.train()
    running_loss = 0
    for eda, bvp, resp, spo2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/50"):
        eda, bvp, resp, spo2, labels = eda.to(device), bvp.to(device), resp.to(device), spo2.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(eda, bvp, resp, spo2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    val_loss = 0
    with torch.no_grad():
        for eda, bvp, resp, spo2, labels in val_loader:
            eda, bvp, resp, spo2, labels = eda.to(device), bvp.to(device), resp.to(device), spo2.to(device), labels.to(device)
            outputs = model(eda, bvp, resp, spo2)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_losses.append(val_loss / len(val_loader))
    acc = accuracy_score(all_labels, all_preds)
    val_accuracies.append(acc)
    val_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Val Acc: {acc*100:.2f}% | F1 Score: {val_f1:.4f}")
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "best_model_transformer.pth")
        print(f"✅ Saved model with best F1: {best_val_f1:.4f}")

# ========= STEP 5: EVALUATION ==========
def evaluate_model_all(model, loader, device, set_name="Test"):
    print(f"\n{set_name} Evaluation:")
    print(f"{'Set':<10}{'Modality':<10}{'Accuracy':<10}{'F1-Score'}")

    def run_eval(forward_fn, name):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for eda, bvp, resp, spo2, labels in loader:
                eda, bvp, resp, spo2, labels = eda.to(device), bvp.to(device), resp.to(device), spo2.to(device), labels.to(device)
                outputs = forward_fn(eda, bvp, resp, spo2)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"{set_name:<10}{name:<10}{acc*100:.2f}%\t\t{f1:.4f}")

    run_eval(lambda e,b,r,s: model(e,b,r,s), "Multimodal")
    run_eval(lambda e,b,r,s: model(e, torch.zeros_like(b), torch.zeros_like(r), torch.zeros_like(s)), "EDA")
    run_eval(lambda e,b,r,s: model(torch.zeros_like(e), b, torch.zeros_like(r), torch.zeros_like(s)), "BVP")
    run_eval(lambda e,b,r,s: model(torch.zeros_like(e), torch.zeros_like(b), r, torch.zeros_like(s)), "RESP")
    run_eval(lambda e,b,r,s: model(torch.zeros_like(e), torch.zeros_like(b), torch.zeros_like(r), s), "SpO₂")

# ========= STEP 6: VISUALIZATION ==========
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss Curve")
plt.legend()
plt.show()

# ========= STEP 7: FINAL TEST ==========
model.load_state_dict(torch.load("best_model_transformer.pth"))
evaluate_model_all(model, test_loader, device, set_name="Test")
