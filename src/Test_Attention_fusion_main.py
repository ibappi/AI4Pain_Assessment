# Test code for pain assessment classification
import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# ========== STEP 1: DEVICE ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== STEP 2: DATA PREPARATION ==========
def load_and_preprocess(path):
    df = pd.read_csv(path)
    df['signal'] = df['signal'].apply(ast.literal_eval).apply(np.array)
    df['label'] = LabelEncoder().fit_transform(df['label'])  # NP=0, LP=1, HP=2
    return df

test_eda = load_and_preprocess("preprocessed_eda_segments_test.csv")
test_bvp = load_and_preprocess("preprocessed_bvp_segments_test.csv")
test_resp = load_and_preprocess("preprocessed_resp_segments_test.csv")
test_spo2 = load_and_preprocess("preprocessed_spo2_segments_test.csv")

# ========== STEP 3: DATASET ==========
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

test_dataset = PainDataset(test_eda, test_bvp, test_resp, test_spo2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# ========== STEP 4: MODEL ARCHITECTURE ==========
class ModalityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
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

# ========== STEP 5: LOAD TRAINED MODEL ==========
model = MultimodalPainClassifier().to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# ========== STEP 6: EVALUATION ==========
def evaluate_model_all(model, loader, device, set_name="Test"):
    print(f"{set_name} Evaluation:")
    print(f"Set\t\tModality\tAccuracy\tF1-Score")

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
        print(f"{set_name}\t{name:<10}\t{acc * 100:.2f}%\t\t{f1:.4f}")

    run_eval(lambda eda, bvp, resp, spo2: model(eda, bvp, resp, spo2), "Multimodal")
    run_eval(lambda eda, bvp, resp, spo2: model(eda, torch.zeros_like(bvp), torch.zeros_like(resp), torch.zeros_like(spo2)), "EDA")
    run_eval(lambda eda, bvp, resp, spo2: model(torch.zeros_like(eda), bvp, torch.zeros_like(resp), torch.zeros_like(spo2)), "BVP")
    run_eval(lambda eda, bvp, resp, spo2: model(torch.zeros_like(eda), torch.zeros_like(bvp), resp, torch.zeros_like(spo2)), "RESP")
    run_eval(lambda eda, bvp, resp, spo2: model(torch.zeros_like(eda), torch.zeros_like(bvp), torch.zeros_like(resp), spo2), "SpO₂")

evaluate_model_all(model, test_loader, device, set_name="Test")
