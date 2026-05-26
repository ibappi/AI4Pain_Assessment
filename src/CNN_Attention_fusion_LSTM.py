# Full dataset : Train + Test + Validation with all, used full segmentation full lenght not 320 only
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
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, matthews_corrcoef)
import itertools

# ========== STEP 1: DATA PREPARATION ==========
def load_and_preprocess(path):
    df = pd.read_csv(path)
    df['signal'] = df['signal'].apply(ast.literal_eval).apply(np.array)
    df['label'] = LabelEncoder().fit_transform(df['label'])  # NP=0, LP=1, HP=2
    return df

eda = load_and_preprocess("../dataset/train/preprocessed_eda_segments_320.csv")
bvp = load_and_preprocess("../dataset/train/preprocessed_bvp_segments_320.csv")
resp = load_and_preprocess("../dataset/train/preprocessed_rasp_segments_320.csv")
spo2 = load_and_preprocess("../dataset/train/preprocessed_spo2_segments_320.csv")

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
CLASS_NAMES = ["NP","LP","HP"]  # must match your LabelEncoder order

# --------------

def plot_confusion_matrix(cm, classes=CLASS_NAMES, normalize=False, title='Confusion matrix'):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-12)
    plt.figure(figsize=(4.5,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title); plt.xlabel('Predicted'); plt.ylabel('True')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes); plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2. if cm.size else 0.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout(); plt.colorbar(); plt.show()

def collect_preds_and_probs(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for eda,bvp,resp,spo2,labels in loader:
            eda,bvp,resp,spo2,labels = eda.to(device), bvp.to(device), resp.to(device), spo2.to(device), labels.to(device)
            logits = model(eda,bvp,resp,spo2)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = probs.argmax(axis=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)
            y_prob.append(probs)
    return np.array(y_true), np.array(y_pred), np.vstack(y_prob)

def summarize_metrics(y_true, y_pred, y_prob):
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    kappa    = cohen_kappa_score(y_true, y_pred)
    mcc      = matthews_corrcoef(y_true, y_pred)
    fpr,tpr,roc_auc = {},{},{}
    for i in range(len(CLASS_NAMES)):
        fpr[i], tpr[i], _ = roc_curve((np.array(y_true)==i).astype(int), y_prob[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    macro_auroc = np.mean(list(roc_auc.values()))
    pr, rc, ap = {},{},{}
    for i in range(len(CLASS_NAMES)):
        pr[i], rc[i], _ = precision_recall_curve((np.array(y_true)==i).astype(int), y_prob[:,i])
        ap[i] = average_precision_score((np.array(y_true)==i).astype(int), y_prob[:,i])
    macro_auprc = np.mean(list(ap.values()))
    print(f"Balanced Acc: {bal_acc:.4f} | Macro-F1: {macro_f1:.4f} | Micro-F1: {micro_f1:.4f} | "
          f"Macro-AUROC: {macro_auroc:.4f} | Macro-AUPRC: {macro_auprc:.4f} | "
          f"Cohen's κ: {kappa:.4f} | MCC: {mcc:.4f}")
    return (fpr,tpr,roc_auc), (pr,rc,ap), {'bal_acc':bal_acc,'macro_f1':macro_f1,
            'micro_f1':micro_f1,'macro_auroc':macro_auroc,'macro_auprc':macro_auprc,
            'kappa':kappa,'mcc':mcc}

def plot_roc_curves(fpr, tpr, roc_auc, classes=CLASS_NAMES, title="ROC Curves"):
    plt.figure(figsize=(5.5,4.5))
    for i,c in enumerate(classes):
        plt.plot(fpr[i], tpr[i], label=f"{c} (AUROC={roc_auc[i]:.3f})")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend(); plt.tight_layout(); plt.show()

def plot_pr_curves(pr, rc, ap, classes=CLASS_NAMES, title="Precision-Recall Curves"):
    plt.figure(figsize=(5.5,4.5))
    for i,c in enumerate(classes):
        plt.plot(rc[i], pr[i], label=f"{c} (AUPRC={ap[i]:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(title); plt.legend(); plt.tight_layout(); plt.show()


# -----------------


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

for epoch in range(250):
    model.train()
    running_loss = 0
    for eda, bvp, resp, spo2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/250"):
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
        torch.save(model.state_dict(), "best_model_cnn_lstm_302_v1.pth")
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

#
# ========== STEP 7: FINAL TEST AND EVALUATION ==========
# evaluate_model_all(model, val_loader, device, set_name="Validation")
# evaluate_model_all(model, test_loader, device, set_name="Test")


# ========== STEP 7: FINAL TEST AND EVALUATION (rich) ==========
# 7a) Your existing modality ablation table
evaluate_model_all(model, val_loader, device, set_name="Validation")
evaluate_model_all(model, test_loader, device, set_name="Test")

# 7b) Validation: per-class report, confusion matrices, ROC/PR, summary metrics
y_true_v, y_pred_v, y_prob_v = collect_preds_and_probs(model, val_loader, device)
print("\nValidation classification report:")
print(classification_report(y_true_v, y_pred_v, target_names=CLASS_NAMES, digits=4))

cm_v = confusion_matrix(y_true_v, y_pred_v, labels=[0,1,2])
plot_confusion_matrix(cm_v, title="Validation Confusion Matrix (counts)")
plot_confusion_matrix(cm_v, normalize=True, title="Validation Confusion Matrix (row-normalized)")

(fpr_v,tpr_v,roc_auc_v), (pr_v,rc_v,ap_v), summary_v = summarize_metrics(y_true_v, y_pred_v, y_prob_v)
plot_roc_curves(fpr_v, tpr_v, roc_auc_v, title="Validation ROC Curves (OvR)")
plot_pr_curves(pr_v, rc_v, ap_v, title="Validation PR Curves (OvR)")

# 7c) Test: per-class report, confusion matrices, ROC/PR, summary metrics
y_true_t, y_pred_t, y_prob_t = collect_preds_and_probs(model, test_loader, device)
print("\nTest classification report:")
print(classification_report(y_true_t, y_pred_t, target_names=CLASS_NAMES, digits=4))

cm_t = confusion_matrix(y_true_t, y_pred_t, labels=[0,1,2])
plot_confusion_matrix(cm_t, title="Test Confusion Matrix (counts)")
plot_confusion_matrix(cm_t, normalize=True, title="Test Confusion Matrix (row-normalized)")

(fpr_t,tpr_t,roc_auc_t), (pr_t,rc_t,ap_t), summary_t = summarize_metrics(y_true_t, y_pred_t, y_prob_t)
plot_roc_curves(fpr_t, tpr_t, roc_auc_t, title="Test ROC Curves (OvR)")
plot_pr_curves(pr_t, rc_t, ap_t, title="Test PR Curves (OvR)")


# result for only 1023 segemntation till accuracy
# Epoch 250/250: 100%|██████████| 80/80 [00:00<00:00, 129.43it/s]
# Train Loss: 0.0286 | Val Loss: 0.0204 | Val Acc: 99.37% | F1 Score: 0.8958
# Validation Evaluation:
# Set		Modality	Accuracy	F1-Score
# Validation	Multimodal	99.37%		0.8958
# Validation	EDA         96.53%		0.5184
# Validation	BVP       	96.21%		0.3269
# Validation	RESP      	96.21%		0.3269
# Validation	SpO₂      	96.21%		0.3269

# Test Evaluation:
# Set		Modality	Accuracy	F1-Score
# Test	Multimodal	98.43%		0.7908
# Test	EDA       	96.24%		0.5126
# Test	BVP       	94.98%		0.3248
# Test	RESP      	94.98%		0.3248
# Test	SpO₂      	94.98%		0.3248
#
# Process finished with exit code 0

# result for only 3935 segemntation till accuracy
# Epoch 250/250: 100%|██████████| 2/2 [00:00<00:00, 105.26it/s]
# Train Loss: 0.0004 | Val Loss: 0.0005 | Val Acc: 100.00% | F1 Score: 1.0000
# Validation Evaluation:
# Set		Modality	Accuracy	F1-Score
# Validation	Multimodal	100.00%		1.0000
# Validation	EDA       	100.00%		1.0000
# Validation	BVP       	100.00%		1.0000
# Validation	RESP      	100.00%		1.0000
# Validation	SpO₂      	100.00%		1.0000
# Test Evaluation:
# Set		Modality	Accuracy	F1-Score
# Test	Multimodal	100.00%		1.0000
# Test	EDA       	100.00%		1.0000
# Test	BVP       	100.00%		1.0000
# Test	RESP      	100.00%		1.0000
# Test	SpO₂      	100.00%		1.0000
#
# Process finished with exit code 0

# result for only 5094 segemntation till accuracy
# Epoch 250/250: 100%|██████████| 1/1 [00:00<00:00, 100.00it/s]
# Train Loss: 0.0017 | Val Loss: 0.0014 | Val Acc: 100.00% | F1 Score: 1.0000
# Train Loss: 0.0017 | Val Loss: 0.0014 | Val Acc: 100.00% | F1 Score: 1.0000
# Validation Evaluation:
# Set		Modality	Accuracy	F1-Score
# Validation	Multimodal	100.00%		1.0000
# Validation	EDA       	100.00%		1.0000
# Validation	BVP       	100.00%		1.0000
# Validation	RESP      	100.00%		1.0000
# Validation	SpO₂      	100.00%		1.0000
# Test Evaluation:
# Set		Modality	Accuracy	F1-Score
# Test	Multimodal	100.00%		1.0000
# Test	EDA       	100.00%		1.0000
# Test	BVP       	100.00%		1.0000
# Test	RESP      	100.00%		1.0000
# Test	SpO₂      	100.00%		1.0000
#
# Process finished with exit code 0

# Updated result:
# Train Loss: 0.3477 | Val Loss: 0.4471 | Val Acc: 84.20% | F1 Score: 0.5786
# Validation Evaluation:
# Set		Modality	Accuracy	F1-Score
# Validation	Multimodal	84.20%		0.5786
# Validation	EDA       	82.13%		0.3243
# Validation	BVP       	81.80%		0.3000
# Validation	RESP      	81.80%		0.3000
# Validation	SpO₂      	81.80%		0.3000
# Test Evaluation:
# Set		Modality	Accuracy	F1-Score
# Test	Multimodal	81.36%		0.5395
# Test	EDA       	80.09%		0.3141
# Test	BVP       	80.09%		0.2965
# Test	RESP      	80.09%		0.2965
# Test	SpO₂      	80.09%		0.2965
#
# Validation classification report:
#               precision    recall  f1-score   support
#
#           NP     0.4909    0.3942    0.4372       137
#           LP     0.4400    0.3235    0.3729       136
#           HP     0.9031    0.9495    0.9257      1227
#
#     accuracy                         0.8420      1500
#    macro avg     0.6113    0.5557    0.5786      1500
# weighted avg     0.8235    0.8420    0.8310      1500
#
# Balanced Acc: 0.5557 | Macro-F1: 0.5786 | Micro-F1: 0.8420 | Macro-AUROC: 0.8616 | Macro-AUPRC: 0.5797 | Cohen's κ: 0.4432 | MCC: 0.4482
#
# Test classification report:
#               precision    recall  f1-score   support
#
#           NP     0.4615    0.3243    0.3810       148
#           LP     0.3909    0.2848    0.3295       151
#           HP     0.8781    0.9401    0.9081      1203
#
#     accuracy                         0.8136      1502
#    macro avg     0.5769    0.5164    0.5395      1502
# weighted avg     0.7881    0.8136    0.7980      1502
#
# Balanced Acc: 0.5164 | Macro-F1: 0.5395 | Micro-F1: 0.8136 | Macro-AUROC: 0.8550 | Macro-AUPRC: 0.5757 | Cohen's κ: 0.3765 | MCC: 0.3835
#
# Process finished with exit code 0

