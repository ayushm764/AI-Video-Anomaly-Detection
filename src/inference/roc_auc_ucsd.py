"""
ROC-AUC Evaluation Script for UCSD Ped2
--------------------------------------
Evaluates a trained LSTM Autoencoder using reconstruction loss
and computes ROC curve + AUC score.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm


# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------
# LSTM Autoencoder Model
# -------------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded


# -------------------------------
# Load trained model
# -------------------------------
INPUT_DIM = 512
HIDDEN_DIM = 256

MODEL_PATH = "lstm_autoencoder_ucsd.pth"

model = LSTMAutoencoder(INPUT_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("Model loaded successfully")


# -------------------------------
# Helper: load .npy feature clips
# -------------------------------
def load_feature_clips(folder_path):
    clips = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".npy"):
            clip = np.load(os.path.join(folder_path, file))
            clips.append(clip)
    return np.array(clips)


# -------------------------------
# Paths
# -------------------------------
TRAIN_PATH = "data/processed/UCSDped2/Train_features"
TEST_PATH  = "data/processed/UCSDped2/Test_features"

criterion = nn.MSELoss(reduction="mean")


# -------------------------------
# Compute TRAIN losses (threshold)
# -------------------------------
train_losses = []

print("Computing TRAIN reconstruction losses...")
for file in tqdm(sorted(os.listdir(TRAIN_PATH))):
    if not file.endswith(".npy"):
        continue

    clips = np.load(os.path.join(TRAIN_PATH, file))
    clips = torch.tensor(clips, dtype=torch.float32).to(device)

    with torch.no_grad():
        recon = model(clips)
        loss = criterion(recon, clips)

    train_losses.append(loss.item())

train_losses = np.array(train_losses)
mean_loss = train_losses.mean()
std_loss  = train_losses.std()
threshold = mean_loss + 3 * std_loss

print("\nThreshold calculated:")
print(f"Mean Loss = {mean_loss:.6f}")
print(f"Std Loss  = {std_loss:.6f}")
print(f"Threshold = {threshold:.6f}")


# -------------------------------
# Compute TEST losses
# -------------------------------
test_losses = []

print("\nComputing TEST reconstruction losses...")
for file in tqdm(sorted(os.listdir(TEST_PATH))):
    if not file.endswith(".npy"):
        continue

    clips = np.load(os.path.join(TEST_PATH, file))
    clips = torch.tensor(clips, dtype=torch.float32).to(device)

    with torch.no_grad():
        recon = model(clips)
        loss = criterion(recon, clips)

    test_losses.append(loss.item())

test_losses = np.array(test_losses)


# -------------------------------
# Ground Truth Labels (UCSD Ped2)
# -------------------------------
# Test001–006 → Normal (0)
# Test007–012 → Anomaly (1)
test_labels = np.array([0]*6 + [1]*6)


# -------------------------------
# ROC-AUC
# -------------------------------
fpr, tpr, _ = roc_curve(test_labels, test_losses)
roc_auc = auc(fpr, tpr)

print(f"\nROC-AUC Score: {roc_auc:.4f}")


# -------------------------------
# ROC Curve Plot
# -------------------------------
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - UCSD Ped2")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
