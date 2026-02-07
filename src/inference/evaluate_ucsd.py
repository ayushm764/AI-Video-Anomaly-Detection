"""
Evaluation script for UCSD Ped2 anomaly detection
Uses trained LSTM Autoencoder to detect anomalies
-------------------------------------------------
Pipeline:
- Load extracted CNN feature clips (.npy)
- Compute reconstruction loss using LSTM Autoencoder
- Learn threshold from TRAIN data
- Detect anomalies on TEST data
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm


# -------------------------------
# Device configuration
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------
# LSTM Autoencoder Model
# -------------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMAutoencoder, self).__init__()

        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded


# -------------------------------
# Load trained model
# -------------------------------
INPUT_DIM = 512      # CNN feature size
HIDDEN_DIM = 256

model = LSTMAutoencoder(INPUT_DIM, HIDDEN_DIM)

# IMPORTANT: path relative to project root
MODEL_PATH = "lstm_autoencoder_ucsd.pth"

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("Model loaded successfully")


# -------------------------------
# Loss function
# -------------------------------
criterion = nn.MSELoss(reduction="mean")


# -------------------------------
# TRAIN DATA → Threshold learning
# -------------------------------
train_path = "data/processed/UCSDped2/Train_features"
train_losses = []

print("Computing TRAIN reconstruction losses...")

for file in tqdm(sorted(os.listdir(train_path))):
    if not file.endswith(".npy"):
        continue

    file_path = os.path.join(train_path, file)

    # Shape: (num_clips, seq_len, feature_dim)
    clips = np.load(file_path)
    clips = torch.tensor(clips, dtype=torch.float32).to(device)

    with torch.no_grad():
        recon = model(clips)
        loss = criterion(recon, clips)

    train_losses.append(loss.item())

train_losses = np.array(train_losses)

mean_loss = train_losses.mean()
std_loss = train_losses.std()

THRESHOLD = mean_loss + 3 * std_loss

print(f"\nThreshold calculated:")
print(f"Mean Loss = {mean_loss:.6f}")
print(f"Std Loss  = {std_loss:.6f}")
print(f"Threshold = {THRESHOLD:.6f}")


# -------------------------------
# TEST DATA → Anomaly detection
# -------------------------------
test_path = "data/processed/UCSDped2/Test_features"
test_losses = []

print("\nComputing TEST reconstruction losses...")

for file in tqdm(sorted(os.listdir(test_path))):
    if not file.endswith(".npy"):
        continue

    file_path = os.path.join(test_path, file)

    clips = np.load(file_path)
    clips = torch.tensor(clips, dtype=torch.float32).to(device)

    with torch.no_grad():
        recon = model(clips)
        loss = criterion(recon, clips)

    test_losses.append(loss.item())

test_losses = np.array(test_losses)


# -------------------------------
# Anomaly decision
# -------------------------------
anomaly_flags = test_losses > THRESHOLD

num_anomalies = anomaly_flags.sum()
print(f"\nDetected {num_anomalies} anomalous test videos out of {len(test_losses)}")


# -------------------------------
# Visualization
# -------------------------------
plt.figure(figsize=(12, 5))

plt.plot(test_losses, label="Test Reconstruction Loss", marker="o")
plt.axhline(
    THRESHOLD,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Anomaly Threshold"
)

plt.xlabel("Test Video Index")
plt.ylabel("Reconstruction Loss")
plt.title("UCSD Ped2 Video Anomaly Detection")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
