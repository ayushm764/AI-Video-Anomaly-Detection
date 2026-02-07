import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================

DATA_ROOT = "data/processed/UCSDped2"
OUTPUT_ROOT = "data/processed/UCSDped2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_LENGTH = 16
FEATURE_DIM = 512

# =========================
# IMAGE TRANSFORM
# =========================

# ResNet expects ImageNet-style normalization
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# LOAD PRETRAINED CNN
# =========================

def load_cnn():
    """
    Load pretrained ResNet18 and remove classification layer.
    Output: feature vector of size 512
    """
    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity()  # remove classifier
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model.to(DEVICE)

# =========================
# FEATURE EXTRACTION
# =========================

def extract_features(split):
    """
    Extract spatial features for Train or Test split
    """
    input_dir = os.path.join(DATA_ROOT, split)
    output_dir = os.path.join(OUTPUT_ROOT, f"{split}_features")
    os.makedirs(output_dir, exist_ok=True)

    cnn = load_cnn()

    for video_folder in sorted(os.listdir(input_dir)):
        video_path = os.path.join(input_dir, video_folder)
        if not os.path.isdir(video_path):
            continue

        all_clips_features = []

        clip_files = sorted(os.listdir(video_path))

        print(f"Processing {split}/{video_folder}...")

        for clip_file in tqdm(clip_files):
            clip_path = os.path.join(video_path, clip_file)

            # Load clip: (16, H, W, C)
            clip = np.load(clip_path)

            clip_features = []

            for frame in clip:
                frame = transform(frame).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    feature = cnn(frame)  # (1, 512)

                clip_features.append(feature.squeeze(0).cpu().numpy())

            clip_features = np.stack(clip_features)  # (16, 512)
            all_clips_features.append(clip_features)

        all_clips_features = np.stack(all_clips_features)  # (num_clips, 16, 512)

        save_path = os.path.join(output_dir, f"{video_folder}.npy")
        np.save(save_path, all_clips_features)

        print(f"Saved: {save_path} | Shape: {all_clips_features.shape}")

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    extract_features("Train")
    extract_features("Test")
