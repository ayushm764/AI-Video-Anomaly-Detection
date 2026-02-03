import os
import numpy as np
import torch
from torch.utils.data import Dataset


class UCSDPed2Dataset(Dataset):
    """
    PyTorch Dataset for UCSD Ped2 anomaly detection.

    This dataset:
    - Loads video clips stored as .npy files
    - Each clip represents a short temporal sequence of frames
    - No labels are used (unsupervised learning)
    """

    def __init__(self, root_dir):
        """
        Constructor for the dataset.

        Args:
            root_dir (str):
                Path to the processed dataset split.
                Example:
                - data/processed/UCSDped2/Train
                - data/processed/UCSDped2/Test
        """
        self.clip_paths = []

        # Iterate through all video sequence folders (Train001, Train002, ...)
        for seq_folder in sorted(os.listdir(root_dir)):
            seq_path = os.path.join(root_dir, seq_folder)

            # Skip files; only process directories
            if not os.path.isdir(seq_path):
                continue

            # Collect all .npy clip files inside each sequence folder
            for file in sorted(os.listdir(seq_path)):
                if file.endswith(".npy"):
                    self.clip_paths.append(os.path.join(seq_path, file))

    def __len__(self):
        """
        Returns:
            int: Total number of clips in the dataset
        """
        return len(self.clip_paths)

    def __getitem__(self, idx):
        """
        Loads and returns one clip.

        Args:
            idx (int): Index of the clip

        Returns:
            torch.Tensor:
                Shape: (T, C, H, W)
                where:
                - T = number of frames (e.g., 16)
                - C = channels (3 for RGB)
                - H, W = frame height and width
        """
        # Load clip from disk (shape: T, H, W, C)
        clip = np.load(self.clip_paths[idx])

        # Convert NumPy array to PyTorch tensor
        clip = torch.from_numpy(clip).float()

        # Rearrange dimensions to PyTorch standard: (T, C, H, W)
        clip = clip.permute(0, 3, 1, 2)

        return clip
