from torch.utils.data import DataLoader
from src.dataset.ucsd_dataset import UCSDPed2Dataset


def get_dataloader(root_dir, batch_size=4, shuffle=True, num_workers=2):
    """
    Creates a PyTorch DataLoader for UCSD Ped2 dataset.

    Args:
        root_dir (str):
            Path to dataset split directory
            Example: data/processed/UCSDped2/Train

        batch_size (int):
            Number of clips per batch

        shuffle (bool):
            Whether to shuffle data (True for training)

        num_workers (int):
            Number of parallel workers for data loading

    Returns:
        DataLoader: PyTorch DataLoader object
    """

    # Initialize dataset
    dataset = UCSDPed2Dataset(root_dir)

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # speeds up GPU transfer later
    )

    return dataloader
