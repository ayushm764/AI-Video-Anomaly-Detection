import os
import cv2
import numpy as np
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================

RAW_DATA_DIR = "data/raw/UCSDped2"
PROCESSED_DATA_DIR = "data/processed/UCSDped2"

# Preprocessing parameters
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
CLIP_LENGTH = 16      # number of frames per clip
STRIDE = 8            # sliding window step

# =========================
# UTILITY FUNCTIONS
# =========================

def load_and_preprocess_frame(frame_path):
    """
    Load a single frame and apply preprocessing:
    - Read image
    - Resize
    - Normalize to [0, 1]
    """
    frame = cv2.imread(frame_path)

    if frame is None:
        raise ValueError(f"Failed to read frame: {frame_path}")

    # Convert BGR (OpenCV default) to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize frame
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # Normalize pixel values
    frame = frame.astype(np.float32) / 255.0

    return frame


def frames_to_clips(frames, clip_length, stride):
    """
    Convert a list of frames into temporal clips using sliding window.
    """
    clips = []

    for start_idx in range(0, len(frames) - clip_length + 1, stride):
        clip = frames[start_idx:start_idx + clip_length]
        clips.append(np.stack(clip, axis=0))  # shape: [T, H, W, C]

    return clips


# =========================
# MAIN PREPROCESSING LOGIC
# =========================

def preprocess_split(split_name):
    """
    Preprocess either Train or Test split.
    """
    input_split_dir = os.path.join(RAW_DATA_DIR, split_name)
    output_split_dir = os.path.join(PROCESSED_DATA_DIR, split_name)

    os.makedirs(output_split_dir, exist_ok=True)

    video_folders = sorted(os.listdir(input_split_dir))

    print(f"\nProcessing {split_name} split...")
    print(f"Found {len(video_folders)} video folders")

    for video_folder in tqdm(video_folders, desc=f"{split_name} videos"):
        video_path = os.path.join(input_split_dir, video_folder)

        if not os.path.isdir(video_path):
            continue

        # Read and sort frame files
        frame_files = sorted([
            f for f in os.listdir(video_path)
            if f.lower().endswith((".tif", ".jpg", ".png"))
        ])

        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(video_path, frame_file)
            frame = load_and_preprocess_frame(frame_path)
            frames.append(frame)

        if len(frames) < CLIP_LENGTH:
            continue  # skip very short videos

        # Create temporal clips
        clips = frames_to_clips(frames, CLIP_LENGTH, STRIDE)

        # Save clips
        video_output_dir = os.path.join(output_split_dir, video_folder)
        os.makedirs(video_output_dir, exist_ok=True)

        for idx, clip in enumerate(clips):
            clip_path = os.path.join(video_output_dir, f"clip_{idx:04d}.npy")
            np.save(clip_path, clip)


def main():
    """
    Entry point.
    """
    preprocess_split("Train")
    preprocess_split("Test")

    print("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    main()
