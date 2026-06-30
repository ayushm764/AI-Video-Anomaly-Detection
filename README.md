# AI Video Anomaly Detection

An unsupervised deep learning system for detecting anomalous activities in CCTV surveillance videos using a **Convolutional Autoencoder**. The model learns normal pedestrian behavior from surveillance frames and flags unusual events using reconstruction error.

This project is built for **frame-level anomaly detection** on the **UCSD Ped2** dataset, where normal scenes contain pedestrians walking and anomalies include bicycles, skateboarders, vehicles, and other non-pedestrian activities.

---

## Project Overview

Surveillance cameras continuously generate large amounts of video data. Manually monitoring this footage is not scalable, and collecting labeled examples for every possible anomaly is difficult because abnormal events are rare and unpredictable.

This project solves the problem using an **unsupervised learning approach**:

- Train the model only on normal surveillance frames.
- Reconstruct each input frame using a Convolutional Autoencoder.
- Compute reconstruction error between the original and reconstructed frame.
- Flag frames with high reconstruction error as anomalous.

The core idea is simple:  
a model trained only on normal scenes reconstructs normal frames well, but struggles to reconstruct abnormal frames, causing a higher reconstruction error.

---

## Key Features

- Unsupervised CCTV video anomaly detection
- PyTorch-based Convolutional Autoencoder
- Frame-level anomaly classification
- Reconstruction-error based anomaly scoring
- MSE + SSIM combined reconstruction loss
- Percentile-based threshold calibration
- UCSD Ped2 dataset support
- Evaluation using Precision, Recall, F1-score, and ROC-AUC
- Visualization of anomaly scores, error distributions, ROC curve, and evaluation metrics
- Modular code structure for training, evaluation, visualization, and inference

---

## Tech Stack

| Category | Tools / Libraries |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Computer Vision | OpenCV, Pillow |
| Numerical Computing | NumPy |
| Evaluation | scikit-learn |
| Visualization | Matplotlib |
| Model Type | Convolutional Autoencoder |
| Dataset | UCSD Ped2 |
| Task | Unsupervised Video Anomaly Detection |

---

## Dataset

### UCSD Ped2 Dataset

The project uses the **UCSD Pedestrian 2 (UCSD Ped2)** dataset, a standard benchmark dataset for video anomaly detection.

| Property | Description |
|---|---|
| Dataset | UCSD Ped2 |
| Scene | Pedestrian walkway |
| Camera Type | Fixed overhead surveillance camera |
| Normal Activity | People walking |
| Anomalous Activity | Bicycles, skateboarders, vehicles, wheelchairs |
| Training Data | Normal frames only |
| Testing Data | Normal + anomalous frames |
| Frame Format | `.tif` grayscale images |
| Ground Truth | `.bmp` anomaly masks |

### Dataset Structure

```text
UCSDped2/
│
├── Train/
│   ├── Train001/
│   ├── Train002/
│   └── ...
│
└── Test/
    ├── Test001/
    ├── Test001_gt/
    ├── Test002/
    ├── Test002_gt/
    └── ...
```

The training set contains only normal pedestrian activity.  
The test set contains both normal and anomalous frames.

Ground-truth pixel-level masks are converted into frame-level labels:

```text
If any pixel in the ground-truth mask is anomalous:
    frame label = 1
else:
    frame label = 0
```

---

## Problem Statement

The goal is to automatically detect unusual activities in surveillance videos without requiring manually labeled examples of every possible anomaly.

In real-world CCTV monitoring:

- Anomalies are rare.
- Anomalies are unpredictable.
- Manual labeling is expensive.
- Supervised models may fail on unseen anomaly types.
- Continuous human monitoring is not scalable.

Therefore, this project uses an unsupervised reconstruction-based approach that learns only from normal data.

---

## Approach

### Why Unsupervised Learning?

In anomaly detection, we often have a large amount of normal data but very limited anomalous data. Instead of training a classifier to recognize specific anomaly classes, the model learns the normal distribution of surveillance frames.

During inference:

```text
Low reconstruction error  → Normal frame
High reconstruction error → Anomalous frame
```

---

## Model Architecture

The model is a symmetric **Convolutional Autoencoder** consisting of:

1. Encoder  
2. Bottleneck latent representation  
3. Decoder  

### High-Level Architecture

```text
Input CCTV Frame
        ↓
Preprocessing
        ↓
Encoder
        ↓
Latent Vector
        ↓
Decoder
        ↓
Reconstructed Frame
        ↓
Reconstruction Error
        ↓
Threshold Comparison
        ↓
Normal / Anomaly
```

---

## Encoder

The encoder compresses a grayscale surveillance frame into a compact latent representation.

```text
Input: 1 × 64 × 64 grayscale frame

Conv2D: 1 → 32
BatchNorm
LeakyReLU

Conv2D: 32 → 64
BatchNorm
LeakyReLU

Conv2D: 64 → 128
BatchNorm
LeakyReLU

Conv2D: 128 → 256
BatchNorm
LeakyReLU

Flatten: 256 × 4 × 4 = 4096

Fully Connected:
4096 → 256

Output:
256-dimensional latent vector
```

The encoder learns compressed representations of normal pedestrian scenes.

---

## Decoder

The decoder reconstructs the original frame from the latent representation.

```text
Input:
256-dimensional latent vector

Fully Connected:
256 → 4096

Reshape:
256 × 4 × 4

ConvTranspose2D: 256 → 128
BatchNorm
LeakyReLU

ConvTranspose2D: 128 → 64
BatchNorm
LeakyReLU

ConvTranspose2D: 64 → 32
BatchNorm
LeakyReLU

ConvTranspose2D: 32 → 1
Sigmoid

Output:
Reconstructed 1 × 64 × 64 grayscale frame
```

The final Sigmoid activation ensures output pixel values remain in the range `[0, 1]`.

---

## Loss Function

The model is trained using a combined reconstruction loss:

```text
Loss = MSE(original, reconstructed) + 0.5 × (1 - SSIM(original, reconstructed))
```

### Why MSE?

MSE captures pixel-level reconstruction error.

### Why SSIM?

SSIM captures structural and perceptual differences such as:

- Edges
- Object shapes
- Contrast
- Texture
- Local visual structure

Using both MSE and SSIM helps the model detect anomalies that may not create large pixel-wise differences but still change the visual structure of the scene.

---

## Training Pipeline

```text
Load normal training frames
        ↓
Resize and normalize frames
        ↓
Train Convolutional Autoencoder
        ↓
Compute reconstruction loss
        ↓
Optimize using Adam
        ↓
Save trained model
```

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-5 |
| Batch Size | 32 |
| Epochs | 50 |
| Image Size | 64 × 64 |
| Latent Dimension | 256 |
| Training Type | Unsupervised |
| Training Data | Normal frames only |

---

## Threshold Calibration

After training, the model is run on all normal training frames to calculate reconstruction errors.

The anomaly threshold is selected using a percentile-based method:

```text
threshold = 95th percentile of training reconstruction errors
```

This means most normal frames fall below the threshold, while frames with unusually high reconstruction error are flagged as anomalies.

```text
if reconstruction_error > threshold:
    prediction = anomaly
else:
    prediction = normal
```

---

## Evaluation

The model is evaluated at frame level using ground-truth labels derived from UCSD Ped2 masks.

### Evaluation Metrics

| Metric | Value |
|---|---:|
| Precision | ~84% |
| Recall | ~92% |
| F1-score | ~87.8% |
| ROC-AUC | ~0.70 |

These results were obtained after threshold tuning to balance false alarms and missed detections.

### Metric Interpretation

- **Precision (~84%)**: Most frames flagged as anomalous were truly anomalous.
- **Recall (~92%)**: The model detected most actual anomalous frames.
- **F1-score (~87.8%)**: The model achieved a strong balance between precision and recall.
- **ROC-AUC (~0.70)**: The reconstruction score showed useful separation between normal and anomalous frames.

---

## Results Summary

The system achieved strong frame-level anomaly detection performance while using only normal frames during training.

```text
Precision : ~84%
Recall    : ~92%
F1-score  : ~87.8%
ROC-AUC   : ~0.70
```

The model is especially useful for surveillance use cases where detecting most anomalies is more important than completely eliminating false alarms.

---

## Visualizations

The evaluation pipeline generates the following visual outputs:

- Training loss curve
- Reconstruction error distribution
- Frame-level anomaly score timeline
- Metrics summary chart
- ROC curve

### Example Visualization Outputs

```text
outputs/
│
├── training_loss.png
├── error_distribution.png
├── frame_error_timeline.png
├── metrics_summary.png
└── roc_curve.png
```

These plots help analyze how well the reconstruction error separates normal and anomalous frames.

---

## Project Workflow

```text
1. Load UCSD Ped2 dataset
2. Preprocess grayscale frames
3. Train Convolutional Autoencoder on normal frames
4. Compute training reconstruction errors
5. Calibrate anomaly threshold
6. Run inference on test frames
7. Compute reconstruction error for each frame
8. Classify frames as normal or anomalous
9. Evaluate using Precision, Recall, F1-score, and ROC-AUC
10. Generate result visualizations
```

---

## Complete System Architecture

```text
                ┌─────────────────────────┐
                │     CCTV / Video Input   │
                └───────────┬─────────────┘
                            │
                            ▼
                ┌─────────────────────────┐
                │   Frame Extraction       │
                │   OpenCV / Dataset Loader│
                └───────────┬─────────────┘
                            │
                            ▼
                ┌─────────────────────────┐
                │   Preprocessing          │
                │   Resize 64×64           │
                │   Normalize [0,1]        │
                │   Tensor Conversion      │
                └───────────┬─────────────┘
                            │
                            ▼
                ┌─────────────────────────┐
                │ Conv Autoencoder Model   │
                │ Encoder + Decoder        │
                └───────────┬─────────────┘
                            │
                            ▼
                ┌─────────────────────────┐
                │ Reconstruction Error     │
                │ MSE + SSIM Error         │
                └───────────┬─────────────┘
                            │
                            ▼
                ┌─────────────────────────┐
                │ Threshold Comparison     │
                │ Error > Threshold?       │
                └───────┬─────────┬───────┘
                        │         │
                    Normal     Anomaly
                        │         │
                        │         ▼
                        │  ┌─────────────────────┐
                        │  │ Alert / Flag Frame   │
                        │  │ Save Result          │
                        │  └─────────────────────┘
                        │
                        ▼
                ┌─────────────────────────┐
                │ Logs + Visualizations    │
                └─────────────────────────┘
```

---

## Old Approach and Improvement

Initially, the project explored a CNN + LSTM Autoencoder approach:

```text
Video frames
    ↓
CNN / ResNet feature extraction
    ↓
LSTM Autoencoder
    ↓
Temporal reconstruction error
    ↓
Anomaly prediction
```

However, this approach produced poor ROC-AUC performance.

### Why the Initial Approach Did Not Work Well

- Pretrained CNN features were not well aligned with grayscale surveillance footage.
- UCSD Ped2 is relatively small for training a temporal LSTM-based model.
- Feature-level reconstruction did not clearly separate normal and anomalous frames.
- Sequence alignment and temporal windowing made the pipeline more complex.
- The anomaly score did not rank abnormal frames higher than normal frames consistently.

### Final Improvement

The final model uses a simpler and more stable Convolutional Autoencoder trained directly on surveillance frames.

Benefits of the improved approach:

- Direct frame-level reconstruction
- More interpretable anomaly score
- Easier threshold calibration
- Better suited for the UCSD Ped2 fixed-camera setting
- Lower complexity than CNN + LSTM

---

## Folder Structure

```text
AI-Video-Anomaly-Detection/
│
├── data/
│   ├── dataset.py
│   └── preprocessing.py
│
├── models/
│   ├── autoencoder.py
│   └── detector.py
│
├── evaluation/
│   └── metrics.py
│
├── utils/
│   └── visualization.py
│
├── outputs/
│   ├── autoencoder.pth
│   ├── training_loss.png
│   ├── error_distribution.png
│   ├── frame_error_timeline.png
│   ├── metrics_summary.png
│   └── roc_curve.png
│
├── UCSDped2/
│   ├── Train/
│   └── Test/
│
├── inference_video.py
├── main.py
├── config.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ayushm764/AI-Video-Anomaly-Detection.git
cd AI-Video-Anomaly-Detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

For Windows:

```bash
venv\Scripts\activate
```

For macOS / Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

Example `requirements.txt`:

```text
torch
torchvision
opencv-python
numpy
pillow
scikit-learn
matplotlib
tqdm
```

---

## How to Run

### 1. Prepare Dataset

Download and place the UCSD Ped2 dataset in the project directory:

```text
AI-Video-Anomaly-Detection/
└── UCSDped2/
    ├── Train/
    └── Test/
```

---

### 2. Train and Evaluate

Run the main pipeline:

```bash
python main.py
```

This will:

- Load and preprocess the dataset
- Train the Convolutional Autoencoder
- Calibrate the anomaly threshold
- Evaluate on the test set
- Save model weights
- Generate visualization plots

---

### 3. Run Inference on a Video

```bash
python inference_video.py --video path/to/video.mp4
```

Expected output:

```text
Frame 001: Normal
Frame 002: Normal
Frame 003: Anomaly
...
```

The script can be extended to save anomalous frames or suspicious video clips.

---

## Inference Logic

```python
frame = preprocess(input_frame)
reconstructed = model(frame)
error = compute_reconstruction_error(frame, reconstructed)

if error > threshold:
    label = "Anomaly"
else:
    label = "Normal"
```

---

## Deployment Plan

The project can be deployed as a real-time CCTV anomaly detection system.

### Deployment Architecture

```text
CCTV Camera / RTSP Stream
        ↓
OpenCV Video Capture
        ↓
Frame Sampling
        ↓
Preprocessing
        ↓
PyTorch / ONNX Model Inference
        ↓
Reconstruction Error Calculation
        ↓
Threshold Decision
        ↓
Alert Generation
        ↓
Dashboard + Logs + Saved Clips
```

### Deployment Steps

1. Train the model offline using normal CCTV footage.
2. Save model weights and calibrated threshold.
3. Build a FastAPI inference service.
4. Use OpenCV to read frames from CCTV or RTSP streams.
5. Run model inference on sampled frames.
6. Calculate reconstruction error.
7. Raise an alert if anomaly score remains high across multiple frames.
8. Save suspicious frames or video clips.
9. Display alerts and score timeline on a dashboard.
10. Monitor false positives and recalibrate the threshold when needed.

### Real-Time Alert Logic

To avoid false alarms from a single noisy frame, alerts should be triggered only when anomaly scores remain high for multiple consecutive frames.

```text
if anomaly_score > threshold for N consecutive frames:
    raise alert
else:
    continue monitoring
```

---

## Possible Dashboard Features

A real-time dashboard can include:

- Live camera feed
- Current anomaly score
- Threshold line
- Normal / anomaly status
- Recent alerts
- Saved suspicious clips
- Frame-level anomaly timeline
- Camera-wise logs

Suggested stack:

| Component | Technology |
|---|---|
| Backend API | FastAPI / Flask |
| Video Processing | OpenCV |
| Frontend | React / HTML / CSS / JavaScript |
| Database | SQLite / PostgreSQL / MongoDB |
| Model Inference | PyTorch / ONNX |
| Deployment | Docker |

---

## Limitations

- The model is best suited for fixed-camera surveillance scenes.
- It performs frame-level detection and does not fully understand long-term event context.
- Sudden lighting changes, shadows, camera noise, or blur may increase reconstruction error.
- The threshold may need recalibration for different cameras or environments.
- Autoencoders may sometimes reconstruct visually similar anomalies too well.
- The current model detects anomalous frames but does not precisely localize the anomaly region.
- Real-world deployment would require testing on larger and more diverse CCTV datasets.

---

## Future Improvements

- Add ConvLSTM for better temporal motion modeling.
- Use optical flow to capture abnormal movement patterns.
- Generate anomaly heatmaps for better explainability.
- Add object detection to identify specific objects such as bicycles, vehicles, or skateboards.
- Use adaptive thresholds for different cameras and lighting conditions.
- Evaluate on larger datasets such as ShanghaiTech, Avenue, and UCF-Crime.
- Optimize inference using TorchScript, ONNX, or TensorRT.
- Build a real-time web dashboard for monitoring alerts.
- Save suspicious clips automatically for security review.
- Add model drift monitoring and periodic threshold recalibration.

---

## Resume Summary

```text
AI Video Anomaly Detection | Python, PyTorch, OpenCV, Convolutional Autoencoder, NumPy

• Developed an unsupervised computer vision model for CCTV anomaly detection using a Convolutional Autoencoder trained on normal UCSD Ped2 surveillance frames.
• Designed an end-to-end PyTorch pipeline for frame preprocessing, model training, reconstruction-error based anomaly scoring, threshold calibration, and evaluation.
• Applied MSE + SSIM loss to improve detection of pixel-level and structural anomalies such as bicycles, skateboarders, and vehicles in pedestrian scenes.
• Achieved approx. 84% precision, 92% recall, 87.8% F1-score, and ~0.70 ROC-AUC after threshold tuning for balanced frame-level anomaly detection.
```

---

## Interview Explanation

This project detects anomalies in CCTV surveillance footage using unsupervised learning. I trained a Convolutional Autoencoder only on normal pedestrian frames from the UCSD Ped2 dataset. During testing, the model reconstructs each frame and calculates reconstruction error. Since the model has learned normal scenes, it reconstructs normal frames well, but anomalous frames such as bicycles or skateboarders produce higher reconstruction error.

I used a combined MSE + SSIM loss because MSE captures pixel-level differences while SSIM captures structural differences such as edges and object shapes. After training, I calibrated the anomaly threshold using training reconstruction errors and evaluated the model at frame level. The final model achieved approximately 84% precision, 92% recall, 87.8% F1-score, and around 0.70 ROC-AUC.

---

## Author

**Ayush Mittal**  
B.Tech Computer Science and Engineering  
Specialization: Artificial Intelligence and Machine Learning

---

## License

This project is intended for academic and learning purposes.
