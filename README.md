# Driver Drowsiness Detection

A real-time driver drowsiness detection system that uses **computer vision** and **deep learning** to monitor a driver's eyes through a webcam. When closed eyes are detected, an audible alarm is triggered to alert the driver.

> **Note:** This is a research/educational prototype. It should not be used as a sole safety mechanism while driving.

## Overview

The system captures live video from a webcam, detects the driver's eyes using OpenCV's Haar Cascade classifier, and classifies each detected eye region as **open** or **closed** using a trained CNN model. If drowsiness (closed eyes) is detected, a buzzer sound plays via Pygame.

## Features

- Real-time webcam-based eye detection
- Deep learning classification of open vs. closed eyes
- Audible alarm on drowsiness detection
- Transfer learning with **InceptionV3** backbone
- Data augmentation for robust training
- End-to-end pipeline: data prep → training → live inference

## Tech Stack

| Layer | Technology |
|-------|------------|
| Computer vision | OpenCV (Haar Cascade) |
| Deep learning | TensorFlow / Keras |
| Model architecture | InceptionV3 (transfer learning) + custom CNN |
| Audio alert | Pygame |
| Data prep | NumPy, tqdm |

## Project Structure

```
DriverDrowsinessDetection/
├── Data Preparation.ipynb       # Organize eye image dataset
├── data_prep_drowsiness .ipynb  # Split open/closed eye images
├── Model Training.ipynb         # Train InceptionV3-based model
├── Trial-ModelTraining.ipynb    # Experimental training runs
├── main.ipynb                   # Live webcam inference + alarm
└── README.md
```

## Workflow

### 1. Data Preparation

Eye images are organized into **open** and **closed** categories and split into train/validation/test sets using the data preparation notebooks.

### 2. Model Training

- **Base model:** InceptionV3 (pre-trained on ImageNet)
- **Input size:** 80×80 RGB eye crops
- **Augmentation:** ImageDataGenerator (rotation, zoom, flip)
- **Callbacks:** EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- **Training accuracy:** ~94% (train), ~91% (test)

### 3. Live Detection (`main.ipynb`)

1. Webcam captures frames in a loop
2. Haar Cascade detects eye regions in each frame
3. Each eye crop is resized to 80×80 and normalized
4. The trained model predicts open (0) or closed (1)
5. If closed eyes are detected, an alarm sound plays

## Getting Started

### Prerequisites

- Python 3.8+
- Webcam
- GPU recommended for training (CPU works for inference)

### Installation

```bash
git clone https://github.com/OjasAdhikari11/DriverDrowsinessDetection.git
cd DriverDrowsinessDetection

pip install tensorflow opencv-python numpy pygame jupyter tqdm
```

### Dataset

This project uses a dataset of labeled eye images (open/closed). You will need to download or prepare eye image data and organize it as described in `Data Preparation.ipynb`. The trained model file (`.h5`) is not included in the repository.

### Run

1. Run `Data Preparation.ipynb` to set up the dataset
2. Run `Model Training.ipynb` to train and save the model
3. Update the model path and alarm sound path in `main.ipynb`
4. Run `main.ipynb` and allow webcam access

## Model Performance

| Metric | Value |
|--------|-------|
| Training accuracy | ~94.6% |
| Test accuracy | ~90.8% |
| Validation accuracy | ~92.6% |

## Limitations

- Requires good lighting and a clear view of the driver's face
- Haar Cascade eye detection can fail with glasses, head turns, or low light
- Alarm sound path is hardcoded and must be updated for your environment
- Trained model weights are not included — train locally first
- Not suitable for production driver safety systems without further validation

