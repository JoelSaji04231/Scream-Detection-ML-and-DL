# Scream-Detection-ML-and-DL

An intelligent audio classification system that detects screams in audio files using both traditional Machine Learning (ML) and Deep Learning (DL) approaches. The system achieves high accuracy using ensemble predictions from multiple trained models.

##  Table of Contents
- [Features](#features)
- [Models](#models)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Requirements](#requirements)

##  Features

- **Dual Approach**: Compares ML (SVM, Random Forest, Logistic Regression) and DL (CNN, CNN-LSTM) models as separate approaches to show performance differences between traditional and deep learning methods
- **Feature Extraction**: Comprehensive audio feature extraction including:
  - 13 MFCC coefficients (mean & std)
  - RMS Energy
  - Zero Crossing Rate
  - Spectral Centroid
  - Spectral Rolloff
- **Audio Preprocessing**: Advanced audio cleaning with silence removal and normalization
- **Spectrogram Generation**: Creates both regular and mel-spectrograms for deep learning models
- **Ensemble Predictions**: Consensus-based decision making from multiple models
- **Class Imbalance Handling**: Implements class weighting for better minority class detection
- **Comprehensive Analysis**: Audio visualization and statistical analysis tools

##  Models

### Machine Learning Models
1. **Support Vector Machine (SVM)** - 96.98% accuracy (**Best Model**)
   - Uses RBF kernel with optimized hyperparameters
   - Class-weighted for imbalanced data
2. **Random Forest** - 96.00% accuracy
   - 100 estimators with balanced class weights
3. **Logistic Regression** - 93.86% accuracy
   - L2 regularization with class balancing

### Deep Learning Models
1. **CNN (Regular Spectrogram)** - 96.20% accuracy
   - Uses standard STFT spectrograms
3. **CNN (Mel-Spectrogram)** - 94.05% accuracy
   - Uses mel-scaled spectrograms
5. **CNN-LSTM** - 86.74% accuracy
   - Combines spatial (CNN) and temporal (LSTM) features
   - Uses mel-spectrograms with class weighting

##  Dataset

The system is trained on a combined dataset of:
- **Original Dataset**: Scream and non-scream audio samples by [aananehsansiam](https://www.kaggle.com/datasets/aananehsansiam/audio-dataset-of-scream-and-non-scream)
- **ESC-50**: Environmental Sound Classification dataset (2000 samples)
- **Total Samples**: 5,128 audio files
  - Scream: 1,583 samples (30.87%)
  - Non-scream: 3,545 samples (69.13%)

**Class Distribution**: The dataset has a 1:2.24 imbalance ratio, which is addressed through class weighting in all models.

##  Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for deep learning models)
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/CrimeAlert.git
cd CrimeAlert
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download ESC-50 dataset** (if not already present)
   - Place the ESC-50 dataset in the `ESC-50-master/` directory

##  Usage

### 1. Feature Extraction & Dataset Preparation

Extract features from audio files and generate spectrograms:

```bash
python extract_features.py
```

This creates:
- `audio_features.csv` - Features for ML models
- `data/cnn_spectrograms.npy` - Regular spectrograms
- `data/cnn_mel_spectrograms.npy` - Mel-spectrograms
- `data/cnn_labels.npy` - Labels for DL models

### 2. Train Models

**Train Machine Learning models:**
```bash
python train_ml.py
```

**Train Deep Learning models:**
```bash
python train_cnn.py          # CNN with regular spectrograms
python train_cnn_mel.py      # CNN with mel-spectrograms
python train_cnn_lstm.py     # CNN-LSTM model
```

### 3. Make Predictions

**Analyze a single audio file:**
```bash
python main.py --file "path/to/audio.wav"
```

**Use default test file:**
```bash
python main.py
```

**Individual model inference:**
```bash
# SVM inference
python inference_svm.py

# CNN inference
python inference_cnn.py
```

### 4. Compare Models

Evaluate all trained models on the test set:
```bash
python compare_models.py
```

### 5. Audio Analysis & Visualization

Generate comprehensive audio analysis and visualizations:
```bash
python analyze_audio.py
```

This creates visualizations for:
- Waveforms
- Spectrograms
- Mel-spectrograms
- MFCC statistics
- Feature comparisons (RMS, ZCR, Spectral features)

## Project Structure

```
CrimeAlert/
│
├── extract_features.py          # Feature extraction & spectrogram generation
├── train_ml.py                  # Train ML models (SVM, RF, LR)
├── train_cnn.py                 # Train CNN with regular spectrograms
├── train_cnn_mel.py             # Train CNN with mel-spectrograms
├── train_cnn_lstm.py            # Train CNN-LSTM model
├── inference_svm.py             # SVM inference script
├── inference_cnn.py             # CNN inference script
├── main.py                      # Main prediction interface (ensemble)
├── compare_models.py            # Model comparison & evaluation
├── analyze_audio.py             # Audio analysis & visualization
│
├── models/                      # Trained model files
│   ├── svm_esc50_pipeline.pkl
│   ├── random_forest_esc50_pipeline.pkl
│   ├── logistic_esc50_pipeline.pkl
│   ├── cnn_model.pth
│   ├── cnn_mel_model.pth
│   └── cnn_lstm_model.pth
│
├── data/                        # Preprocessed data for DL models
│   ├── cnn_spectrograms.npy
│   ├── cnn_mel_spectrograms.npy
│   └── cnn_labels.npy
│
├── Converted_Separately/        # Audio dataset
│   ├── scream/
│   └── non_scream/
│
├── ESC-50-master/               # ESC-50 dataset
│   └── audio/
│
├── audio_features.csv           # Extracted features for ML models
└── README.md                    # This file
```

##  Model Performance

### Machine Learning Models (Test Set Performance)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| SVM | 96.98% | 97% | 97% | 97% | 99.40% |
| Random Forest | 96.00% | 96% | 96% | 96% | 99.05% |
| Logistic Regression | 93.86% | 94% | 94% | 94% | 98.03% |

### Deep Learning Models

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **CNN (Spectrogram)** | 96.20% | 96% | 96% | 96% | 99.52% |
| CNN (Mel-Spectrogram) | 94.05% | 94% | 94% | 94% | 98.43% |
| CNN-LSTM | 86.74% | 90% | 87% | 87% | 97.14% |

### SVM Detailed Performance (Best Model)

```
Classification Report:
              precision    recall  f1-score   support

  non_scream       0.99      0.97      0.98       709
      scream       0.93      0.98      0.95       317

    accuracy                           0.97      1026
   macro avg       0.96      0.97      0.97      1026
weighted avg       0.97      0.97      0.97      1026

Confusion Matrix:
[[685  24]
 [  7 310]]
```

##  Requirements

```
pandas>=1.3.0
numpy>=1.21.0
librosa>=0.9.0
scikit-learn>=1.0.0
joblib>=1.1.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
torch>=1.10.0
torchaudio>=0.10.0
```

**For GPU acceleration:**
```
torch>=1.10.0+cu113
torchaudio>=0.10.0+cu113
```

##  How It Works

### 1. **Audio Preprocessing**
   - Load audio files at 22,050 Hz sample rate
   - Remove silence using librosa's trim function
   - Normalize amplitude to [-1, 1] range
   - Handle NaN/Inf values

### 2. **Feature Extraction**
   - **For ML Models**: Extract 34 statistical features
     - 26 MFCC features (13 coefficients × 2 statistics)
     - 8 additional audio features (RMS, ZCR, Spectral features)
   - **For DL Models**: Generate 128×128 spectrograms

### 3. **Training**
   - Apply class weighting to handle imbalanced data
   - Use StandardScaler for feature normalization (ML)
   - Train-test split: 80%-20%
   - Cross-validation for hyperparameter tuning

### 4. **Prediction**
   - Load pre-trained models
   - Extract features from input audio
   - Run predictions through multiple models
   - Provide ensemble consensus with confidence scores

##  Key Features of the System

- **Robust Audio Cleaning**: Handles noisy audio with silence removal and normalization
- **Class Imbalance Handling**: Uses class weighting (pos_weight=2.24 for DL models)
- **GPU Acceleration**: Automatic GPU detection and CUDA support for PyTorch models
- **Mixed Precision Training**: AMP (Automatic Mixed Precision) for faster training
- **Early Stopping**: Prevents overfitting with patience-based early stopping
- **Comprehensive Evaluation**: Multiple metrics (Accuracy, Precision, Recall, F1, ROC-AUC, MCC)

##  Notes

- The SVM model performs best overall with 96.98% accuracy
- Ensemble predictions from multiple models provide more reliable results
- The system handles both short clips and longer audio files
- GPU is recommended but not required for inference
- Training deep learning models requires CUDA-capable GPU


##  Author

Joel Saji

##  Acknowledgments

- ESC-50 dataset for environmental sounds
- [aananehsansiam](https://www.kaggle.com/datasets/aananehsansiam/audio-dataset-of-scream-and-non-scream) for scream dataset
- librosa library for audio processing
- PyTorch for deep learning framework
- scikit-learn for machine learning algorithms

