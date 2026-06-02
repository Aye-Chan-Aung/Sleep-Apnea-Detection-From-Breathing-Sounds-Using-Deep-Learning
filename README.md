# Deep Learning-Based Sleep Apnea Detection from Breathing Audio Using SE-ResNet

A compact Squeeze-and-Excitation Residual Network (SE-ResNet) for automated detection of obstructive sleep apnea events from breathing audio recordings. Converts raw audio into mel-spectrograms and classifies segments as **Apnea** or **Normal** with 76.3% accuracy and 0.818 ROC-AUC.

> **Course Project** — CSC532 Machine Learning

---

## Highlights

- **76.3% test accuracy** with patient-level evaluation (no data leakage)
- **0.818 ROC-AUC** — strong discriminative ability
- **0.87 apnea recall** — catches 87% of apnea events (clinically critical)
- **324K parameters** — lightweight enough for edge deployment
- **Patient-level splitting** — realistic generalization to unseen patients

---

## Problem

Obstructive Sleep Apnea (OSA) affects millions worldwide and is linked to cardiovascular disease, stroke, and cognitive impairment. The current diagnostic standard — overnight polysomnography — is expensive, inconvenient, and unavailable in many regions. This project explores whether deep learning can detect apnea events from breathing audio alone, enabling low-cost screening with just a microphone.

---

## Dataset

**PSG-Audio Apnea Dataset** from [Kaggle](https://www.kaggle.com/datasets/bryandarquea/psg-audio-apnea-audios)

| Property | Value |
|---|---|
| Total segments | 11,349 |
| Patients | 192 |
| Apnea segments | 5,738 (50.6%) |
| Normal segments | 5,611 (49.4%) |
| Sample rate | 16,000 Hz |
| Format | Pre-segmented `.npy` arrays |

Labels are derived from filenames: `*_ap.npy` (apnea) and `*_nap.npy` (normal). A maximum of 30 segments per file prevents any single patient from dominating the training data.

---

## Architecture

### SE-ResNet Overview

```
Input: 128 × 313 × 1 (Mel-Spectrogram)
    ↓
Conv2D (7×7, 32, stride=2) → BatchNorm → ReLU
    ↓
MaxPooling2D (3×3, stride=2)
    ↓
SE-ResBlock (32 filters)
    ↓
SE-ResBlock (64 filters, stride=2)
    ↓
SE-ResBlock (128 filters, stride=2)
    ↓
GlobalAveragePooling2D
    ↓
Dropout(0.5) → Dense(64, ReLU) → Dropout(0.3)
    ↓
Dense(1, Sigmoid) → Output
```

### Squeeze-and-Excitation Block

Each residual block includes an SE attention module that learns **which frequency channels matter most** for the current input:

1. **Squeeze** — Global average pooling compresses each channel to a single value
2. **Excite** — Two dense layers (with reduction ratio r=8) learn channel importance weights
3. **Scale** — Original feature maps are multiplied by the learned weights

This is especially useful for audio spectrograms, where different frequency bands carry different diagnostic information.

### SE-ResBlock Structure

```
Input
  ├──→ Conv2D (3×3) → BN → ReLU → Conv2D (3×3) → BN → SE Block ──→ Add → ReLU → Output
  │                                                                    ↑
  └──── Skip Connection (1×1 Conv if dimensions change) ──────────────┘
```

---

## Pipeline

### 1. Feature Extraction

Raw audio segments are converted to mel-spectrograms:

```python
SR         = 16000
N_MELS     = 128     # frequency bins
N_FFT      = 2048    # FFT window size
HOP_LENGTH = 512     # hop between windows
```

Each spectrogram (shape: `128 × 313`) is converted to dB scale and min-max normalized to `[0, 1]`.

### 2. Data Splitting (Patient-Level)

```
Train:      70% → 7,926 segments (~134 patients)
Validation: 15% → 1,705 segments (~29 patients)
Test:       15% → 1,718 segments (~29 patients)
```

**No patient appears in multiple splits.** This prevents the model from memorizing patient-specific characteristics (breathing rhythm, microphone setup, background noise) and ensures the test accuracy reflects real-world performance on unseen patients.

### 3. Data Augmentation (Training Only)

Each augmentation is applied with 50% probability per sample:

- **Frequency masking** (SpecAugment) — zeros out 5–20 random frequency bins
- **Time masking** (SpecAugment) — zeros out 5–20 random time frames
- **Random gain** — multiplies spectrogram by a factor in [0.8, 1.2]

### 4. Training

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-4 (cosine annealing) |
| Weight decay | 1e-4 |
| Batch size | 64 |
| Max epochs | 40 |
| Early stopping | Patience 12, monitoring val_accuracy |
| Class weights | Balanced (~1.01 Normal, ~0.99 Apnea) |

Training stopped at epoch 27 (early stopping). Best validation accuracy: 71.3% at epoch 15.

---

## Results

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Normal | 0.83 | 0.66 | 0.73 | 848 |
| Apnea | 0.72 | 0.87 | 0.79 | 870 |
| **Macro Avg** | **0.78** | **0.76** | **0.76** | **1,718** |

### Confusion Matrix

|  | Predicted Normal | Predicted Apnea |
|---|---|---|
| **True Normal** | 557 (65.7%) | 291 (34.3%) |
| **True Apnea** | 116 (13.3%) | 754 (86.7%) |

### Inference Strategy Comparison

| Strategy | Accuracy | AUC |
|---|---|---|
| **Standard (t=0.5)** | **76.3%** | **0.818** |
| Standard (t=0.59) | 74.2% | 0.818 |
| TTA (t=0.5) | 74.5% | 0.814 |
| TTA (t=0.59) | 73.6% | 0.814 |

**Key finding:** The simplest strategy (standard prediction, default threshold) outperformed all alternatives. Test-time augmentation and threshold optimization both degraded test performance — an honest result that validates the patient-level evaluation approach.

---

## Key Takeaways

**Why 76.3% and not higher?** Patient-level splitting. With random splitting (data leakage), accuracy would likely reach mid-80s, but that number would be misleading. Our 76.3% is an honest estimate of performance on truly unseen patients.

**Why high apnea recall matters:** In clinical screening, missing an apnea case (false negative) is more dangerous than a false alarm (false positive). The model's 0.87 apnea recall means it catches 87% of real apnea events — appropriate for a screening tool.

**Why TTA didn't help:** The augmentations introduced noise that pushed borderline predictions in the wrong direction. The clean spectrogram was the most reliable input.

---

## Project Structure

```
├── resnet-v3.ipynb          # Main training notebook
├── models/
│   └── best_model_v3.keras  # Saved best model weights
├── results/
│   ├── evaluation_v3.png    # Confusion matrix, ROC curve, distributions
│   └── training_history.png # Training/validation curves
└── README.md
```

---

## Requirements

```
tensorflow >= 2.19.0
numpy
pandas
librosa
scikit-learn
matplotlib
tqdm
```

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/sleep-apnea-se-resnet.git
cd sleep-apnea-se-resnet

# Install dependencies
pip install tensorflow numpy pandas librosa scikit-learn matplotlib tqdm

# Run on Kaggle (recommended for GPU)
# Upload resnet-v3.ipynb to Kaggle and attach the PSG-Audio dataset
```

The notebook is designed to run on **Kaggle** with GPU acceleration. It will:
1. Load and process audio from the PSG-Audio dataset
2. Extract mel-spectrograms
3. Split data at the patient level
4. Train the SE-ResNet model
5. Evaluate with multiple inference strategies
6. Generate visualizations (confusion matrix, ROC curve, training history)

---

## Version History

| Version | Accuracy | Key Changes |
|---|---|---|
| v1 (Baseline) | 73.6% | Basic ResNet, patient-level split |
| v2 | Lower | Added mixup augmentation — hurt performance |
| **v3 (Final)** | **76.3%** | SE blocks, cosine LR, no mixup, longer training |

---

## Future Improvements

- **Multi-channel input** — stack delta and delta-delta features as additional channels
- **CBAM attention** — add spatial attention alongside channel attention
- **Grad-CAM** — visualize which spectrogram regions drive predictions
- **Transfer learning** — pretrain on AudioSet for better feature representations
- **K-fold cross-validation** — more robust evaluation with only 29 test patients

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{sleepapnea2026,
  title={Deep Learning-Based Sleep Apnea Detection from Breathing Audio Using SE-ResNet},
  author={Anonymous},
  booktitle={The 14th International Conference on Advances in Information Technology (IAIT 2026)},
  year={2026}
}
```

---

## License

This project is for educational and research purposes. The PSG-Audio dataset is publicly available on [Kaggle](https://www.kaggle.com/datasets/bryandarquea/psg-audio-apnea-audios).
