# VocalGuard — Voice-Based Stress Detection (Applied AI)

VocalGuard is an applied audio ML project focused on detecting **stress vs non-stress** in conversational speech using acoustic features.

This repository contains **v1**, a fully reproducible **MFCC + CNN baseline**, designed as a foundation for future **temporal stress escalation detection** systems.

---

## 1. Problem Definition

**Task:** Binary stress detection from speech  
**Inference Type:** Offline (no real-time constraints)

### Input

- 4-second mono audio windows
- Sampling rate: 16 kHz

### Output

- Stress probability → binary label
  - `0` = Non-Stress
  - `1` = Stress

### Applications

- Call-center monitoring
- Mental health screening
- Conversational risk detection systems

---

## 2. Dataset

### Primary Dataset

- **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)

### Label Mapping

Utterance-level emotion labels are mapped to stress labels as follows:

| Emotion                   | Stress Label   |
| ------------------------- | -------------- |
| neutral, calm, happy, sad | 0 (Non-Stress) |
| angry, fearful, disgust   | 1 (Stress)     |
| surprised                 | Excluded       |

---

## 3. Preprocessing Pipeline (v1)

The preprocessing pipeline is **speaker-aware** and **fully reproducible**.

### Steps

1. Load raw RAVDESS audio
2. Trim silence
3. Amplitude normalization
4. Windowing:
   - Window size: **4 seconds**
   - Hop size: **4 seconds** (no overlap)
5. Speaker-disjoint split:
   - Train / Validation / Test

### Output Artifacts

- Windowed audio files:`data/processed/audio/{train,val,test}/`
- Window-level metadata:`data/processed/windows.csv`

Each window record contains:

- `file_path`
- `speaker_id`
- `label`
- `split`

> **Important Assumption:**  
> Window-level labels are inherited from utterance-level annotations, which may introduce **label noise** when stress is not uniformly present across the utterance.

---

## 4. Feature Extraction

For each 4-second window:

- MFCCs: 13 coefficients
- Delta
- Delta-Delta

Final input tensor shape:

```
(3, 13, 128)
```

All feature extraction is performed on-the-fly during training and inference.

---

## 5. Model (v1)

### Architecture

- CNN-only classifier
- No temporal aggregation
- No recurrence or attention

This model serves as a strong acoustic baseline, not a final system.

---

## 6. Training Setup

- Loss: Binary Cross Entropy (BCE)
- Optimizer: Adam
- Batch size: 16
- Epochs: 5
- Best model checkpoint saved as:

```
checkpoints/cnn_best.pt
```

---

## 7. Evaluation Results (Test Set)

| **Metric**.       | **Value** |
| ----------------- | --------- |
| Accuracy          | ~77%      |
| Non-Stress Recall | ~93%      |
| Stress Recall     | ~58%      |
| Macro F1          | ~0.76     |

Artifacts saved in:

```
outputs/
├── classification_report.txt
├── confusion_matrix.png
├── test_metrics.json
```

### Error Characteristics

- Strong performance on Non-Stress
- Missed detections for subtle stress
- Errors likely due to:
  - window-level label noise
  - lack of temporal context

---

## 8. Baselines

- Majority-class baseline `~ 54% `
- Simple Classical Model Baseline `~ 60% `
- MFCC + CNN (this implementation) `~ 77% `

More advanced baselines (e.g., wav2vec) are intentionally deferred to later versions.

---

## 9. Limitations (v1)

- No temporal modeling
- Stress labels propagated from utterance to window
- Single dataset (RAVDESS only)
- Offline inference only

These limitations are **intentional** and define the scope of **v1**.

---

## 10. Roadmap

### v2 (Planned)

- CNN-LSTM / temporal aggregation
- Stress escalation detection across windows
- Cross-dataset evaluation (CREMA-D)
- Robustness to background noise

---

## 11. Project Status

**v1 complete and frozen**

- Preprocessing ✔
- Dataset pipeline ✔
- CNN training ✔
- Evaluation ✔

**Tag:**

```
v1.0-mfcc-cnn
```

---

# Preprocessing & Dataset Creation (v1)

> This section documents the exact commands used to construct the v1 dataset.
> These are one-time runners and are not part of runtime training code.

### 1. Build Metadata (ONE-TIME)

```
python - <<EOF
from src.data.make_metadata import build_metadata

df = build_metadata("data/raw/ravdess")
df.to_csv("data/processed/metadata.csv", index=False)

print(df.head())
print(df['label'].value_counts())
print("Speakers:", df['speaker_id'].nunique())
EOF
```

> This defines the dataset version. Run only once.

---

### 2. Generate Speaker Splits (ONE-TIME)

```
python - <<EOF
from src.data.make_splits import make_speaker_splits

splits = make_speaker_splits(
    metadata_path="data/processed/metadata.csv",
    output_dir="data/splits",
    seed=42
)

print({k: len(v) for k, v in splits.items()})
print("Train speakers:", splits["train"])
print("Val speakers:", splits["val"])
print("Test speakers:", splits["test"])
EOF

```

> ⚠️ Do not re-run with different seeds later. Splits are now part of the dataset definition.

---

### 3. Verify No Speaker Leakage (MANDATORY)

```
python - <<EOF
import pandas as pd

df = pd.read_csv("data/processed/metadata.csv")

def load_split(name):
    with open(f"data/splits/{name}_speakers.txt") as f:
        return set(f.read().splitlines())

train = load_split("train")
val = load_split("val")
test = load_split("test")

assert train.isdisjoint(val)
assert train.isdisjoint(test)
assert val.isdisjoint(test)

print("✅ No speaker leakage across splits.")
EOF
```

> This sanity check prevents speaker overlap between splits.

---

### 4. Preprocessing & Windowing (ONE-TIME)

```
python - <<EOF
from src.data.preprocess import preprocess_and_window

df = preprocess_and_window(
    metadata_path="data/processed/metadata.csv",
    splits_dir="data/splits",
    output_root="data/processed/audio",
    windows_csv_path="data/processed/windows.csv",
)

print("Preprocessing complete.")
print(df.head())
print(df["split"].value_counts())
print(df["label"].value_counts())
EOF
```

---

### 5. Inspect Processed Audio

> To Inspect one audio file

```
python - <<EOF
import os
import soundfile as sf

train_dir = "data/processed/audio/train"
print("Files in train:", len(os.listdir(train_dir)))

fname = os.listdir(train_dir)[0]
y, sr = sf.read(os.path.join(train_dir, fname))

print("Sample rate:", sr)
print("Duration (sec):", len(y) / sr)
EOF
```

**Expected Output:**

```
Files in train: 624
Sample rate: 16000
Duration (sec): 4.0
```

---

### 6. Count Windows per Split

```
for s in train val test; do
  echo "$s: $(ls data/processed/audio/$s | wc -l)"
done
```

> Ratios should roughly follow speaker split proportions.

---

# Sanity Checks

### 1. Dataset Sanity

```
python - <<EOF
from src.data.dataset import StressDataset

ds = StressDataset("data/processed/windows.csv")
X, y = ds[0]

print("X shape:", X.shape)   # (3, 13, 128)
print("y:", y)               # 0 or 1
print("dtype:", X.dtype)
EOF
```

**Expected Output:**

```
X shape: torch.Size([3, 13, 128])
y: tensor(0)
dtype: torch.float32
```

---

### 2. Model Sanity

```
python - <<EOF
import torch
from src.models.cnn import CNNStressClassifier

model = CNNStressClassifier()
x = torch.randn(4, 3, 13, 128)
y = model(x)

print("Output shape:", y.shape)
print("Output range:", y.min().item(), y.max().item())
EOF
```

**Expected Output:**

- Output shape: torch.Size([4])
- Values between 0 and 1

---

### 3. DataLoader Sanity

```
python - <<EOF
from src.data.dataloaders import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    "data/processed/windows.csv",
    batch_size=8,
    num_workers=0,   # Important on Mac / Windows
)

X, y = next(iter(train_loader))
print("Batch X shape:", X.shape)
print("Batch y shape:", y.shape)
print("Labels:", y[:5])
EOF
```

**Expected Output:**

- X: (8, 3, 13, 128)
- y: (8,)
- Values 0/1

> Confirms X shape (8,3,13,128) and label values 0/1.

---
