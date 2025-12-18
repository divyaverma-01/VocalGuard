# Voice Stress Detection (Applied AI)

Binary classification of short conversational speech into
stress vs non-stress using acoustic features.

## Problem Definition

- Input: 4-second audio windows, 16 kHz, mono
- Output: Stress probability + binary label
- Offline inference only

## Datasets

- RAVDESS (primary)
- CREMA-D (optional, later)

## Models

- MFCC + CNN-LSTM (from scratch)
- wav2vec embeddings + classifier (baseline)

## Evaluation

Precision, Recall, F1-score  
Speaker-independent splits to avoid leakage.

## Status

Step 3: Repository & environment setup completed.
