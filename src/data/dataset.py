import torch
from torch.utils.data import Dataset
import pandas as pd
import librosa

from src.features.mfcc import extract_mfcc


class StressDataset(Dataset):
    def __init__(self, windows_csv, sr=16000):
        self.df = pd.read_csv(windows_csv)
        self.sr = sr

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load audio
        y, sr = librosa.load(row["file_path"], sr=self.sr)

        # Amplitude normalization
        if y.max() > 0:
            y = y / abs(y).max()

        # MFCC tensor
        X = extract_mfcc(y, sr)

        # Label
        y_label = int(row["label"])

        return torch.tensor(X), torch.tensor(y_label, dtype=torch.long)
