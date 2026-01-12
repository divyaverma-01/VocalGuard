import json
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.dataset import StressDataset


class SequenceDataset(Dataset):
    """
    Temporal dataset for v2.
    Each item = sequence of K MFCC windows.
    """

    def __init__(self, sequences_csv: str, split: str):
        self.df = pd.read_csv(sequences_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.split = split

    def __len__(self):
        return len(self.df)

    def _load_window(self, wav_path: str, label: int):
        """
        Load one window using v1 MFCC pipeline.
        """
        temp_df = pd.DataFrame([{
            "file_path": wav_path,
            "label": label
        }])

        ds = StressDataset(temp_df)
        
        X, _ = ds[0]
        return X

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        window_files = json.loads(row["window_files"])
        labels = json.loads(row["labels"])
        sequence_label = int(row["sequence_label"])

        features = []

        for wav_path, lbl in zip(window_files, labels):
            X = self._load_window(wav_path, lbl)
            features.append(X)

        # Shape: (K, 3, 13, 128)
        X_seq = torch.stack(features, dim=0)

        y = torch.tensor(sequence_label, dtype=torch.long)

        return X_seq, y
