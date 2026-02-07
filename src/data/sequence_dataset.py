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

    def __init__(self, sequences_csv: str, split: str, target_mode="sequence"):
        self.df = pd.read_csv(sequences_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.escalation_labels = self.df["escalation_label"].tolist()

        self.split = split
        self.target_mode = target_mode

        assert self.target_mode in ["sequence", "escalation"], \
        f"Invalid target_mode: {self.target_mode}"


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
        
        if self.target_mode == "sequence":
            target = int(row["sequence_label"])
        else:
            target = int(row["escalation_label"])

        features = []

        for wav_path, lbl in zip(window_files, labels):
            X = self._load_window(wav_path, lbl)
            features.append(X)

        # Shape: (K, 3, 13, 128)
        X_seq = torch.stack(features, dim=0)

        y = torch.tensor(target, dtype=torch.long)

        return X_seq, y
