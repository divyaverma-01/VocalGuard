import pandas as pd
from torch.utils.data import DataLoader

from src.data.dataset import StressDataset


def create_dataloaders(
    windows_csv,
    batch_size=16,
    num_workers=2,
):
    df = pd.read_csv(windows_csv)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_ds = StressDataset(train_df)
    val_ds = StressDataset(val_df)
    test_ds = StressDataset(test_df)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
