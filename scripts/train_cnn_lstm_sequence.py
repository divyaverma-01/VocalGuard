import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from src.data.sequence_dataloaders import create_sequence_dataloaders
from src.models.cnn_lstm import CNNLSTMStressClassifier


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        y = y.to(device).float()

        optimizer.zero_grad()
        logits = model(x)          # raw logits
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="Val", leave=False):
        x = x.to(device)
        y = y.to(device).float()

        logits = model(x)
        loss = criterion(logits, y)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

        running_loss += loss.item() * x.size(0)
        correct += (preds.cpu() == y.cpu()).sum().item()
        total += y.size(0)

    return running_loss / len(loader.dataset), correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, _ = create_sequence_dataloaders(
        "data/processed/sequences.csv",
        batch_size=8,
        num_workers=2,
    )

    model = CNNLSTMStressClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")

    num_epochs = 20

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                "checkpoints/cnn_lstm_sequence_best.pt",
            )
            print("✔ Saved new best model")


if __name__ == "__main__":
    main()
