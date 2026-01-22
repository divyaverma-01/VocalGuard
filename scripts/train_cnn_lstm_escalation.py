import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support

from src.data.sequence_dataloaders import create_sequence_dataloaders
from src.models.cnn_lstm import CNNLSTMStressClassifier


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        y = y.to(device).float()

        optimizer.zero_grad()

        logits = model(x)  # (B,)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_targets = []

    for x, y in tqdm(loader, desc="Val", leave=False):
        x = x.to(device)
        y = y.to(device).float()

        logits = model(x)
        loss = criterion(logits, y)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

        running_loss += loss.item() * x.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets,
        all_preds,
        average="binary",
        zero_division=0
    )

    acc = (torch.tensor(all_preds) == torch.tensor(all_targets)).float().mean().item()

    return val_loss, acc, precision, recall, f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    csv_path = "data/processed/sequences.csv"

    # -------------------------
    # Escalation label stats
    # -------------------------
    df = pd.read_csv(csv_path)
    esc_counts = df[df["split"] == "train"]["escalation_label"].value_counts()

    neg = esc_counts[0]
    pos = esc_counts[1]

    pos_weight = torch.tensor([neg / pos], device=device)

    print(f"Escalation pos_weight: {pos_weight.item():.2f}")

    train_loader, val_loader, _ = create_sequence_dataloaders(
        csv_path,
        batch_size=8,
        num_workers=2,
        target_mode="escalation",
    )

    # -------------------------
    # Model
    # -------------------------
    model = CNNLSTMStressClassifier().to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=1e-4)

    os.makedirs("checkpoints", exist_ok=True)
    best_f1 = 0.0

    num_epochs = 20

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, acc, precision, recall, f1 = validate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {acc:.3f} | "
            f"P: {precision:.3f} | "
            f"R: {recall:.3f} | "
            f"F1: {f1:.3f}"
        )

        # ✅ Save best by F1
        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "f1": f1,
                },
                "checkpoints/cnn_lstm_escalation_best.pt",
            )
            print("✔ Saved new best escalation model")


if __name__ == "__main__":
    main()
