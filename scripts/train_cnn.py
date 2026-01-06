import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from src.models.cnn import CNNStressClassifier
from src.data.dataloaders import create_dataloaders


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train() # Training mode of the model
    total_loss = 0.0

    for X, y in tqdm(loader, desc="Train", leave=False):
        X = X.to(device)
        y = y.float().to(device)

        optimizer.zero_grad()
        preds = model(X)                 # Forward pass: Make a prediction
        loss = criterion(preds, y)       # Calculate: How wrong was the model?
        loss.backward()                  # Backward pass: Figure out which "weights" caused the error.
        optimizer.step()                 # Update: Change the weights slightly to be more accurate next time.

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval() # Evaluation mode of the model
    total_loss = 0.0

    with torch.no_grad():
        for X, y in tqdm(loader, desc="Val", leave=False):
            X = X.to(device)
            y = y.float().to(device)

            preds = model(X)
            loss = criterion(preds, y)

            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = create_dataloaders(
        "data/processed/windows.csv",
        batch_size=16,
        num_workers=2,   
    )

    model = CNNStressClassifier().to(device)

    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    epochs = 5

    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")


    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss = validate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/cnn_best.pt")
            print("✔ Saved new best model")



if __name__ == "__main__":
    main()
