import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from src.data.sequence_dataloaders import create_sequence_dataloaders
from src.models.cnn_lstm import CNNLSTMStressClassifier


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = torch.sigmoid(logits)

        preds = (probs > 0.5).long()

        all_labels.extend(y.tolist())          # Actual/true labels
        all_preds.extend(preds.cpu().tolist()) # Binary predictions (0 or 1)
        all_probs.extend(probs.cpu().tolist()) # Predicted probabilities (after sigmoid) -> values in [0, 1]

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = create_sequence_dataloaders(
        "data/processed/sequences.csv",
        batch_size=8,
        num_workers=2,
    )

    model = CNNLSTMStressClassifier().to(device)

    checkpoint = torch.load(
        "checkpoints/cnn_lstm_best.pt",
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state"])

    y_true, y_pred, y_prob = evaluate(model, test_loader, device)

    os.makedirs("outputs", exist_ok=True)

    # Metrics
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Non-Stress", "Stress"],
        digits=4,
    )

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # Save outputs
    with open("outputs/classification_report.txt", "w") as f:
        f.write(report)

    with open("outputs/test_metrics.json", "w") as f:
        json.dump(
            {"accuracy": acc},
            f,
            indent=2,
        )

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Stress", "Stress"],
        yticklabels=["Non-Stress", "Stress"],
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("CNN-LSTM Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()


    print("=== CNN-LSTM Test Results ===")
    print(report)
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)


if __name__ == "__main__":
    main()
