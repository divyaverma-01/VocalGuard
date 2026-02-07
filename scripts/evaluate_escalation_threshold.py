import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc,
)

from src.data.sequence_dataloaders import create_sequence_dataloaders
from src.models.cnn_lstm import CNNLSTMStressClassifier


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    os.makedirs("outputs/v2.3.1", exist_ok=True)

    # ---------------------------
    # Load validation data
    # ---------------------------
    _, val_loader, _ = create_sequence_dataloaders(
        "data/processed/sequences.csv",
        batch_size=8,
        num_workers=2,
        target_mode="escalation",
    )

    # ---------------------------
    # Load best trained model
    # ---------------------------
    model = CNNLSTMStressClassifier().to(device)

    checkpoint = torch.load(
        "checkpoints/cnn_lstm_escalation_best_imbalance_handling_v2.3.pt", # cnn_lstm_escalation_best.pt
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("✔ Loaded best trained model")

    # ---------------------------
    # Collect probabilities
    # ---------------------------
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Save raw predictions
    np.save("outputs/v2.3.1/val_probs.npy", all_probs)
    np.save("outputs/v2.3.1/val_labels.npy", all_labels)

    # ---------------------------
    # Threshold Sweep
    # ---------------------------
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    print("\nThreshold | Precision | Recall | F1")
    print("-----------------------------------")

    results = []

    for t in thresholds:
        preds = (all_probs >= t).astype(int)

        p = precision_score(all_labels, preds, zero_division=0)
        r = recall_score(all_labels, preds, zero_division=0)
        f1 = f1_score(all_labels, preds, zero_division=0)

        print(f"{t:>9} | {p:>9.3f} | {r:>6.3f} | {f1:>5.3f}")

        results.append({
            "threshold": t,
            "precision": p,
            "recall": r,
            "f1": f1
        })

    # Save sweep results
    pd.DataFrame(results).to_csv(
        "outputs/v2.3.1/threshold_sweep.csv",
        index=False
    )

    # ---------------------------
    # Precision-Recall Curve
    # ---------------------------
    precision, recall, pr_thresholds = precision_recall_curve(
        all_labels,
        all_probs
    )

    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (Escalation Detection)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/v2.3.1/pr_curve.png")
    plt.close()

    print(f"\nPR-AUC: {pr_auc:.4f}")
    print("✔ Saved PR curve to outputs/v2.3.1/pr_curve.png")

    # ---------------------------
    # Choose F1-optimal threshold
    # ---------------------------
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = pr_thresholds[best_idx]

    print(f"\nBest F1 Threshold (dense search): {best_threshold:.3f}")


if __name__ == "__main__":
    main()