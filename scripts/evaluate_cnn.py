import os
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import json

from src.models.cnn import CNNStressClassifier
from src.data.dataloaders import create_dataloaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate():
    os.makedirs("outputs", exist_ok=True)

    _, _, test_loader = create_dataloaders(
        windows_csv="data/processed/windows.csv",
        batch_size=16,
        num_workers=2,
    )

    model = CNNStressClassifier().to(DEVICE)
    model.load_state_dict(
        torch.load("checkpoints/cnn_best.pt", map_location=DEVICE)
    )
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            outputs = model(X)
            preds = (outputs > 0.5).long()  # 0 or 1
            # preds = torch.argmax(outputs, dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Non-Stress", "Stress"]
    )
    cm = confusion_matrix(y_true, y_pred)

    print("Accuracy:", acc)
    print(report)

    # Save metrics
    with open("outputs/test_metrics.json", "w") as f:
        json.dump({"accuracy": acc}, f, indent=4)

    with open("outputs/classification_report.txt", "w") as f:
        f.write(report)

    # Confusion matrix plot
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
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    evaluate()
