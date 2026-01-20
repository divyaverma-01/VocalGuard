import torch.nn as nn
from src.models.cnn_encoder import CNNEncoder


class CNNLSTMStressClassifier(nn.Module):
    """
    Regularized temporal stress classifier using CNN + BiLSTM + temporal pooling.

    Input:  (batch, K, 3, 13, 128)
    Output: (batch,) stress probability
    """

    def __init__(self, embedding_dim=128, hidden_dim=64, num_layers=1, dropout=0.3):
        super().__init__()

        self.cnn = CNNEncoder()

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        # BiLSTM → hidden_dim * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        B, K, C, H, W = x.shape

        # CNN encoding
        # Merge batch & time for CNN
        x = x.view(B * K, C, H, W)
        x = self.cnn(x)              # (B*K, 128)

        # Restore sequence dimension
        x = x.view(B, K, -1)         # (B, K, 128)

        # BiLSTM
        lstm_out, _ = self.lstm(x)   # (B, K, 2*hidden_dim)

        # Temporal mean pooling
        pooled = lstm_out.mean(dim=1)  # (B, 2*hidden_dim)

        logits = self.fc(pooled)

        return logits.squeeze(1)
