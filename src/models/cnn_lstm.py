import torch
import torch.nn as nn

from src.models.cnn_encoder import CNNEncoder


class CNNLSTMStressClassifier(nn.Module):
    """
    Temporal stress classifier using CNN + LSTM.

    Input:  (batch, K, 3, 13, 128)
    Output: (batch,) probability of stress
    """

    def __init__(self, embedding_dim=128, hidden_dim=128, num_layers=1):
        super().__init__()

        self.cnn = CNNEncoder()

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        B, K, C, H, W = x.shape

        # Merge batch & time for CNN
        x = x.view(B * K, C, H, W)
        x = self.cnn(x)              # (B*K, 128)

        # Restore sequence dimension
        x = x.view(B, K, -1)         # (B, K, 128)

        # LSTM over time
        _, (h_n, _) = self.lstm(x) # h_n → final hidden states

        # Last layer's final hidden state
        h_last = h_n[-1]             # (B, hidden_dim)

        out = torch.sigmoid(self.fc(h_last))

        return out.squeeze(1)
