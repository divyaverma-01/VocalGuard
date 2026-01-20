import torch.nn as nn

class CNNEncoder(nn.Module):
    """
    Encodes a single MFCC window into a feature vector.
    Input:  (batch, 3, 13, 128)
    Output: (batch, 128)
    """

    def __init__(self, dropout=0.3):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2)   # (13x128) → (6x64)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2)   # (6x64) → (3x32)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.AdaptiveAvgPool2d((1, 1))  # → (1x1)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return x.view(x.size(0), -1)  # (B, 128)

