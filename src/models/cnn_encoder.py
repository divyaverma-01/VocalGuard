import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    """
    Encodes a single MFCC window into a feature vector.
    Input:  (batch, 3, 13, 128)
    Output: (batch, 128)
    """

    def __init__(self):
        super().__init__()

        # Convolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Max Pooling
        self.pool = nn.MaxPool2d(2)

        # Dense layer
        self.fc = nn.Linear(64 * 3 * 32, 128)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x  # (batch, 128)
