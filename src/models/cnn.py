import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNStressClassifier(nn.Module):
    """
    CNN-based classifier for stress detection from MFCC features.

    Input shape:
        (batch_size, 3, 13, 128)

    Output:
        (batch_size,) probability of stress
    """

    def __init__(self):
        super().__init__()

        #Convolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        #Max Pooling
        self.pool = nn.MaxPool2d(2)

        # After conv + pooling:
        # The Input: (3, 13, 128)
        # -> conv1 + pool -> (16, 6, 64)
        # -> conv2 + pool -> (32, 3, 32)
        # -> conv3 -> (64, 3, 32)

        #Dense layers (Input: 64 channels * 3 height * 32 width = 6144)
        self.fc1 = nn.Linear(64 * 3 * 32, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x.squeeze(1)
