import torch.nn as nn
import torch.nn.functional as F
from kernels import *


class LinearKernelCNN(nn.Module):

    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=7),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.maxpool = nn.AdaptiveMaxPool2d(output_size=1)

        self.fc = nn.Sequential(
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=n_classes)
        )

    def forward(self, image):
        x = self.features(image)
        x = self.maxpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        if not self.training:
            x = F.softmax(x, dim=1)
        return x


class MLPKernelCNN(nn.Module):

    def __init__(self, in_channels, n_classes, n_hidden=0):
        super().__init__()

        self.features = nn.Sequential(
            MLPKernel(in_channels=in_channels, out_channels=16, kernel_size=7, n_hidden=n_hidden),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            MLPKernel(in_channels=16, out_channels=32, kernel_size=5, n_hidden=n_hidden),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.maxpool = nn.AdaptiveMaxPool2d(output_size=1)

        self.fc = nn.Sequential(
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=n_classes)
        )

    def forward(self, image):
        x = self.features(image)
        x = self.maxpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        if not self.training:
            x = F.softmax(x, dim=1)
        return x
