import torch
from torch import nn


class MLPKernel(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, n_hidden=0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size)]
        for i in range(n_hidden):
            layers += [nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)
