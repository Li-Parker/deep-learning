import torch
import torch.nn as nn
from torch.nn import functional as F


class YOLU3(nn.Module):
    def __init__(self):
        super().__init__()


class DarknetConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False):
        super().__init__()
        self.sub_module = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.sub_module(x)


class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sub_module = nn.Sequential(
            DarknetConv(in_channels, out_channels, kernel_size=3, padding=1),
            DarknetConv(out_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.sub_module(x)
