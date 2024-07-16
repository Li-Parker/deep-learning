import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, width_dim=48, height_dim=48):
        super().__init__()
        self.inner_channels = 8
        self.sub_module = nn.Sequential(
            nn.Conv2d(in_channels, self.inner_channels, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Linear((width_dim//2)*(height_dim//2)*self.inner_channels, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.sub_module(x)


class Generator(nn.Module):
    def __init__(self, z_dim, out_channels=1, width_dim=48, height_dim=48):
        super().__init__()

        self.out_channels = out_channels
        self.width_dim = width_dim
        self.height_dim = height_dim

        self.gen = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, width_dim*height_dim*out_channels),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.gen(x)
        return x.reshape(self.out_channels, self.width_dim, self.height_dim)