import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class FCNPre(nn.Module):
    def __init__(self, in_channels, x_size, y_size):
        super().__init__()
        self.in_channels = in_channels
        self.x_size = x_size
        self.y_size = y_size

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, padding=1)
        self.max1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.max2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.max3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.max4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.max5 = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        result = {}
        x = self.max1(F.relu(self.conv1(x)))
        x = self.max2(F.relu(self.conv2(x)))
        x = self.max3(F.relu(self.conv3(x)))
        result['3'] = x
        x = self.max4(F.relu(self.conv4(x)))
        result['4'] = x
        x = self.max5(F.relu(self.conv5(x)))
        result['5'] = x

        return result


class FCN32(nn.Module):
    def __init__(self, n_class, pretrained_module=None):
        super().__init__()
        self.pretrained_module = pretrained_module  ## 特征提取的部分
        self.n_class = n_class

        self.relu = nn.ReLU(inplace=True)

        """下面是恢复图像大小"""
        """转置卷积大小计算 = (initial_size-1)*stride + kernel_size - 2*padding + output_padding """
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.pretrained_module(x)
        x = x['5']

        x = self.bn1(self.relu(self.deconv1(x)))
        x = self.bn2(self.relu(self.deconv2(x)))
        x = self.bn3(self.relu(self.deconv3(x)))
        x = self.bn4(self.relu(self.deconv4(x)))
        x = self.bn5(self.relu(self.deconv5(x)))
        x = self.classifier(x)

        return x


class FCN16(nn.Module):
    def __init__(self, n_class, pretrained_module=None):
        super().__init__()
        self.pretrained_module = pretrained_module  ## 特征提取的部分
        self.n_class = n_class

        self.relu = nn.ReLU(inplace=True)

        """下面是恢复图像大小"""
        """转置卷积大小计算 = (initial_size-1)*stride + kernel_size - 2*padding + output_padding """
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1, padding=0)

    def forward(self, x):
        tmp = self.pretrained_module(x)

        x1 = tmp['5']
        x2 = tmp['4']

        x1 = torch.tile(x1,(2,2))

        x = x1 + x2
        x = self.bn1(self.relu(self.deconv1(x)))
        x = self.bn2(self.relu(self.deconv2(x)))
        x = self.bn3(self.relu(self.deconv3(x)))
        x = self.bn4(self.relu(self.deconv4(x)))
        x = self.classifier(x)

        return x


class FCN8(nn.Module):
    def __init__(self, n_class, pretrained_module=None):
        super().__init__()
        self.pretrained_module = pretrained_module  ## 特征提取的部分
        self.n_class = n_class

        self.relu = nn.ReLU(inplace=True)

        """下面是恢复图像大小"""
        """转置卷积大小计算 = (initial_size-1)*stride + kernel_size - 2*padding + output_padding """
        self.deconv1 = nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1, padding=0)

    def forward(self, x):
        tmp = self.pretrained_module(x)

        x1 = tmp['5']
        x2 = tmp['4']
        x3 = tmp['3']

        x1 = torch.tile(x1,(2,2))

        x1_2 = (x1 + x2)
        x1_2 = torch.tile(x1,(2,2))

        x = x1_2 + x3
        x = self.bn1(self.relu(self.deconv1(x)))
        x = self.bn2(self.relu(self.deconv2(x)))
        x = self.bn3(self.relu(self.deconv3(x)))
        x = self.classifier(x)

        return x


class FCN(nn.Module):
    def __init__(self, n_class, in_channels, x_size, y_size):
        super().__init__()
        self.n_class = n_class
        self.pretrained_module = FCNPre(in_channels, x_size, y_size)
        self.fcn8 = FCN8(n_class, self.pretrained_module)
        self.fcn16 = FCN16(n_class, self.pretrained_module)
        self.fcn32 = FCN32(n_class, self.pretrained_module)

    def forward(self, x):
        return self.fcn8(x)
