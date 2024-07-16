import torch
import torch.nn as nn

class FCN32(nn.Module):
    def __init__(self, n_class, pretrained_module=None):
        super().__init__()
        self.pretrained_module = pretrained_module ## 特征提取的部分
        self.n_class = n_class

        self.relu = nn.ReLU(inplace=True)

        """下面是恢复图像大小"""
        """转置卷积大小计算 = (initial_size-1)*stride + kernel_size - 2*padding + output_padding """
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512,256,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,padding=1,output_padding=1)
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
