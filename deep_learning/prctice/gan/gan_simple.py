import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, width_dim=28, height_dim=28):
        super().__init__()
        self.inner_channels = 8
        self.sub_module1 = nn.Sequential(
            nn.Conv2d(in_channels, self.inner_channels, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
        )
        self.sub_module2 = nn.Sequential(
            nn.Linear((width_dim // 2) * (height_dim // 2) * self.inner_channels, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.sub_module1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.sub_module2(x)
        return x


class Generator(nn.Module):
    def __init__(self, z_dim, out_channels=1, width_dim=28, height_dim=28):
        super().__init__()

        self.out_channels = out_channels
        self.width_dim = width_dim
        self.height_dim = height_dim

        self.gen = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, width_dim * height_dim * out_channels),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.gen(x)
        return x.reshape(x.shape[0], self.out_channels, self.width_dim, self.height_dim)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ## 创建文件夹
    os.makedirs("./images/gan/", exist_ok=True)  ## 记录训练过程的图片效果
    os.makedirs("./save/gan/", exist_ok=True)  ## 训练完成时模型保存的位置
    os.makedirs("./datasets/mnist", exist_ok=True)  ## 下载数据集存放的位置
    ## mnist数据集下载
    mnist = datasets.MNIST(
        root='./datasets/', train=True, download=True, transform=transforms.Compose(
            [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    )
    ## 配置数据到加载器
    dataloader = DataLoader(
        mnist,
        batch_size=64,
        shuffle=True,
    )

    generator = Generator(z_dim=100, width_dim=28, height_dim=28).to(device)
    discriminator = Discriminator(width_dim=28, height_dim=28).to(device)
    ## 定义loss的度量方式  （二分类的交叉熵）
    criterion = torch.nn.BCELoss().to(device)
    ## 其次定义 优化函数,优化函数的学习率为0.0003
    ## betas:用于计算梯度以及梯度平方的运行平均值的系数
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-3, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999))

    for epoch in range(10):
        for i, (imgs, _) in enumerate(dataloader):
            real_img = Variable(imgs).to(device)
            real_label = Variable(torch.ones(imgs.size(0), 1)).to(device)
            fake_label = Variable(torch.zeros(imgs.size(0), 1)).to(device)

            real_out = discriminator(real_img)
            loss_real_D = criterion(real_out, real_label)
            real_scores = real_out

            z = Variable(torch.randn(imgs.size(0), 100)).to(device)
            fake_img = generator(z)
            fake_out = discriminator(fake_img)
            loss_fake_D = criterion(fake_out, fake_label)
            fake_scores = fake_out

            loss_D = loss_real_D + loss_fake_D
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()


            z = Variable(torch.randn(imgs.size(0), 100)).to(device)
            fake_img = generator(z)
            output = discriminator(fake_img)
            loss_G = criterion(output, real_label)
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            if (i + 1) % 100 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [D real: %f] [D fake: %f]"
                    % (epoch, 10, i, len(dataloader), loss_D.item(), loss_G.item(), real_scores.data.mean(),
                       fake_scores.data.mean())
                )
            ## 保存训练过程中的图像
            batches_done = epoch * len(dataloader) + i
            if batches_done % 500 == 0:
                save_image(fake_img.data[:25], "./images/gan/%d.png" % batches_done, nrow=5, normalize=True)