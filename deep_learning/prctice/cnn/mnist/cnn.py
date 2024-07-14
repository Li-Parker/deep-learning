import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import torchvision


class CNN(nn.Module):
    def __init__(self, x, y, channel):
        super().__init__()
        self.linearSize = int((x / 4) * (y / 4) * 12)
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=6, kernel_size=3, padding=1)
        self.maxPool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding=1)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2)
        self.fac1 = nn.Linear(self.linearSize, 256)
        self.fac2 = nn.Linear(256, 10)
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(12)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.maxPool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxPool2(x)

        x = self.fac1(x.view(-1, self.linearSize))
        x = F.relu(x)
        x = self.fac2(x)
        return x


if __name__ == '__main__':
    batch_size = 512
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    """加载数据"""
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST('mnist_data', train=False, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('mnist_data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    """定义模型和优化器"""
    model = CNN(28, 28, 1).to(device)
    opt = optim.Adam(params=model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    """训练"""
    epochs = 10
    train_loss = []
    model.train(True)
    for epoch in range(epochs):
        for _, (x, y) in enumerate(train_loader):
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            opt.step()
            train_loss.append(loss)

    """测试"""
    model.train(False)
    total_correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            total_correct += pred.eq(y).sum().item()

    accuracy = total_correct / len(test_loader.dataset)
    print(f'Test Accuracy: {accuracy:.4f}')
