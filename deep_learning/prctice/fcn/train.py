import argparse
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from bag_data import BagDataset
from fcn import *

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10, help="训练的epoch个数")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--data_suffer", type=bool, default=True)
parser.add_argument("--n_class", type=int, default=3)
parser.add_argument("--in_channels", type=int, default=3)
parser.add_argument("--x_size", type=int, default=160)
parser.add_argument("--y_size", type=int, default=160)

opt = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    bag = BagDataset(transform)

    train_size = int(0.9 * len(bag))
    test_size = len(bag) - train_size
    train_dataset, test_dataset = random_split(bag, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)
    return train_dataloader, test_dataloader
if __name__ == '__main__':
    train_dataloader, test_dataloader = get_data()
    fcn_module = FCN(n_class=opt.n_class, in_channels=opt.in_channels, x_size=opt.x_size, y_size=opt.y_size).to(device)
    # optimizer = optim.Adam(fcn_module.parameters(), lr=1e-2)
    optimizer = optim.SGD(fcn_module.parameters(), lr=1e-2, momentum=0.7)
    criterion = nn.BCELoss().to(device)

    # fcn_module.train() ## Sets the module in training mode.
    for epoch in range(opt.epochs):
        for index, (bag, bagmsk) in enumerate(train_dataloader):
            bag = bag.to(device)
            bagmsk = bagmsk.to(device)
            optimizer.zero_grad()
            output = fcn_module(bag)
            output = torch.sigmoid(output)
            loss = criterion(output, bagmsk) ## BCELoss计算中含有log函数，要保证数值在0-1之间，所以要用sigmoid激活函数，再者，sigmoid函数十分适合二分类任务
            optimizer.step()
            print(loss.item())


