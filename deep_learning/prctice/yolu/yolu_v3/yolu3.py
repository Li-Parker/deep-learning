import torch
import torch.nn as nn
from torch.nn import functional as F


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
            DarknetConv(in_channels, out_channels, kernel_size=1, padding=0),
            DarknetConv(out_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.sub_module(x)


class DBL5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sub_module = nn.Sequential(
            DarknetConv(in_channels, out_channels, kernel_size=1, padding=0),
            DarknetConv(out_channels, in_channels, kernel_size=3, padding=1),

            DarknetConv(in_channels, out_channels, kernel_size=1, padding=0),
            DarknetConv(out_channels, in_channels, kernel_size=3, padding=1),

            DarknetConv(in_channels, out_channels, kernel_size=1, padding=0)
        )

    def forward(self, x):
        return self.sub_module(x)


class DownSamplingLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sub_module = DarknetConv(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.sub_module(x)


class UpSamplingLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear')  ## 上采样


class ResBlockBody(nn.Module):
    def __init__(self, in_channels, out_channels, res_unit_num):
        super().__init__()
        self.sub_module1 = DownSamplingLayer(in_channels, out_channels)
        self.sub_module2 = nn.ModuleList([
            ResUnit(out_channels, out_channels*2)
            for _ in range(res_unit_num)
        ])

    def forward(self, x):
        x = self.sub_module1(x)
        for module_item in self.sub_module2:
            x = module_item(x)
        return x


class YOLU3(nn.Module):
    def __init__(self, in_channels=3, out_channels=255):
        super().__init__()
        self.inner_channels0 = 16
        self.inner_channels1 = 32
        self.inner_channels2 = 64
        self.inner_channels3 = 256
        self.inner_channels4 = 512
        self.inner_channels5 = 1024
        self.dark_net_52 = nn.Sequential(
            DarknetConv(in_channels, self.inner_channels0, kernel_size=3, padding=1),
            ResBlockBody(self.inner_channels0, self.inner_channels1, res_unit_num=1),
            ResBlockBody(self.inner_channels1, self.inner_channels2, res_unit_num=2),
            ResBlockBody(self.inner_channels2, self.inner_channels3, res_unit_num=8)
        )
        self.dark_net_26 = ResBlockBody(self.inner_channels3, self.inner_channels4, res_unit_num=8)
        self.dark_net_13 = ResBlockBody(self.inner_channels4, self.inner_channels5, res_unit_num=4)

        self.dark_net_13_1 = DBL5(self.inner_channels5, self.inner_channels4)   ##   [13,13,inner_channels4]
        self.dark_net_13_2 = DarknetConv(self.inner_channels4, self.inner_channels3, kernel_size=3, padding=1)  ## [13,13,inner_channels3]
        self.dark_net_13_3 = nn.Conv2d(self.inner_channels3, out_channels, kernel_size=3, padding=1) ## [13,13,out_channels]

        self.dark_net_13_to_26 = nn.Sequential(
            DarknetConv(self.inner_channels4, self.inner_channels4,kernel_size=3, padding=1),
            UpSamplingLayer()
        )
        self.dark_net_26_1 = DBL5(self.inner_channels4, self.inner_channels3)
        self.dark_net_26_2 = DarknetConv(self.inner_channels3, self.inner_channels3, kernel_size=3, padding=1)
        self.dark_net_26_3 = nn.Conv2d(self.inner_channels3, out_channels, kernel_size=3, padding=1)

        self.dark_net_26_to_52 = nn.Sequential(
            DarknetConv(self.inner_channels3, self.inner_channels3, kernel_size=3, padding=1),
            UpSamplingLayer()
        )
        self.dark_net_52_1 = DBL5(self.inner_channels3, self.inner_channels4)
        self.dark_net_52_2 = DarknetConv(self.inner_channels4, self.inner_channels5, kernel_size=3, padding=1)
        self.dark_net_52_3 = nn.Conv2d(self.inner_channels5, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        tmp_darknet_52 = self.dark_net_52(x)
        tmp_darknet_26 = self.dark_net_26(tmp_darknet_52)
        tmp_darknet_13 = self.dark_net_13(tmp_darknet_26)

        tmp_darknet_13_1 = self.dark_net_13_1(tmp_darknet_13)
        tmp_darknet_13_2 = self.dark_net_13_2(tmp_darknet_13_1)
        result_13 = self.dark_net_13_3(tmp_darknet_13_2)

        tmp_darknet_13_to_26 = self.dark_net_13_to_26(tmp_darknet_13_1)
        tmp_darknet_26_1 = self.dark_net_26_1(tmp_darknet_13_to_26 + tmp_darknet_26)
        tmp_darknet_26_2 = self.dark_net_26_2(tmp_darknet_26_1)
        result_26 = self.dark_net_26_3(tmp_darknet_26_2)

        tmp_darknet_26_to_52 = self.dark_net_26_to_52(tmp_darknet_26_1)
        tmp_darknet_52_1 = self.dark_net_52_1(tmp_darknet_26_to_52 + tmp_darknet_52)
        tmp_darknet_52_2 = self.dark_net_52_2(tmp_darknet_52_1)
        result_52 = self.dark_net_52_3(tmp_darknet_52_2)

        return result_13, result_26, result_52

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module_yolu3 = YOLU3().to(device)
    data_item = torch.rand(1, 3, 416, 416).to(device)
    result1, result2, result3 = module_yolu3(data_item)
    print(result1.shape)
    print(result2.shape)
    print(result3.shape)
