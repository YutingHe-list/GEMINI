import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.STN import SpatialTransformer, Re_SpatialTransformer, AffineTransformer
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels//8, out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels//8, out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.double_conv(x)

class OneConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.one_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels//8, out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.one_conv(x)

class DoubleConvK1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(out_channels//8, out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.GroupNorm(out_channels//8, out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet_base(nn.Module):
    def __init__(self, n_channels, chs=(32, 64, 128, 256, 512, 256, 128, 64, 32)):
        super(UNet_base, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, chs[0])
        self.down1 = Down(chs[0], chs[1])
        self.down2 = Down(chs[1], chs[2])
        self.down3 = Down(chs[2], chs[3])
        self.down4 = Down(chs[3], chs[4])
        self.up1 = Up(chs[4] + chs[3], chs[5])
        self.up2 = Up(chs[5] + chs[2], chs[6])
        self.up3 = Up(chs[6] + chs[1], chs[7])
        self.up4 = Up(chs[7] + chs[0], chs[8])
        self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)

    def forward(self, x):
        Y = x.size()[2]
        X = x.size()[3]
        diffY = (16 - Y % 16) % 16
        diffX = (16 - X % 16) % 16
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return x[:, :, diffY//2: Y+diffY//2, diffX // 2:X + diffX // 2]

class UNet_shallow(nn.Module):
    def __init__(self, n_channels, chs=(32, 64, 64, 64, 32)):
        super(UNet_shallow, self).__init__()
        self.n_channels = n_channels
        self.chs = chs

        self.Conv1 = OneConv(n_channels, chs[0])
        self.Conv2 = OneConv(chs[0], chs[1])
        self.Conv3 = OneConv(chs[1], chs[2])
        self.Conv4 = OneConv(chs[2]+chs[1], chs[3])
        self.Conv5 = OneConv(chs[3]+chs[0], chs[4])

        self.max_pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)

    def forward(self, x):
        Y = x.size()[2]
        X = x.size()[3]
        diffY = (4 - Y % 4) % 4
        diffX = (4 - X % 4) % 4
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x1 = self.Conv1(x)
        x = self.max_pool(x1)
        x2 = self.Conv2(x)
        x = self.max_pool(x2)
        x3 = self.Conv3(x)
        x = self.up(x3)
        x4 = self.Conv4(torch.cat([x, x2], dim=1))
        x = self.up(x4)
        x5 = self.Conv5(torch.cat([x, x1], dim=1))

        return x5[:, :, diffY//2: Y+diffY//2, diffX // 2:X + diffX // 2]
