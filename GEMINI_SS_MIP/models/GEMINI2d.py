import torch
import torch.nn as nn

from models.backbone_unet2d import UNet_base, UNet_shallow


class GEMINI(nn.Module):
    def __init__(self, n_channels=1, chan=(32, 64, 128, 256, 512, 256, 128, 64, 32), classes=1):
        super(GEMINI, self).__init__()
        self.backbone = UNet_base(n_channels=n_channels, chs=chan)
        sp_conv = UNet_shallow(chan[-1]*2)
        self.deformer = nn.Sequential(sp_conv,
                                     nn.Conv2d(sp_conv.chs[-1], 2, 3, padding=1))

        self.res_conv = nn.Sequential(nn.Conv2d(chan[-1], 16, 3, padding=1),
                                      nn.GroupNorm(16//4, 16),
                                      nn.LeakyReLU(0.2),
                                      nn.Conv2d(16, classes, 1))

        self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)

    def test_res(self, A):
        with torch.no_grad():
            fA = self.backbone(A)
            res_A = self.res_conv(fA)
        return res_A

    def test_reg(self, A, B):
        with torch.no_grad():
            fA = self.backbone(A)
            fB = self.backbone(B)

            flow_AB = self.deformer(torch.cat([fA, fB], dim=1))
        return flow_AB

    def forward(self, A, B, res=False):
        if res:
            fA = self.backbone(A)
            res_A = self.res_conv(fA)
            return res_A
        else:
            fA = self.backbone(A)
            fB = self.backbone(B)
            flow_AB = self.deformer(torch.cat([fA, fB], dim=1))
            flow_BA = self.deformer(torch.cat([fB, fA], dim=1))

            return fA, fB, flow_AB, flow_BA
