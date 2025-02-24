import os
import torch
from torch.utils.data import DataLoader

from models.GEMINI2d import GEMINI
from utils.STN import SpatialTransformer
from utils.Transform_2d import SpatialTransform2D, AppearanceTransform_appearance
from utils.dataloader_SSP_chestxray import DatasetFromFolder2D as DatasetFromFolder2D_train
from utils.losses2d import gradient_loss, MSE, ncc_loss_mask, cos_loss
from utils.utils import AverageMeter
import numpy as np

class Trainer(object):
    def __init__(self, start_epoch=0,
                 n_channels=1,
                 n_classes=1,
                 lr=1e-4,
                 epoches=1000,
                 iters=200,
                 batch_size=5,
                 unlabeled_dir='data/chestxray',
                 checkpoint_dir='weights',
                 result_dir='results',
                 model_name='GEMINI_SS_MIP_2d'
                 ):
        super(Trainer, self).__init__()
        # initialize parameters
        self.start_epoch = start_epoch
        self.n_classes = n_classes
        self.epoches = epoches
        self.iters = iters
        self.lr = lr
        self.unlabeled_dir = unlabeled_dir

        self.results_dir = result_dir
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name

        # data augmentation
        self.spatial_aug = SpatialTransform2D(do_rotation=True,
                                              angle_x=(-np.pi / 9, np.pi / 9),
                                              angle_y=(-np.pi / 9, np.pi / 9),
                                              do_scale=True,
                                              scale_x=(0.75, 1.25),
                                              scale_y=(0.75, 1.25),
                                              do_translate=False,
                                              do_shear=False,
                                              do_elastic_deform=False)
        self.style_aug_genesis = AppearanceTransform_appearance()

        # initialize networks
        self.Network = GEMINI(n_channels=n_channels, classes=n_classes)

        if torch.cuda.is_available():
            self.Network = self.Network.cuda()

        # initialize optimizer
        self.opt = torch.optim.Adam(self.Network.parameters(), lr=lr)

        self.stn = SpatialTransformer()

        # initialize dataloader
        train_dataset = DatasetFromFolder2D_train(self.unlabeled_dir)
        self.dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # define loss
        self.L_smo = gradient_loss
        self.L_sim = ncc_loss_mask
        self.L_cos = cos_loss

        self.L_mse = MSE

        # define loss log
        self.L_smo_log = AverageMeter(name='L_smo')
        self.L_gvs_log = AverageMeter(name='L_gvs')
        self.L_gss_log = AverageMeter(name='L_gss')

        self.L_mse_log = AverageMeter(name='L_mse')

    def train_iterator(self, unlabed_img1_aug_app, unlabed_img1, unlabed_img2):
        # Train GEMINI
        imgA = unlabed_img1
        imgB = unlabed_img2

        fA, fB, flow_AB, flow_BA = self.Network(imgA, imgB)

        # Generate mask
        mask = torch.full(imgA.shape, 1.)
        if torch.cuda.is_available():
            mask = mask.cuda()
        mask_AB = self.stn(mask, flow_AB, mode='nearest').detach()
        if torch.cuda.is_available():
            mask = mask.cuda()
        mask_BA = self.stn(mask, flow_BA, mode='nearest').detach()

        # smooth loss for continuity
        loss_smo = self.L_smo(flow_AB)
        loss_smo += self.L_smo(flow_BA)

        # similarity loss for correspondence
        # GVS
        warp_AB = self.stn(imgA, flow_AB)
        warp_BA = self.stn(imgB, flow_BA)
        loss_gvs = self.L_sim(warp_AB, imgB, mask_AB)
        loss_gvs += self.L_sim(warp_BA, imgA, mask_BA)

        # GSS
        warp_fAB = self.stn(fA, flow_AB.detach())
        loss_gss = 0.1 * self.L_cos(warp_fAB, fB, mask_AB)
        warp_fAB = self.stn(fA.detach(), flow_AB)
        loss_gss += self.L_cos(warp_fAB, fB.detach(), mask_AB)

        warp_fBA = self.stn(fB, flow_BA.detach())
        loss_gss += 0.1 * self.L_cos(warp_fBA, fA, mask_BA)
        warp_fBA = self.stn(fB.detach(), flow_BA)
        loss_gss += self.L_cos(warp_fBA, fA.detach(), mask_BA)

        loss_total = 0.4 * loss_smo + 0.8 * loss_gss + loss_gvs

        if not (loss_total == torch.inf or loss_total == -torch.inf or loss_total == torch.nan):
            self.L_smo_log.update(loss_smo.data, imgA.size(0))
            self.L_gvs_log.update(loss_gvs.data, imgA.size(0))
            self.L_gss_log.update(loss_gss.data, imgA.size(0))
            loss_total.backward()
            self.opt.step()
            self.Network.zero_grad()
            self.opt.zero_grad()

        # train restore
        imgA = unlabed_img1_aug_app
        imgA_org = unlabed_img1
        res_A = self.Network(imgA, None, gvs=False)

        loss_mse = self.L_mse(res_A, imgA_org)
        loss_total = 100 * loss_mse

        if not (loss_total == torch.inf or loss_total == -torch.inf or loss_total == torch.nan):
            self.L_mse_log.update(loss_mse.data, imgA.size(0))
            loss_total.backward()
            self.opt.step()
            self.Network.zero_grad()
            self.opt.zero_grad()

    def train_epoch(self, epoch):
        self.Network.train()
        for i in range(self.iters):
            unlabed_img1, unlabed_img2 = next(self.dataloader_train.__iter__())
            unlabed_img1_aug_app = []
            for batch in range(unlabed_img1.shape[0]):
                tmp = unlabed_img1.data.numpy()[batch].copy()
                tmp = self.style_aug_genesis.rand_aug(tmp)
                unlabed_img1_aug_app.append(torch.from_numpy(tmp[np.newaxis, :, :, :]))

            unlabed_img1_aug_app = torch.cat(unlabed_img1_aug_app, dim=0)

            if torch.cuda.is_available():
                unlabed_img1 = unlabed_img1.cuda()
                unlabed_img2 = unlabed_img2.cuda()
                unlabed_img1_aug_app = unlabed_img1_aug_app.cuda()

            flow_gt = []
            for j in range(unlabed_img1.shape[0]):
                flow_gt.append(self.spatial_aug.rand_coords(unlabed_img1.shape[2:]))
            flow_gt = torch.cat(flow_gt, dim=0)
            unlabed_img1 = self.stn(unlabed_img1, flow_gt)
            unlabed_img2 = self.stn(unlabed_img2, flow_gt)

            unlabed_img1_aug_app = self.stn(unlabed_img1_aug_app, flow_gt)

            self.train_iterator(unlabed_img1_aug_app, unlabed_img1, unlabed_img2)

            res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, self.epoches),
                             'Iter: [%d/%d]' % (i + 1, self.iters),
                             self.L_smo_log.__str__(),
                             self.L_gss_log.__str__(),
                             self.L_gvs_log.__str__(),
                             self.L_mse_log.__str__()])
            print(res)

    def checkpoint(self, epoch):
        torch.save(self.Network.state_dict(),
                   '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, self.model_name, epoch+self.start_epoch))

    def load(self):
        self.Network.load_state_dict(
            torch.load('{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, self.model_name, str(self.start_epoch))))

    def train(self):
        for epoch in range(self.epoches-self.start_epoch):
            self.L_smo_log.reset()
            self.L_gvs_log.reset()
            self.L_gss_log.reset()
            self.L_mse_log.reset()
            self.train_epoch(epoch+self.start_epoch)
            if epoch % 20 == 0:
                self.checkpoint(epoch)
        self.checkpoint(self.epoches-self.start_epoch)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    trainer = Trainer()
    # trainer.load()
    trainer.train()


