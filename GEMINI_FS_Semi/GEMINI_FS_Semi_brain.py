import os
from os.path import join
import SimpleITK as sitk
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from models.GEMINI3d import GEMINI
from utils.STN import SpatialTransformer
from utils.augmentation import SpatialTransform
from utils.dataloader_brain_train import DatasetFromFolder3D as DatasetFromFolder3D_train
from utils.dataloader_brain_test_reg import DatasetFromFolder3D as DatasetFromFolder3D_test_reg
from utils.dataloader_brain_test_seg import DatasetFromFolder3D as DatasetFromFolder3D_test_seg
from utils.losses import gradient_loss, dice_loss, crossentropy, cos_loss, ncc_loss_mask
from utils.utils import AverageMeter, to_categorical, dice

class Trainer(object):
    def __init__(self,
                 start_epoch=0,
                 n_channels=1,
                 n_classes=29,
                 lr=1e-4,
                 epoches=200,
                 iters=200,
                 batch_size=1,
                 is_aug=True,
                 labeled_train_dir='/data/Brain/train/labeled',
                 unlabeled_train_dir='/data/Brain/train/unlabeled',
                 labeled_test_dir='/data/Brain/test/labeled',
                 checkpoint_dir='weights',
                 result_dir='results',
                 model_name='GEMINI_FS_Semi_brain'):
        super(Trainer, self).__init__()
        # initialize parameters
        self.start_epoch = start_epoch
        self.n_classes = n_classes
        self.epoches = epoches
        self.iters = iters
        self.lr = lr
        self.is_aug = is_aug

        self.labeled_train_dir = labeled_train_dir
        self.unlabeled_train_dir = unlabeled_train_dir
        self.labeled_test_dir = labeled_test_dir

        self.results_dir = result_dir
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name

        # tools
        self.stn = SpatialTransformer() # Spatial Transformer
        self.softmax = nn.Softmax(dim=1)

        # data augmentation
        self.spatial_aug = SpatialTransform(do_rotation=True,
                                            angle_x=(-np.pi / 9, np.pi / 9),
                                            angle_y=(-np.pi / 9, np.pi / 9),
                                            angle_z=(-np.pi / 9, np.pi / 9),
                                            do_scale=True,
                                            scale=(0.75, 1.25))

        # initialize networks
        self.Network = GEMINI(n_channels=n_channels, classes=n_classes)

        if torch.cuda.is_available():
            self.Network = self.Network.cuda()

        # initialize optimizer
        self.opt = torch.optim.Adam(self.Network.parameters(), lr=lr)

        # initialize dataloader
        train_dataset = DatasetFromFolder3D_train(self.unlabeled_train_dir, self.labeled_train_dir)
        self.dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset_seg = DatasetFromFolder3D_test_seg(self.labeled_test_dir)
        self.dataloader_test_seg = DataLoader(test_dataset_seg, batch_size=1, shuffle=False)
        test_dataset_reg = DatasetFromFolder3D_test_reg(self.labeled_test_dir)
        self.dataloader_test_reg = DataLoader(test_dataset_reg, batch_size=1, shuffle=False)

        # define loss
        self.L_smo = gradient_loss
        self.L_sim = ncc_loss_mask
        self.L_cos = cos_loss

        self.L_dice = dice_loss
        self.L_ce = crossentropy

        # define loss log
        self.L_smo_log = AverageMeter(name='L_smo')
        self.L_gvs_log = AverageMeter(name='L_gvs')
        self.L_gss_log = AverageMeter(name='L_gss')

        self.L_dice_log = AverageMeter(name='L_dice')
        self.L_ce_log = AverageMeter(name='L_ce')

    def train_iterator(self, labed_img, labed_lab, unlabed_img1, unlabed_img2):
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
        self.L_smo_log.update(loss_smo.data, imgA.size(0))

        # similarity loss for correspondence
        # GVS
        warp_AB = self.stn(imgA, flow_AB)
        warp_BA = self.stn(imgB, flow_BA)
        loss_gvs = self.L_sim(warp_AB, imgB, mask_AB)
        loss_gvs += self.L_sim(warp_BA, imgA, mask_BA)
        self.L_gvs_log.update(loss_gvs.data, imgA.size(0))

        # GSS
        warp_fAB = self.stn(fA, flow_AB.detach())
        warp_fBA = self.stn(fB, flow_BA.detach())

        loss_gss = self.L_cos(warp_fAB, fB, mask_AB)
        loss_gss += self.L_cos(warp_fBA, fA, mask_AB)
        self.L_gss_log.update(loss_gss.data, imgA.size(0))

        loss_total = 0.8*loss_smo + loss_gss + 0.8*loss_gvs

        loss_total.backward()
        self.opt.step()
        self.Network.zero_grad()
        self.opt.zero_grad()

        # Train seg
        imgA = labed_img
        labA = labed_lab
        res_A = self.Network(imgA, None, res=True)

        loss_ce = self.L_ce(self.softmax(res_A), labA)
        loss_dice = self.L_dice(self.softmax(res_A), labA)
        self.L_ce_log.update(loss_ce.data, imgA.size(0))
        self.L_dice_log.update(loss_dice.data, imgA.size(0))
        loss_total = 100 * (loss_ce + loss_dice)

        loss_total.backward()
        self.opt.step()
        self.Network.zero_grad()
        self.opt.zero_grad()

    def train_epoch(self, epoch):
        self.Network.train()
        for i in range(self.iters):
            labed_img, labed_lab, unlabed_img1, unlabed_img2 = next(self.dataloader_train.__iter__())

            if torch.cuda.is_available():
                labed_img = labed_img.cuda()
                labed_lab = labed_lab.cuda()
                unlabed_img1 = unlabed_img1.cuda()
                unlabed_img2 = unlabed_img2.cuda()

            if self.is_aug:
                code_spa = self.spatial_aug.rand_coords(labed_img.shape[2:])
                labed_img = self.spatial_aug.augment_spatial(labed_img, code_spa)
                labed_lab = self.spatial_aug.augment_spatial(labed_lab, code_spa, mode='nearest')
                unlabed_img1 = self.spatial_aug.augment_spatial(unlabed_img1, code_spa)
                unlabed_img2 = self.spatial_aug.augment_spatial(unlabed_img2, code_spa)

            self.train_iterator(labed_img, labed_lab, unlabed_img1, unlabed_img2)
            log = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, self.epoches),
                             'Iter: [%d/%d]' % (i + 1, self.iters),
                             self.L_smo_log.__str__(),
                             self.L_gss_log.__str__(),
                             self.L_gvs_log.__str__(),
                             self.L_ce_log.__str__(),
                             self.L_dice_log.__str__()])
            print(log)


    def test_iterator_seg(self, mi):
        with torch.no_grad():
            # Seg
            s_m = self.Network.test_res(mi)
        return s_m

    def test_iterator_reg(self, mi, fi, ml):
        with torch.no_grad():
            # Reg
            flow = self.Network.test_reg(mi, fi)
            w_m_to_f = self.stn(mi, flow)
            w_label_m_to_f = self.stn(ml, flow, mode='nearest')

        return w_m_to_f, w_label_m_to_f, flow

    def test(self):
        self.Network.eval()
        for i, (mi, ml, name) in enumerate(self.dataloader_test_seg):
            name = name[0]
            if torch.cuda.is_available():
                mi = mi.cuda()
            s_m = self.test_iterator_seg(mi)
            s_m = np.argmax(s_m.data.cpu().numpy()[0], axis=0)
            s_m = s_m.astype(np.int8)
            if not os.path.exists(join(self.results_dir, self.model_name, 'seg')):
                os.makedirs(join(self.results_dir, self.model_name, 'seg'))

            s_m = sitk.GetImageFromArray(s_m)
            sitk.WriteImage(s_m, join(self.results_dir, self.model_name, 'seg', name[:-4]+'.nii'))
            print(name[:-4]+'.nii')

        for i, (mi, ml, fi, fl, name1, name2) in enumerate(self.dataloader_test_reg):
            name1 = name1[0]
            name2 = name2[0]
            if name1 is not name2:
                if torch.cuda.is_available():
                    mi = mi.cuda()
                    fi = fi.cuda()
                    ml = ml.cuda()
                    fl = fl.cuda()

                w_m_to_f, w_label_m_to_f, flow = self.test_iterator_reg(mi, fi, ml)

                flow = flow.data.cpu().numpy()[0]
                w_m_to_f = w_m_to_f.data.cpu().numpy()[0, 0]
                w_label_m_to_f = np.argmax(w_label_m_to_f.data.cpu().numpy()[0], axis=0)

                flow = flow.astype(np.float32)
                w_m_to_f = w_m_to_f.astype(np.float32)
                w_label_m_to_f = w_label_m_to_f.astype(np.int8)

                if not os.path.exists(join(self.results_dir, self.model_name, 'flow')):
                    os.makedirs(join(self.results_dir, self.model_name, 'flow'))
                if not os.path.exists(join(self.results_dir, self.model_name, 'w_m_to_f')):
                    os.makedirs(join(self.results_dir, self.model_name, 'w_m_to_f'))
                if not os.path.exists(join(self.results_dir, self.model_name, 'w_label_m_to_f')):
                    os.makedirs(join(self.results_dir, self.model_name, 'w_label_m_to_f'))

                w_m_to_f = sitk.GetImageFromArray(w_m_to_f)
                sitk.WriteImage(w_m_to_f, join(self.results_dir, self.model_name, 'w_m_to_f', name2[:-4]+'_'+name1[:-4]+'.nii'))
                w_label_m_to_f = sitk.GetImageFromArray(w_label_m_to_f)
                sitk.WriteImage(w_label_m_to_f, join(self.results_dir, self.model_name, 'w_label_m_to_f', name2[:-4]+'_'+name1[:-4]+'.nii'))
                flow = sitk.GetImageFromArray(flow)
                sitk.WriteImage(flow, join(self.results_dir, self.model_name, 'flow', name2[:-4]+'_'+name1[:-4]+'.nii'))
                print(name2[:-4]+'_'+name1[:-4]+'.nii')


    def checkpoint(self, epoch, start_epoch):
        torch.save(self.Network.state_dict(),
                   '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, self.model_name, epoch+start_epoch),
                   _use_new_zipfile_serialization=False)

    def load(self):
        self.Network.load_state_dict(
            torch.load('{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, self.model_name, str(self.start_epoch))))

    def train(self):
        for epoch in range(self.epoches-self.start_epoch):
            self.L_smo_log.reset()
            self.L_gvs_log.reset()
            self.L_gss_log.reset()
            self.L_dice_log.reset()
            self.L_ce_log.reset()
            self.train_epoch(epoch+self.start_epoch)
            if epoch % 20 == 0:
                self.checkpoint(epoch, self.start_epoch)
        self.checkpoint(self.epoches-self.start_epoch, self.start_epoch)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    trainer = Trainer()
    # trainer.load()
    trainer.train()
    trainer.test()



