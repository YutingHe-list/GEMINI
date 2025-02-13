from os.path import join
from os import listdir
from scipy.io import loadmat
import SimpleITK as sitk
import pandas as pd
from torch.utils import data
import numpy as np
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii.gz"])

def imgnorm(N_I, index1=0.015, index2=0.015):
    I_sort = np.sort(N_I.flatten())
    I_min = I_sort[int(index1 * len(I_sort))]
    I_max = I_sort[-int(index2 * len(I_sort))]

    N_I = 1.0 * (N_I - I_min) / (I_max - I_min+1e-6)
    N_I[N_I > 1.0] = 1.0
    N_I[N_I < 0.0] = 0.0
    N_I2 = N_I.astype(np.float32)
    return N_I2

def limit(image):
    max = np.where(image < 0)
    image[max] = 0
    return image

def Nor(data):
    data = np.asarray(data)
    min = np.min(data)
    max = np.max(data)
    data = (data - min) / (max - min)
    return data

class DatasetFromFolder3D(data.Dataset):
    def __init__(self, unlabeled_file_dir, labeled_file_dir):
        super(DatasetFromFolder3D, self).__init__()
        self.unlabeled_filenames = [x for x in listdir(join(unlabeled_file_dir, 'image')) if is_image_file(x)]
        self.unlabeled_file_dir = unlabeled_file_dir

        self.labeled_filenames = [x for x in listdir(join(labeled_file_dir, 'image')) if is_image_file(x)]
        self.labeled_file_dir = labeled_file_dir

        self.list = [0,2,3,4,7,8,10,11,12,13,14,15,16,17,18,24,28,41,42,43,46,47,49,50,51,52,53,54,60]

    def __getitem__(self, index):
        random_index = np.random.randint(low=0, high=len(self.labeled_filenames))
        labed_img = sitk.ReadImage(join(self.labeled_file_dir, 'image', self.labeled_filenames[random_index]))
        labed_img = sitk.GetArrayFromImage(labed_img)
        labed_img = Nor(limit(labed_img))
        labed_img = labed_img.astype(np.float32)
        labed_lab = sitk.ReadImage(join(self.labeled_file_dir, 'label', self.labeled_filenames[random_index]))
        labed_lab = sitk.GetArrayFromImage(labed_lab)
        label_mask = np.where(labed_lab > 0, 1, 0).astype(np.float32)
        labed_img = labed_img * label_mask
        labed_img = labed_img[np.newaxis, :, :, :]
        labed_lab = [np.where(labed_lab == i, 1, 0)[np.newaxis, :, :, :] for i in self.list]
        labed_lab = np.concatenate(labed_lab, axis=0)
        labed_lab = labed_lab.astype(np.float32)

        random_index = np.random.randint(low=0, high=len(self.unlabeled_filenames))
        unlabed_img1 = sitk.ReadImage(join(self.unlabeled_file_dir, 'image', self.unlabeled_filenames[random_index]))
        unlabed_img1 = sitk.GetArrayFromImage(unlabed_img1)
        unlabed_img1 = Nor(limit(unlabed_img1))
        unlabed_img1 = unlabed_img1.astype(np.float32)
        mask = sitk.ReadImage(join(self.unlabeled_file_dir, 'label', self.unlabeled_filenames[random_index]))
        mask = sitk.GetArrayFromImage(mask)
        mask = np.where(mask > 0, 1, 0).astype(np.float32)
        unlabed_img1 = unlabed_img1 * mask
        unlabed_img1 = unlabed_img1[np.newaxis, :, :, :]

        random_index = np.random.randint(low=0, high=len(self.unlabeled_filenames))
        unlabed_img2 = sitk.ReadImage(join(self.unlabeled_file_dir, 'image', self.unlabeled_filenames[random_index]))
        unlabed_img2 = sitk.GetArrayFromImage(unlabed_img2)
        unlabed_img2 = Nor(limit(unlabed_img2))
        unlabed_img2 = unlabed_img2.astype(np.float32)
        mask = sitk.ReadImage(join(self.unlabeled_file_dir, 'label', self.unlabeled_filenames[random_index]))
        mask = sitk.GetArrayFromImage(mask)
        mask = np.where(mask > 0, 1, 0).astype(np.float32)
        unlabed_img2 = unlabed_img2 * mask
        unlabed_img2 = unlabed_img2[np.newaxis, :, :, :]

        return labed_img, labed_lab, unlabed_img1, unlabed_img2

    def to_categorical(self, y, num_classes=None):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((num_classes, n))
        categorical[y, np.arange(n)] = 1
        output_shape = (num_classes,) + input_shape
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def __len__(self):
        return len(self.unlabeled_filenames)+len(self.labeled_filenames)

