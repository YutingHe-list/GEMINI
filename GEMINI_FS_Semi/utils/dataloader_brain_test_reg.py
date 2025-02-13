from os.path import join
from os import listdir
import SimpleITK as sitk
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
    def __init__(self, file_dir):
        super(DatasetFromFolder3D, self).__init__()
        self.labeled_filenames = [x for x in listdir(join(file_dir, 'image')) if is_image_file(x)]
        self.file_dir = file_dir
        self.list = [0,2,3,4,7,8,10,11,12,13,14,15,16,17,18,24,28,41,42,43,46,47,49,50,51,52,53,54,60]

    def __getitem__(self, index):
        labed_img1 = sitk.ReadImage(join(self.file_dir, 'image', self.labeled_filenames[index // len(self.labeled_filenames)]))
        labed_img1 = sitk.GetArrayFromImage(labed_img1)
        labed_img1 = Nor(limit(labed_img1))
        labed_img1 = labed_img1.astype(np.float32)
        labed_lab1 = sitk.ReadImage(join(self.file_dir, 'label', self.labeled_filenames[index // len(self.labeled_filenames)]))
        labed_lab1 = sitk.GetArrayFromImage(labed_lab1)
        label_mask = np.where(labed_lab1 > 0, 1, 0).astype(np.float32)
        labed_img1 = labed_img1 * label_mask
        labed_img1 = labed_img1[np.newaxis, :, :, :]
        labed_lab1 = [np.where(labed_lab1 == i, 1, 0)[np.newaxis, :, :, :] for i in self.list]
        labed_lab1 = np.concatenate(labed_lab1, axis=0)
        labed_lab1 = labed_lab1.astype(np.float32)

        labed_img2 = sitk.ReadImage(
            join(self.file_dir, 'image', self.labeled_filenames[index % len(self.labeled_filenames)]))
        labed_img2 = sitk.GetArrayFromImage(labed_img2)
        labed_img2 = Nor(limit(labed_img2))
        labed_img2 = labed_img2.astype(np.float32)
        labed_lab2 = sitk.ReadImage(
            join(self.file_dir, 'label', self.labeled_filenames[index % len(self.labeled_filenames)]))
        labed_lab2 = sitk.GetArrayFromImage(labed_lab2)
        label_mask = np.where(labed_lab2 > 0, 1, 0).astype(np.float32)
        labed_img2 = labed_img2 * label_mask
        labed_img2 = labed_img2[np.newaxis, :, :, :]
        labed_lab2 = [np.where(labed_lab2 == i, 1, 0)[np.newaxis, :, :, :] for i in self.list]
        labed_lab2 = np.concatenate(labed_lab2, axis=0)
        labed_lab2 = labed_lab2.astype(np.float32)

        return labed_img1, labed_lab1, labed_img2, labed_lab2, \
               self.labeled_filenames[index // len(self.labeled_filenames)], self.labeled_filenames[
                   index % len(self.labeled_filenames)]

    def __len__(self):
        return len(self.labeled_filenames) * len(self.labeled_filenames)

