from os.path import join
from os import listdir
import SimpleITK as sitk
from torch.utils import data
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii"])

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
    def __init__(self, unlabeled_file_dir):
        super(DatasetFromFolder3D, self).__init__()
        self.unlabeled_filenames = [x for x in listdir(unlabeled_file_dir) if is_image_file(x)]
        self.unlabeled_file_dir = unlabeled_file_dir

    def __getitem__(self, index):
        random_index = np.random.randint(low=0, high=len(self.unlabeled_filenames))
        unlabed_img1 = sitk.ReadImage(join(self.unlabeled_file_dir, self.unlabeled_filenames[random_index]))
        unlabed_img1 = sitk.GetArrayFromImage(unlabed_img1)
        unlabed_img1 = Nor(limit(unlabed_img1))
        unlabed_img1 = unlabed_img1.astype(np.float32)
        unlabed_img1 = unlabed_img1[np.newaxis, :, :, :]

        random_index = np.random.randint(low=0, high=len(self.unlabeled_filenames))
        unlabed_img2 = sitk.ReadImage(join(self.unlabeled_file_dir, self.unlabeled_filenames[random_index]))
        unlabed_img2 = sitk.GetArrayFromImage(unlabed_img2)
        unlabed_img2 = Nor(limit(unlabed_img2))
        unlabed_img2 = unlabed_img2.astype(np.float32)

        unlabed_img2 = unlabed_img2[np.newaxis, :, :, :]

        return unlabed_img1, unlabed_img2

    def __len__(self):
        return len(self.unlabeled_filenames)

