from os.path import join
from os import listdir
from torch.utils import data
import numpy as np
import cv2


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

class DatasetFromFolder2D(data.Dataset):
    def __init__(self, file_dir):
        super(DatasetFromFolder2D, self).__init__()
        self.filenames = [x for x in listdir(join(file_dir, 'image')) if is_image_file(x)]
        self.file_dir = file_dir
        self.labeled_filenames = self.filenames

    def __getitem__(self, index):
        labed_img = cv2.imread(join(self.file_dir, 'image', self.labeled_filenames[index]),
                                      flags=cv2.IMREAD_GRAYSCALE)
        labed_img = cv2.resize(labed_img, (512, 512), interpolation=cv2.INTER_LINEAR)
        labed_img = labed_img / 255.
        labed_img = labed_img.astype(np.float32)
        labed_img = labed_img[np.newaxis, :, :]

        labed_lab_heart = cv2.imread(join(self.file_dir, 'label', 'heart', self.labeled_filenames[index]),
                                      flags=cv2.IMREAD_GRAYSCALE)
        labed_lab_heart = cv2.resize(labed_lab_heart, (512, 512), interpolation=cv2.INTER_NEAREST)
        labed_lab_heart = np.where(labed_lab_heart > 0, 1, 0).astype(np.float32)
        labed_lab_heart = labed_lab_heart[np.newaxis, :, :]

        labed_lab_lung = cv2.imread(join(self.file_dir, 'label', 'lung', self.labeled_filenames[index]),
                                      flags=cv2.IMREAD_GRAYSCALE)
        labed_lab_lung = cv2.resize(labed_lab_lung, (512, 512), interpolation=cv2.INTER_NEAREST)
        labed_lab_lung = np.where(labed_lab_lung > 0, 1, 0).astype(np.float32)
        labed_lab_lung = labed_lab_lung[np.newaxis, :, :]

        labed_lab_clavicle = cv2.imread(join(self.file_dir, 'label', 'clavicle', self.labeled_filenames[index]),
                                      flags=cv2.IMREAD_GRAYSCALE)
        labed_lab_clavicle = cv2.resize(labed_lab_clavicle, (512, 512), interpolation=cv2.INTER_NEAREST)
        labed_lab_clavicle = np.where(labed_lab_clavicle > 0, 1, 0).astype(np.float32)
        labed_lab_clavicle = labed_lab_clavicle[np.newaxis, :, :]

        labed_lab = np.concatenate([labed_lab_heart, labed_lab_lung, labed_lab_clavicle], axis=0)
        # labed_lab = np.concatenate([labed_lab_heart, labed_lab_lung], axis=0)
        labed_lab = labed_lab.astype(np.float32)


        return labed_img, labed_lab, self.labeled_filenames[index]

    def __len__(self):
        return len(self.labeled_filenames)

