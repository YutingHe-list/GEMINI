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
        labed_img1 = cv2.imread(join(self.file_dir, 'image', self.labeled_filenames[index // len(self.labeled_filenames)]),
                                      flags=cv2.IMREAD_GRAYSCALE)
        labed_img1 = cv2.resize(labed_img1, (512, 512), interpolation=cv2.INTER_LINEAR)
        labed_img1 = labed_img1 / 255.
        labed_img1 = labed_img1.astype(np.float32)
        labed_img1 = labed_img1[np.newaxis, :, :]

        labed_lab1_heart = cv2.imread(join(self.file_dir, 'label', 'heart', self.labeled_filenames[index // len(self.labeled_filenames)]),
                                     flags=cv2.IMREAD_GRAYSCALE)
        labed_lab1_heart = cv2.resize(labed_lab1_heart, (512, 512), interpolation=cv2.INTER_NEAREST)
        labed_lab1_heart = np.where(labed_lab1_heart > 0, 1, 0).astype(np.float32)
        labed_lab1_heart = labed_lab1_heart[np.newaxis, :, :]

        labed_lab1_lung = cv2.imread(join(self.file_dir, 'label', 'lung', self.labeled_filenames[index // len(self.labeled_filenames)]),
                                    flags=cv2.IMREAD_GRAYSCALE)
        labed_lab1_lung = cv2.resize(labed_lab1_lung, (512, 512), interpolation=cv2.INTER_NEAREST)
        labed_lab1_lung = np.where(labed_lab1_lung > 0, 1, 0).astype(np.float32)
        labed_lab1_lung = labed_lab1_lung[np.newaxis, :, :]

        labed_lab1_clavicle = cv2.imread(join(self.file_dir, 'label', 'clavicle', self.labeled_filenames[index // len(self.labeled_filenames)]),
                                        flags=cv2.IMREAD_GRAYSCALE)
        labed_lab1_clavicle = cv2.resize(labed_lab1_clavicle, (512, 512), interpolation=cv2.INTER_NEAREST)
        labed_lab1_clavicle = np.where(labed_lab1_clavicle > 0, 1, 0).astype(np.float32)
        labed_lab1_clavicle = labed_lab1_clavicle[np.newaxis, :, :]

        labed_lab1 = np.concatenate([labed_lab1_heart, labed_lab1_lung, labed_lab1_clavicle], axis=0)
        # labed_lab1 = np.concatenate([labed_lab1_heart, labed_lab1_lung], axis=0)
        labed_lab1 = labed_lab1.astype(np.float32)

        labed_img2 = cv2.imread(join(self.file_dir, 'image', self.labeled_filenames[index % len(self.labeled_filenames)]),
                                        flags=cv2.IMREAD_GRAYSCALE)
        labed_img2 = cv2.resize(labed_img2, (512, 512), interpolation=cv2.INTER_LINEAR)
        labed_img2 = labed_img2 / 255.
        labed_img2 = labed_img2.astype(np.float32)
        labed_img2 = labed_img2[np.newaxis, :, :]

        labed_lab2_heart = cv2.imread(
            join(self.file_dir, 'label', 'heart', self.labeled_filenames[index % len(self.labeled_filenames)]),
            flags=cv2.IMREAD_GRAYSCALE)
        labed_lab2_heart = cv2.resize(labed_lab2_heart, (512, 512), interpolation=cv2.INTER_NEAREST)
        labed_lab2_heart = np.where(labed_lab2_heart > 0, 1, 0).astype(np.float32)
        labed_lab2_heart = labed_lab2_heart[np.newaxis, :, :]

        labed_lab2_lung = cv2.imread(
            join(self.file_dir, 'label', 'lung', self.labeled_filenames[index % len(self.labeled_filenames)]),
            flags=cv2.IMREAD_GRAYSCALE)
        labed_lab2_lung = cv2.resize(labed_lab2_lung, (512, 512), interpolation=cv2.INTER_NEAREST)
        labed_lab2_lung = np.where(labed_lab2_lung > 0, 1, 0).astype(np.float32)
        labed_lab2_lung = labed_lab2_lung[np.newaxis, :, :]

        labed_lab2_clavicle = cv2.imread(
            join(self.file_dir, 'label', 'clavicle', self.labeled_filenames[index % len(self.labeled_filenames)]),
            flags=cv2.IMREAD_GRAYSCALE)
        labed_lab2_clavicle = cv2.resize(labed_lab2_clavicle, (512, 512), interpolation=cv2.INTER_NEAREST)
        labed_lab2_clavicle = np.where(labed_lab2_clavicle > 0, 1, 0).astype(np.float32)
        labed_lab2_clavicle = labed_lab2_clavicle[np.newaxis, :, :]

        labed_lab2 = np.concatenate([labed_lab2_heart, labed_lab2_lung, labed_lab2_clavicle], axis=0)
        labed_lab2 = labed_lab2.astype(np.float32)

        return labed_img1, labed_lab1, labed_img2, labed_lab2, \
               self.labeled_filenames[index // len(self.labeled_filenames)], self.labeled_filenames[
                   index % len(self.labeled_filenames)]

    def __len__(self):
        return len(self.labeled_filenames) * len(self.labeled_filenames)

