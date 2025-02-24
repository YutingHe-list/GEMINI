from os.path import join
from os import listdir
from torch.utils import data
import numpy as np
import cv2

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

class DatasetFromFolder2D(data.Dataset):
    def __init__(self, unlabeled_file_dir, labeled_file_dir):
        super(DatasetFromFolder2D, self).__init__()
        self.labeled_filenames = [x for x in listdir(join(labeled_file_dir, 'image')) if is_image_file(x)]
        self.unlabeled_filenames = [x for x in listdir(join(unlabeled_file_dir, 'image')) if is_image_file(x)]
        self.labeled_file_dir = labeled_file_dir
        self.unlabeled_file_dir = unlabeled_file_dir

    def __getitem__(self, index):
        random_index = np.random.randint(low=0, high=len(self.labeled_filenames))
        labed_img = cv2.imread(join(self.labeled_file_dir, 'image', self.labeled_filenames[random_index]),
                                      flags=cv2.IMREAD_GRAYSCALE)
        labed_img = cv2.resize(labed_img, (512, 512))

        labed_img = labed_img / 255.
        labed_img = labed_img.astype(np.float32)
        labed_img = labed_img[np.newaxis, :, :]

        labed_lab_heart = cv2.imread(join(self.labeled_file_dir, 'label', 'heart', self.labeled_filenames[random_index]),
                                      flags=cv2.IMREAD_GRAYSCALE)
        labed_lab_heart = cv2.resize(labed_lab_heart, (512, 512), interpolation=cv2.INTER_NEAREST)

        labed_lab_heart = np.where(labed_lab_heart > 0, 1, 0).astype(np.float32)
        labed_lab_heart = labed_lab_heart[np.newaxis, :, :]

        labed_lab_lung = cv2.imread(join(self.labeled_file_dir, 'label', 'lung', self.labeled_filenames[random_index]),
                                      flags=cv2.IMREAD_GRAYSCALE)
        labed_lab_lung = cv2.resize(labed_lab_lung, (512, 512), interpolation=cv2.INTER_NEAREST)
        labed_lab_lung = np.where(labed_lab_lung > 0, 1, 0).astype(np.float32)
        labed_lab_lung = labed_lab_lung[np.newaxis, :, :]

        labed_lab_clavicle = cv2.imread(join(self.labeled_file_dir, 'label', 'clavicle', self.labeled_filenames[random_index]),
                                      flags=cv2.IMREAD_GRAYSCALE)
        labed_lab_clavicle = cv2.resize(labed_lab_clavicle, (512, 512), interpolation=cv2.INTER_NEAREST)
        labed_lab_clavicle = np.where(labed_lab_clavicle > 0, 1, 0).astype(np.float32)
        labed_lab_clavicle = labed_lab_clavicle[np.newaxis, :, :]

        labed_lab = np.concatenate([labed_lab_heart, labed_lab_lung, labed_lab_clavicle], axis=0)
        labed_lab = labed_lab.astype(np.float32)

        random_index = np.random.randint(low=0, high=len(self.unlabeled_filenames))
        unlabed_img1 = cv2.imread(join(self.unlabeled_file_dir, 'image', self.unlabeled_filenames[random_index]),
                                      flags=cv2.IMREAD_GRAYSCALE)
        unlabed_img1 = cv2.resize(unlabed_img1, (512, 512))
        unlabed_img1 = unlabed_img1 / 255.
        unlabed_img1 = unlabed_img1.astype(np.float32)
        unlabed_img1 = unlabed_img1[np.newaxis, :, :]

        random_index = np.random.randint(low=0, high=len(self.unlabeled_filenames))
        unlabed_img2 = cv2.imread(join(self.unlabeled_file_dir, 'image', self.unlabeled_filenames[random_index]),
                                      flags=cv2.IMREAD_GRAYSCALE)
        unlabed_img2 = cv2.resize(unlabed_img2, (512, 512))
        unlabed_img2 = unlabed_img2 / 255.
        unlabed_img2 = unlabed_img2.astype(np.float32)
        unlabed_img2 = unlabed_img2[np.newaxis, :, :]

        return labed_img, labed_lab, unlabed_img1, unlabed_img2

    def __len__(self):
        return len(self.unlabeled_filenames)+len(self.labeled_filenames)

