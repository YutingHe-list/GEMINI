from os.path import join
from os import listdir
import cv2
from torch.utils import data
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

class DatasetFromFolder2D(data.Dataset):
    def __init__(self, unlabeled_file_dir):
        super(DatasetFromFolder2D, self).__init__()
        self.unlabeled_filenames_AP_M = [x for x in listdir(join(unlabeled_file_dir, 'AP_M')) if is_image_file(x)]
        self.unlabeled_filenames_PA_M = [x for x in listdir(join(unlabeled_file_dir, 'PA_M')) if is_image_file(x)]
        self.unlabeled_filenames_AP_F = [x for x in listdir(join(unlabeled_file_dir, 'AP_F')) if is_image_file(x)]
        self.unlabeled_filenames_PA_F = [x for x in listdir(join(unlabeled_file_dir, 'PA_F')) if is_image_file(x)]
        self.unlabeled_file_dir = unlabeled_file_dir

    def __getitem__(self, index):
        random_AP_PA_F_M = np.random.randint(low=0, high=4)
        if random_AP_PA_F_M == 0:
            random_index = np.random.randint(low=0, high=len(self.unlabeled_filenames_AP_F))
            unlabed_img1 = cv2.imread(join(self.unlabeled_file_dir, 'AP_F', self.unlabeled_filenames_AP_F[random_index]),
                                      flags=cv2.IMREAD_GRAYSCALE)
            random_index = np.random.randint(low=0, high=len(self.unlabeled_filenames_AP_F))
            unlabed_img2 = cv2.imread(join(self.unlabeled_file_dir, 'AP_F', self.unlabeled_filenames_AP_F[random_index]),
                                      flags=cv2.IMREAD_GRAYSCALE)
        elif random_AP_PA_F_M == 1:
            random_index = np.random.randint(low=0, high=len(self.unlabeled_filenames_AP_M))
            unlabed_img1 = cv2.imread(join(self.unlabeled_file_dir, 'AP_M', self.unlabeled_filenames_AP_M[random_index]),
                                      flags=cv2.IMREAD_GRAYSCALE)
            random_index = np.random.randint(low=0, high=len(self.unlabeled_filenames_AP_M))
            unlabed_img2 = cv2.imread(join(self.unlabeled_file_dir, 'AP_M', self.unlabeled_filenames_AP_M[random_index]),
                                      flags=cv2.IMREAD_GRAYSCALE)
        elif random_AP_PA_F_M == 2:
            random_index = np.random.randint(low=0, high=len(self.unlabeled_filenames_PA_F))
            unlabed_img1 = cv2.imread(join(self.unlabeled_file_dir, 'PA_F', self.unlabeled_filenames_PA_F[random_index]),
                                      flags=cv2.IMREAD_GRAYSCALE)
            random_index = np.random.randint(low=0, high=len(self.unlabeled_filenames_PA_F))
            unlabed_img2 = cv2.imread(join(self.unlabeled_file_dir, 'PA_F', self.unlabeled_filenames_PA_F[random_index]),
                                      flags=cv2.IMREAD_GRAYSCALE)
        else:
            random_index = np.random.randint(low=0, high=len(self.unlabeled_filenames_PA_M))
            unlabed_img1 = cv2.imread(join(self.unlabeled_file_dir, 'PA_M', self.unlabeled_filenames_PA_M[random_index]),
                                      flags=cv2.IMREAD_GRAYSCALE)
            random_index = np.random.randint(low=0, high=len(self.unlabeled_filenames_PA_M))
            unlabed_img2 = cv2.imread(join(self.unlabeled_file_dir, 'PA_M', self.unlabeled_filenames_PA_M[random_index]),
                                      flags=cv2.IMREAD_GRAYSCALE)

        unlabed_img1 = cv2.resize(unlabed_img1, (512, 512))
        unlabed_img1 = unlabed_img1 / 255.
        unlabed_img1 = unlabed_img1.astype(np.float32)
        unlabed_img1 = unlabed_img1[np.newaxis, :, :]

        unlabed_img2 = cv2.resize(unlabed_img2, (512, 512))
        unlabed_img2 = unlabed_img2 / 255.
        unlabed_img2 = unlabed_img2.astype(np.float32)
        unlabed_img2 = unlabed_img2[np.newaxis, :, :]

        return unlabed_img1, unlabed_img2

    def __len__(self):
        return len(self.unlabeled_filenames_PA_F)+len(self.unlabeled_filenames_PA_M)+len(self.unlabeled_filenames_AP_F)+len(self.unlabeled_filenames_AP_M)

