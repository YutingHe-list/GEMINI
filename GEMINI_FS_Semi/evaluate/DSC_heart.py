import os
from os import listdir
from os.path import join
import SimpleITK as sitk
import numpy as np

from GEMINI_FS_Semi.utils.utils import to_categorical, dice


def DSC(pred_dir, gt_dir):
    image_filenames = listdir(pred_dir)
    DSC = np.zeros((8, len(image_filenames)))

    for i in range(len(image_filenames)):
        name = image_filenames[i]

        predict = sitk.ReadImage(join(pred_dir, name))
        predict = sitk.GetArrayFromImage(predict)
        predict = to_categorical(predict, num_classes=8)

        groundtruth = sitk.ReadImage(join(gt_dir, name))
        groundtruth = sitk.GetArrayFromImage(groundtruth)
        groundtruth = np.where(groundtruth == 205, 1, groundtruth)
        groundtruth = np.where(groundtruth == 420, 2, groundtruth)
        groundtruth = np.where(groundtruth == 500, 3, groundtruth)
        groundtruth = np.where(groundtruth == 550, 4, groundtruth)
        groundtruth = np.where(groundtruth == 600, 5, groundtruth)
        groundtruth = np.where(groundtruth == 820, 6, groundtruth)
        groundtruth = np.where(groundtruth == 850, 7, groundtruth)
        groundtruth = to_categorical(groundtruth, num_classes=8)

        for c in range(8):
            DSC[c, i] = dice(predict[c], groundtruth[c])

        print(name, DSC[1:, i])

    print(np.mean(DSC[1:, :], axis=1))
    print(np.mean(DSC[1:, :]), np.std(np.mean(DSC[1:, :], axis=0)))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    DSC('results\GEMINI_FS_Semi_heart\seg', 'data\T1_CardiacCT\\test\labeled\label')



