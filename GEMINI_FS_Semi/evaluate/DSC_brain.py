import os
from os import listdir
from os.path import join
import SimpleITK as sitk
from GEMINI_FS_Semi.utils.utils import to_categorical, dice
import numpy as np

def DSC(pred_dir, gt_dir):
    list = [0,2,3,4,7,8,10,11,12,13,14,15,16,17,18,24,28,41,42,43,46,47,49,50,51,52,53,54,60]
    image_filenames = listdir(pred_dir)
    DSC = np.zeros((len(list), len(image_filenames)))

    for i in range(len(image_filenames)):
        name = image_filenames[i]

        predict = sitk.ReadImage(join(pred_dir, name))
        predict = sitk.GetArrayFromImage(predict)
        predict = to_categorical(predict, num_classes=len(list))

        labed_lab = sitk.ReadImage(join(gt_dir, name[:-4]+"i.gz"))
        labed_lab = sitk.GetArrayFromImage(labed_lab)
        labed_lab = [np.where(labed_lab == i, 1, 0)[np.newaxis, :, :, :] for i in list]
        labed_lab = np.concatenate(labed_lab, axis=0)

        for c in range(len(list)):
            DSC[c, i] = dice(predict[c], labed_lab[c])

        print(name, DSC[1:, i])

    print(np.mean(DSC[1:, :], axis=1))
    print(np.mean(DSC[1:, :]), np.std(np.mean(DSC[1:, :], axis=0)))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    DSC('G:\GEMINI\GEMINI_FS_Semi/results\GEMINI_FS_Semi_brain\seg', 'G:\GEMINI\GEMINI_FS_Semi/data\T2_BrainMR\\test\labeled\label')



