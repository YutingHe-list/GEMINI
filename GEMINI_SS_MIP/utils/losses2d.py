from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import math

def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])

    if(penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx

    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0

def ncc_loss_mask(I, J, mask, win=None):
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    if win is None:
        win = [9] * ndims

    sum_filt = torch.ones([I.size()[1], 1, *win]).to("cuda")

    pad_no = math.floor(win[0] / 2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)

    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)

    cc = cross * cross / (I_var * J_var + 1e-5)

    return 1 - torch.sum(cc * mask)/(torch.sum(mask)+1e-5)

def ncc_loss(I, J, win=None):
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    if win is None:
        win = [9] * ndims

    sum_filt = torch.ones([1, 1, *win]).to("cuda")

    pad_no = math.floor(win[0]/2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1,1)
        padding = (pad_no, pad_no)
    else:
        stride = (1,1,1)
        padding = (pad_no, pad_no, pad_no)
    
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)

    cc = cross*cross / (I_var*J_var + 1e-5)

    return 1 - torch.mean(cc)

def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J
    I_sum = torch.mean(F.conv2d(I, filt, stride=stride, padding=padding, groups=filt.shape[0]), dim=1, keepdim=True)
    J_sum = torch.mean(F.conv2d(J, filt, stride=stride, padding=padding, groups=filt.shape[0]), dim=1, keepdim=True)
    I2_sum = torch.mean(F.conv2d(I2, filt, stride=stride, padding=padding, groups=filt.shape[0]), dim=1, keepdim=True)
    J2_sum = torch.mean(F.conv2d(J2, filt, stride=stride, padding=padding, groups=filt.shape[0]), dim=1, keepdim=True)
    IJ_sum = torch.mean(F.conv2d(IJ, filt, stride=stride, padding=padding, groups=filt.shape[0]), dim=1, keepdim=True)

    win_size = int(np.prod(win))
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross

def cos_loss(out_1, out_2, mask):
    loss = 1-torch.sum(torch.sum(F.normalize(out_1, dim=1) * F.normalize(out_2, dim=1), dim=1, keepdim=True) * mask)/torch.sum(mask)

    return loss

def MSE(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)
