import os
from tqdm import tqdm
from dipy.io.image import load_nifti
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import logging
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import torch.fft
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import normalized_root_mse as NRMSE
from torchvision.models import vgg19
from math import fabs, sin, cos, radians
import hdf5storage
import cv2


def center_crop_image(image, size):
    """
    Crop the center part of image at last two axes.

    :param image: any type array of shape (..., h, w).
    :param size: tuple or list of int, two elements, (h1, w1).

    :return: the same type array of shape (..., h1, w1).
    """
    h, w = image.shape[-2:]
    h1, w1 = size
    if h < h1 or w < w1:
        raise ValueError("the value of size is not applicable.")
    up_index = (h - h1) // 2
    down_index = up_index + h1
    left_index = (w - w1) // 2
    right_index = left_index + w1
    return image[..., up_index: down_index, left_index: right_index]


def ssim(img1, img2, device, window_size=11, size_average=True):
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    (_, channel, height, width) = img1.size()
    window = create_window(window_size, channel).to(device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(nn.Module):
    def __init__(self, device, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.device = device

    def forward(self, pred, gt):
        return 1 - ssim(pred, gt, self.device, self.window_size, self.size_average)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = vgg19(pretrained=True).features[:16].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, y_pred, y_true):
        y_pred_features = self.vgg(y_pred)
        y_true_features = self.vgg(y_true)

        return F.l1_loss(y_pred_features, y_true_features)


sobel_kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(
    0)
sobel_kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(
    0)

sobel_kernel_x = sobel_kernel_x.repeat(10, 90, 1, 1)
sobel_kernel_y = sobel_kernel_y.repeat(10, 90, 1, 1)


def sobel_filter(img, device):
    # print('---------------------', img.shape)
    g_x = F.conv2d(img, sobel_kernel_x.to(device), padding=1)
    g_y = F.conv2d(img, sobel_kernel_y.to(device), padding=1)
    edge_map = torch.sqrt(g_x ** 2 + g_y ** 2)

    return edge_map


class EdgeLoss(nn.Module):
    def __init__(self, device):
        super(EdgeLoss, self).__init__()
        self.device = device

    def forward(self, pred, gt):
        pred_edge = sobel_filter(pred, self.device)
        gt_edge = sobel_filter(gt, self.device)
        edge_loss = torch.mean(torch.abs(pred_edge - gt_edge))

        return edge_loss


def fft2c3d(data):
    data = torch.fft.ifftshift(data, dim=(-3, -2, -1))
    data = torch.fft.fftn(data, dim=(-3, -2, -1), norm='ortho')
    data = torch.fft.fftshift(data, dim=(-3, -2, -1))
    return data


def ifft2c3d(data):
    data = torch.fft.ifftshift(data, dim=(-3, -2, -1))
    data = torch.fft.ifftn(data, dim=(-3, -2, -1), norm='ortho')
    data = torch.fft.fftshift(data, dim=(-3, -2, -1))
    return data


def fft2c2d(x):
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.fftn(x, dim=(-2, -1), norm='ortho')
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return x


def ifft2c2d(x):
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.ifftn(x, dim=(-2, -1), norm='ortho')
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return x


def batch_psnr_ssim(recon, ref):
    # recon: [b, k]
    recon, ref = recon.detach().cpu().numpy(), ref.detach().cpu().numpy()
    b_psnr, b_ssim = [], []
    for i in range(recon.shape[0]):
        for j in range(recon.shape[1]):
            b_psnr.append(peak_signal_noise_ratio(recon[i, j, ...], ref[i, j, ...], data_range=ref[i, j, ...].max()))
            b_ssim.append(structural_similarity(recon[i, j, ...], ref[i, j, ...], data_range=ref[i, j, ...].max()))

    return np.mean(b_psnr), np.mean(b_ssim)


def get_logger(filepath):
    logger = logging.getLogger('INRSSR')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(filepath)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def dict_to_csv(data, columns, url):
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(url, index=False)


def get_in_brain(data, mask):
    # data: [x, y, z]   mask: [x, y, z]
    # ravel()将多维数组降为一维，flatnonzero()返回数组中非零元素的下标
    nonzero_metrix = data.ravel()[np.flatnonzero(mask)]

    return nonzero_metrix


class second_moment_loss(nn.Module):
    def __init__(self):
        super(second_moment_loss, self).__init__()

    def forward(self, pred, gt):
        pred_mean = torch.mean(pred, dim=-1, keepdim=True)
        gt_mean = torch.mean(gt, dim=-1, keepdim=True)

        pred_second_moment = torch.mean((pred - pred_mean) ** 2, dim=-1)
        gt_second_moment = torch.mean((gt - gt_mean) ** 2, dim=-1)

        return torch.mean(torch.abs(pred_second_moment - gt_second_moment))


class SVL_loss(nn.Module):
    def __init__(self):
        super(SVL_loss, self).__init__()
        self.vl = second_moment_loss()

    def forward(self, pred, gt):
        l1_loss = torch.mean(torch.abs(torch.mean(pred, dim=-1) - torch.mean(gt, dim=-1)))
        vl_loss = self.vl(pred, gt)

        return l1_loss + vl_loss


def concat_dwi(lar_index, lar_dwi, pre_index, pre_dwi, patch):
    """
    dwi合并。
    """
    pre_zero = torch.zeros_like(patch)

    for key, value in enumerate(lar_index):
        pre_zero[:, value] = lar_dwi[:, key]

    for key, value in enumerate(pre_index):
        pre_zero[:, value] = pre_dwi[:, key]

    return pre_zero


def valid_metrics(pred, gt):
    mse = MSE(pred, gt)
    nrmse = NRMSE(pred, gt)
    psnr = PSNR(pred, gt, data_range=1)

    # ssim = SSIM(pred, gt)
    ssim = SSIM(pred, gt, data_range=1, win_size=3)
    return mse, nrmse, psnr, ssim


def get_scores_model(model, data_loader):
    metrics = []
    # toggle model to eval mode
    model.eval()
    # turn off gradients since they will not be used here
    with torch.no_grad():
        for lar_dwi, lar_index, pre_index, patch in data_loader:
            x = lar_dwi.cuda()
            x = x.to(torch.float32)
            y = model(x)
            y_gt = patch.cuda()
            mse, nrmse, psnr, ssim = valid_metrics(y.cpu().detach().numpy(), y_gt.cpu().detach().numpy())
            metrics.append([mse, nrmse, psnr, ssim])
    return np.mean(metrics, axis=0)


def save_test_prediction(model, data_loader):
    # toggle model to eval mode
    model.eval()
    y_pred = np.zeros(0)
    y_gt = np.zeros(0)
    # y_pred = []
    # y_gt = []
    i = 0
    # turn off gradients since they will not be used here
    with torch.no_grad():
        for lar_dwi, lar_index, har_index, har_dwi in data_loader:
            lar_dwi = lar_dwi.cuda()
            lar_dwi = lar_dwi.to(torch.float32)
            har_pred = model(lar_dwi)
            # y_pred.append(har_pred.cpu().detach().numpy())
            # y_gt.append(har_dwi)
            if i == 0:
                y_pred = har_pred.cpu().detach().numpy()
                y_gt = har_dwi.cpu().detach().numpy()
            else:
                y_pred = np.concatenate((y_pred, har_pred.cpu().detach().numpy()), axis=0)
                y_gt = np.concatenate((y_gt, har_dwi), axis=0)
            i += 1
    return np.array(y_pred), np.array(y_gt)


def get_metrics(pred, gt):
    mse = MSE(pred, gt)
    nrmse = NRMSE(pred, gt)
    psnr = PSNR(pred, gt, data_range=1)
    # ssim = SSIM(pred, gt)
    ssim = 0.0
    for i in range(pred.shape[0]):
        for j in range(pred.shape[-2]):
            # ssim += SSIM(pred[i, :, :, :, j], gt[i, :, :, :, j], data_range=np.max(gt[i, :, :, :, j]))
            ssim += SSIM(pred[i, :, :, j, :], gt[i, :, :, j, :], data_range=1)
    ssim = ssim / (pred.shape[0] * pred.shape[-2])

    return mse, nrmse, psnr, ssim


def get_SH_metrics(pred, gt):
    mse = MSE(pred, gt)
    nrmse = NRMSE(pred, gt)
    psnr = PSNR(pred, gt, data_range=1)
    # ssim = SSIM(pred, gt)
    ssim = 0.0
    for i in range(pred.shape[2]):
            ssim += SSIM(pred[:, :, i, :], gt[:, :, i, :], data_range=1)
    ssim = ssim / (pred.shape[2])

    return mse, nrmse, psnr, ssim


def Get_logger(path):
    logger = logging.getLogger(name=path)
    logger.setLevel(level=logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s:\t %(message)s')
    stream_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s:\t %(message)s')
    if not logger.handlers:
        file_handler = logging.FileHandler(filename=path, mode='a+', encoding='utf-8')
        file_handler.setLevel(level=logging.INFO)
        file_handler.setFormatter(file_formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level=logging.INFO)
        stream_handler.setFormatter(stream_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def one_pic_metrics(pred, gt):
    mse = MSE(pred, gt)
    nrmse = NRMSE(pred, gt)
    psnr = PSNR(pred, gt, data_range=1)
    ssim = SSIM(pred, gt, data_range=1)
    return mse, nrmse, psnr, ssim


def one_subject_metrics(pred, gt):
    mse = MSE(pred, gt)
    nrmse = NRMSE(pred, gt)
    psnr = PSNR(pred, gt, data_range=1)
    ssim = 0.0
    for i in range(gt.shape[-1]):
        ssim += SSIM(pred[:, :, :, i], gt[:, :, :, i], data_range=1)
    ssim = ssim / gt.shape[-1]
    return mse, nrmse, psnr, ssim


def get_single_metric(pred, gt):
    mse = MSE(pred, gt)
    nrmse = NRMSE(pred, gt)
    psnr = PSNR(pred, gt, data_range=1)
    ssim = SSIM(pred, gt, data_range=1)
    return mse, nrmse, psnr, ssim


def get_whole_metric(pred, gt):
    mse = MSE(pred, gt)
    nrmse = NRMSE(pred, gt)
    psnr = PSNR(pred, gt, data_range=1)
    # ssim = SSIM(pred, gt)
    ssim = 0.0
    for i in range(pred.shape[0]):
        for j in range(pred.shape[-1]):
            # ssim += SSIM(pred[i, :, :, :, j], gt[i, :, :, :, j], data_range=np.max(gt[i, :, :, :, j]))
            ssim += SSIM(pred[i, :, :, :, j], gt[i, :, :, :, j], data_range=1)
    ssim = ssim / (pred.shape[0] * pred.shape[-1])

    return mse, nrmse, psnr, ssim


def get_center_metric(pred, gt):
    mse = MSE(pred, gt)
    nrmse = NRMSE(pred, gt)
    psnr = PSNR(pred, gt, data_range=1)
    # ssim = SSIM(pred, gt)
    ssim = 0.0
    for i in range(pred.shape[0]):
        for j in range(pred.shape[-2]):
            # ssim += SSIM(pred[i, :, :, :, j], gt[i, :, :, :, j], data_range=np.max(gt[i, :, :, :, j]))
            ssim += SSIM(pred[i, :, :, j, :], gt[i, :, :, j, :], data_range=1)
    ssim = ssim / (pred.shape[0] * pred.shape[-2])

    return mse, nrmse, psnr, ssim


def rotate_bound(image, angle):
    """
     . rotate image in given angle
     . @param image    opencv读取后的图像cv2.error: OpenCV(4.7.0) :-1: error: (-5:Bad argument) in function 'warpAffine'
> Overload resolution failed:
     . @param angle    (逆)旋转角度
    """
    h, w = image.shape[:2]  # 返回(高,宽,色彩通道数),此处取前两个值返回
    newW = int(h * fabs(sin(radians(angle))) + w * fabs(cos(radians(angle))))
    newH = int(w * fabs(sin(radians(angle))) + h * fabs(cos(radians(angle))))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    M[0, 2] += (newW - w) / 2
    M[1, 2] += (newH - h) / 2
    return cv2.warpAffine(image, M, (newW, newH), borderValue=(255, 255, 255))


def recover_1d(flat_data, shape=None):
    if shape is None:
        shape = [10, 145, 174, 10, 90]
    data = np.zeros(shape)

    order = 0
    if flat_data.shape[0] != shape[0] * shape[1] * shape[2] * shape[3]:
        print('Dimension Unmatch')
    for sub in range(data.shape[0]):
        for w in range(data.shape[1]):
            for h in range(data.shape[2]):
                for s in range(data.shape[3]):
                    data[sub, w, h, s, :] = flat_data[order, :]
                    order += 1
    return data


def get_known(gt, d_selected):
    w, h, s, d = gt.shape
    known_data = []
    unknown_data = []
    for i in range(d):
        if i in d_selected:
            known_data.append(gt[:, :, :, i])
        else:
            unknown_data.append(gt[:, :, :, i])
    known_data = np.transpose(np.array(known_data, np.float32), (1, 2, 3, 0))
    unknown_data = np.transpose(np.array(unknown_data, np.float32), (1, 2, 3, 0))
    return known_data, unknown_data