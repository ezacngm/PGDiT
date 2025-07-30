import math
import random
import nibabel as nib

import numpy as np
from math import exp
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error

import math
import random

from PIL import Image
import numpy as np
from math import exp
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import mean_squared_error as MSE
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanSquaredError



def get_metric(gt,img,data_range=1):
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.numpy()
    img_gt_np = gt.astype(np.float32)
    img = img.astype(np.float32)
    ssim_value= SSIM(img_gt_np, img, data_range=data_range)
    psnr_value = PSNR(img_gt_np, img, data_range=data_range)
    rmse_value = np.sqrt(MSE(img_gt_np, img))
    return ssim_value, psnr_value, rmse_value



def metric_nii(image1_path,image2_path):

    # Load the images using nibabel
    img1 = nib.load(image1_path)
    img2 = nib.load(image2_path)

    # Convert the images to NumPy arrays
    data1 = img1.get_fdata()
    data2 = img2.get_fdata()

    # Ensure both images have the same shape
    if data1.shape != data2.shape:
        raise ValueError("The two images must have the same shape to compute SSIM, PSNR, and RMSE.")
    print("img1 range:", np.min(data1), np.max(data1),data1.shape)
    print("img2 range:", np.min(data2), np.max(data2),data2.shape)
    ssim_value, _ = ssim(data1, data2, data_range=data1.max() - data1.min(), full=True)
    psnr_value = psnr(data1, data2, data_range=data1.max() - data1.min())
    rmse_value = np.sqrt(mean_squared_error(data1, data2))
    print(ssim_value, psnr_value,rmse_value)


def metric_tensor(b,a):
    if not isinstance(a, torch.Tensor):
        a = torch.from_numpy(a)
    if not isinstance(b, torch.Tensor):
        b = torch.from_numpy(b)
    psnr = PeakSignalNoiseRatio(data_range=1.0)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    mse = MeanSquaredError()

    psnr_val = psnr(preds=b, target=a).item()
    ssim_val = ssim(preds=b, target=a).item()
    mse_val = mse(preds=b, target=a)
    rmse_val = torch.sqrt(mse_val).item()
    return ssim_val,psnr_val,rmse_val
