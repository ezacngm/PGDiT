import numpy as np
import nibabel as nib
import torch
import torchvision.transforms as T
import os


def to_tensor(input_tensor):
    if isinstance(input_tensor, np.ndarray):
        input_tensor = torch.from_numpy(input_tensor)
    return input_tensor

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.numpy()
    return tensor

def reshape_img(img_input):
    if isinstance(img_input, str):
        img = nib.load(img_input).get_fdata()
        print('loaded image orginal shape: ', img.shape)
    elif isinstance(img_input, np.ndarray):
        img = img_input
        print('Input is NumPy array, original shape:', img.shape)
    else:
        raise ValueError("Input must be a file path string or a NumPy array.")

    trans_th = T.Compose([
        T.Pad([24, 9, 23, 9]),
        T.CenterCrop(160)
    ])
    img = to_tensor(img)
    img = img.permute(3, 2, 1, 0)
    img = torch.flip(trans_th(img), dims=[2])
    img = to_numpy(img)
    print('now shape changed into: ',img.shape)
    return img

def reshape_mask(img_input):
    if isinstance(img_input, str):
        img = nib.load(img_input).get_fdata()
        print('loaded image orginal shape: ', img.shape)
    elif isinstance(img_input, np.ndarray):
        img = img_input
        print('Input is NumPy array, original shape:', img.shape)
    else:
        raise ValueError("Input must be a file path string or a NumPy array.")
    trans_th = T.Compose([
        T.Pad([24, 9, 23, 9]),
        T.CenterCrop(160)
    ])
    img = to_tensor(img)
    img = img.permute(2, 1, 0)
    img = torch.flip(trans_th(img), dims=[1])
    img = to_numpy(img)
    print('now shape changed into: ',img.shape)
    return img

def sliding_windows_cut(volume, patch_size=(63, 63)):
    """
    Args:
        volume: Tensor [90, H, W]
        patch_size: (ph, pw)
    Returns:
        patches: Tensor [N, 90, ph, pw]
        positions: list of (h, w)
    """
    C, H, W = volume.shape
    ph, pw = patch_size

    def compute_positions(size, patch_size):
        positions = set()
        positions.add(0)
        positions.add(size - patch_size)
        positions.add(size // 2 - patch_size // 2)
        return sorted(positions)

    h_starts = compute_positions(H, ph)
    w_starts = compute_positions(W, pw)

    patches = []
    positions = []

    for h in h_starts:
        for w in w_starts:
            patch = volume[:, h:h+ph, w:w+pw]
            patches.append(patch)
            positions.append((h, w))

    return torch.stack(patches), positions


def sliding_windows_stick(patches, positions, output_shape=(90, 160, 120), patch_size=(63, 63)):
    """
    Reconstructs the full volume from patches using mean in overlapping regions.

    Args:
        patches: Tensor of shape [N, 90, ph, pw]
        positions: list of (h, w) coordinates
        output_shape: tuple (C=90, H, W)
        patch_size: (ph, pw)

    Returns:
        volume: Tensor [90, H, W]
    """
    C, H, W = output_shape
    ph, pw = patch_size

    volume = torch.zeros((C, H, W), dtype=patches.dtype, device=patches.device)
    count = torch.zeros((C, H, W), dtype=patches.dtype, device=patches.device)

    for patch, (h, w) in zip(patches, positions):
        volume[:, h:h + ph, w:w + pw] += patch
        count[:, h:h + ph, w:w + pw] += 1

    # To avoid division by zero (just in case, although shouldn't happen)
    count = torch.clamp(count, min=1.0)

    return volume / count
