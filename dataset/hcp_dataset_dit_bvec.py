import os
import os.path as osp
import math
import random
import pickle
import warnings
import glob

import numpy as np
import torchvision.transforms as T
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.utils import save_image
metadata_folder = ""
bvec_folder = ""


class HCPDataset_Bvec(data.Dataset):
    def __init__(self, data_folder, sequence_length, train=True, resolution=160, shell=1, kth_subject=None,patch=False):

        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.classes = None
        self.cache = {}
        ext = f"{(shell) * 90}"
        self.patch = patch
        self.subjects_names = sorted(os.listdir(data_folder))
        if self.train:
            # For training, use the first 210 subjects
            self.patient_ids = self.subjects_names[:100]
            cache_file = osp.join(metadata_folder, f"metadata_frame{sequence_length}_train_{ext}_subs{len(self.patient_ids)}.pkl")
        else:
            if kth_subject is not None:

                self.patient_ids = self.subjects_names[kth_subject:kth_subject+1]
                cache_file = osp.join(metadata_folder,
                                      f"metadata_frame{sequence_length}_test_subject{kth_subject}_{ext}.pkl")
            else:
                self.patient_ids = self.subjects_names[100:120]
                cache_file = osp.join(metadata_folder, f"metadata_frame{sequence_length}_test_{ext}.pkl")

        # Load cache if it exists
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.filepath_all, self.train_length, self.bvecpath_all = pickle.load(f)
                print(f'{cache_file} is using')
        else:
            self.filepath_all = []
            self.bvecpath_all = []
            if shell == 1:
                folder_mid = "timehw"
            else:
                folder_mid = f"timehw_{ext}"
            for patient_id in self.patient_ids:
                filepath_single = sorted(
                    glob.glob(osp.join(data_folder, patient_id, folder_mid, "sub" + patient_id + "_slice*.npy"),
                              recursive=True))
                # Number of slices
                num_slices = len(filepath_single)
                print(f"patient {patient_id} has {num_slices} slices")
                bvec_filepath = osp.join(data_folder, patient_id, f"diffusion_{ext}_directions",
                                         patient_id + f"_bvecs_{ext}.txt")
                bvec_repeated = [bvec_filepath] * num_slices
                self.filepath_all += filepath_single
                self.bvecpath_all += bvec_repeated

            # Save the cache
            assert len(self.filepath_all) == len(self.bvecpath_all)
            self.train_length = len(self.filepath_all)
            with open(cache_file, 'wb') as f:
                pickle.dump((self.filepath_all, self.train_length, self.bvecpath_all), f)
        warnings.filterwarnings('ignore')

    def __len__(self):
        return len(self.filepath_all)

    def __getitem__(self, idx):
        singleslice_90directions = np.load(self.filepath_all[idx])
        singleslice_90directions_th = torch.from_numpy(singleslice_90directions)
        bvec_90directions = torch.from_numpy(np.loadtxt(self.bvecpath_all[idx]))
        if self.patch:
            trans_th = T.Compose([
                T.Pad([24, 9, 23, 9]),
                T.CenterCrop(160),
                T.Lambda(custom_crop_tensor),
                T.RandomCrop((self.resolution, self.resolution)),
                # T.Normalize(mean=(0.5,), std=(0.5,)),
                T.RandomHorizontalFlip()
            ])
        else:
            trans_th = T.Compose([
                T.Pad([24, 9, 23, 9]),
                T.CenterCrop(160),
                T.Lambda(custom_crop_tensor),
                # T.Normalize(mean=(0.5,), std=(0.5,)),
                # T.RandomHorizontalFlip()
            ])
        singleslice_90directions_th = trans_th(singleslice_90directions_th)
        return singleslice_90directions_th, bvec_90directions
def custom_crop_tensor(tensor_img):
    # Assume tensor_img shape is (C, H, W)
    return tensor_img[:, :, :]