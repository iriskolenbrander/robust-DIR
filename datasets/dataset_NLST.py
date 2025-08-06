import os

import numpy as np
import torch
from glob import glob

from datasets.dataset import Dataset


class DatasetNLST(Dataset):
    def __init__(self, root_data,
                 train_val_test,
                 device='cuda',
                 **kwargs):
        super().__init__(train_val_test, device, **kwargs)
        self.root_data = root_data
        self.init_paths()
        self.shape, self.voxel_spacing = self.get_image_header(self.fixed_img_path[0])

        # Label info
        self.organ_list = ['lungs', 'arteries', 'veins']
        self.dice_labels = [[0], [1], [2]]

    def init_paths(self):
        self.image0000 = glob('{}/{}/image/***_0000.nii.gz' .format(self.root_data, self.train_val_test))
        self.image0000.sort()
        self.image0001 = glob('{}/{}/image/***_0001.nii.gz'.format(self.root_data, self.train_val_test))
        self.image0001.sort()

        self.fixed_img_path = self.image0000
        self.moving_img_path = self.image0001
        self.fixed_lbl_path = [path.replace('image', 'label_mask') for path in self.fixed_img_path]
        self.moving_lbl_path = [path.replace('image', 'label_mask') for path in self.moving_img_path]
        self.fixed_pts_path = [path.replace('image', 'keypoints').replace('nii.gz', 'txt') for path in self.fixed_img_path]
        self.moving_pts_path = [path.replace('image', 'keypoints').replace('nii.gz', 'txt') for path in self.moving_img_path]

    def get_case_info(self, i):
        moving_path, fixed_path = self.get_paths(i)
        case = int(moving_path[-13:-11])
        return case

    @staticmethod
    def read_pts(file_name, skiprows=0):
        return torch.tensor(np.loadtxt(file_name, skiprows=skiprows), dtype=torch.float32)

    def get_landmarks(self, i):
        if os.path.isfile(self.fixed_pts_path[i]):
            fixed_landmarks = self.read_pts(self.fixed_pts_path[i]) + torch.tensor(self.offsets).flip(-1)
            moving_landmarks = self.read_pts(self.moving_pts_path[i]) + torch.tensor(self.offsets).flip(-1)

            indices_all = []
            for pts in [fixed_landmarks, moving_landmarks]:
                indices_all = indices_all + np.argwhere(pts[:, 0] >= self.shape[0]).tolist()[0]
                indices_all = indices_all + np.argwhere(pts[:, 1] >= self.shape[1]).tolist()[0]
                indices_all = indices_all + np.argwhere(pts[:, 2] >= self.shape[2]).tolist()[0]
                indices_all = indices_all + np.argwhere(pts[:, 0] < 0).tolist()[0]
                indices_all = indices_all + np.argwhere(pts[:, 1] < 0).tolist()[0]
                indices_all = indices_all + np.argwhere(pts[:, 2] < 0).tolist()[0]

            indices_all = np.unique(indices_all)

            if len(indices_all) > 0:
                fixed_landmarks = np.delete(fixed_landmarks, indices_all, axis=0)
                moving_landmarks = np.delete(moving_landmarks, indices_all, axis=0)
        else:
            fixed_landmarks, moving_landmarks = torch.tensor([0, 0, 0]), torch.tensor([0, 0, 0])
        return moving_landmarks.unsqueeze(0).to(self.device), fixed_landmarks.unsqueeze(0).to(self.device)

    def adjust_shape(self, new_shape):
        self.offsets = [(shp - old_shp)//fctr for (shp, old_shp, fctr) in zip(new_shape, self.shape, [2, 2, 2])]
        self.shape = new_shape

    def overfit_one(self, idx):
        self.overfit = True
        self.moving_img_path, self.fixed_img_path = [self.moving_img_path[idx]], [self.fixed_img_path[idx]]
        self.moving_lbl_path, self.fixed_lbl_path = [self.moving_lbl_path[idx]], [self.fixed_lbl_path[idx]]

    def __getitem__(self, idx):
        moving_img, fixed_img = super().__getitem__(idx)
        moving_lbl = self.read_image(self.moving_lbl_path[idx]).unsqueeze(0)
        fixed_lbl = self.read_image(self.fixed_lbl_path[idx]).unsqueeze(0)

        if self.augment_smod:
            # do this only baed on coin flip using npy
            if np.random.rand() > 0.5:
                print('augmenting')
                moving_img, fixed_img, (DVF_mov_synthetic, DVF_gt, DVF_fix_synthetic) = self.augmenter.augment_full(
                    moving_img.unsqueeze(0))
                self.DVF_mov_synthetic = DVF_mov_synthetic #.permute(0, 1, 4, 3, 2).flip(1)
                self.DVF_fix_synthetic = DVF_fix_synthetic #.permute(0, 1, 4, 3, 2).flip(1)
                fixed_lbl = self.augmenter.transformer_binary(src=moving_lbl.unsqueeze(0),
                                                              flow=DVF_fix_synthetic)
                moving_lbl = self.augmenter.transformer_binary(src=moving_lbl.unsqueeze(0),
                                                               flow=DVF_mov_synthetic)
                moving_img = moving_img.squeeze(0)
                fixed_img = fixed_img.squeeze(0)
                moving_lbl = moving_lbl.squeeze(0)
                fixed_lbl = fixed_lbl.squeeze(0)

        moving_img, _ = self.pad_torch(moving_img, self.shape, 0.0)
        fixed_img, _ = self.pad_torch(fixed_img, self.shape, 0.0)
        moving_lbl, _ = self.pad_torch(moving_lbl, self.shape, 0.0)
        fixed_lbl, _ = self.pad_torch(fixed_lbl, self.shape, 0.0)

        moving_img = self.scaling_function(moving_img)
        fixed_img = self.scaling_function(fixed_img)

        return moving_img, fixed_img, \
               moving_lbl, fixed_lbl
