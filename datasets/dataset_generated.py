import os

import numpy as np
import torch

from datasets.dataset import Dataset


class GeneratedDataset(Dataset):
    def __init__(self, root_data,
                 train_val_test,
                 device='cuda',
                 **kwargs):
        super().__init__(train_val_test, device, **kwargs)
        self.root_data = root_data
        self.on_the_fly_label = False
        self.init_paths()
        self.voxel_spacing = [1.0, 1.0, 1.0]
        self.num_labels = kwargs.pop('num_labels', 26)
        self.intensities = {'m0': kwargs.pop('intensities_m0', np.array([0] + [25] * (self.num_labels - 1))),
                            'm1': kwargs.pop('intensities_m1', np.array([225] * self.num_labels)),
                            's0': kwargs.pop('intensities_s0', np.array([0] + [5] * (self.num_labels - 1))),
                            's1': kwargs.pop('intensities_s1', np.array([25] * self.num_labels)),
                            'zero_background_prob': kwargs.pop('zero_background_prob', 0.2)}

    def init_paths(self):
        self.labels_path = f'{self.root_data}/{self.train_val_test}/labels.npy'
        if os.path.exists(self.labels_path):
            label_maps_np = np.load(self.labels_path)
            self.label_maps = [torch.from_numpy(label_maps_np[i, :, :, :]).type(torch.int64) for i in
                               range(label_maps_np.shape[0])]
            self.shape = self.label_maps[0].shape
            self.num_dim = len(self.shape)
        else:
            print('ON THE FLY LABEL GENERATION')
            self.on_the_fly_label = True
            self.label_maps = []

        print('ON THE FLY IMAGE GENERATION')
        return

    def label_to_image(self, label):
        # Apply gray scale transformations
        # 1. Create synthetic image - \w random intensity means and standard deviations.
        mean = torch.rand((self.num_labels)) * (self.intensities['m1'] - self.intensities['m0']) + self.intensities['m0']
        std = torch.rand((self.num_labels)) * (self.intensities['s1'] - self.intensities['s0']) + self.intensities['s0']
        mean = torch.gather(mean, 0, label.view(-1).type(torch.int64)).view(label.shape)
        std = torch.gather(std, 0, label.view(-1).type(torch.int64)).view(label.shape)
        image = torch.randn_like(label, dtype=torch.float32, requires_grad=False) * std + mean

        # 2. Zero background.
        if self.intensities['zero_background_prob'] > 0:
            rand_flip = torch.rand(*([1] * self.num_dim)) < self.intensities['zero_background_prob']
            image *= 1. - torch.logical_and(label == 0, rand_flip).type(image.dtype)
        return image

    def generate_moving_fixed(self, label):
        # Warp the labels, first moving
        DVF_moving = self.augmenter.generate_DVF_moving_to_fixed()
        moving_lbl = self.augmenter.transformer_binary(
            src=label.unsqueeze(0).unsqueeze(0).type(torch.float32),
            flow=DVF_moving).squeeze(0).squeeze(0)

        # Warp the labels, now fixed
        DVF_fixed = self.augmenter.generate_DVF_moving_to_fixed()
        fixed_lbl = self.augmenter.transformer_binary(
            src=label.unsqueeze(0).unsqueeze(0).type(torch.float32),
            flow=DVF_fixed).squeeze(0).squeeze(0)

        # Transform label to grayscale image and add artifacts
        moving_img = self.label_to_image(moving_lbl).type(torch.float32)
        fixed_img = self.label_to_image(fixed_lbl).type(torch.float32)

        moving_img = self.scaling_function(moving_img)
        fixed_img = self.scaling_function(fixed_img)
        return moving_img, fixed_img, \
               moving_lbl, fixed_lbl

    def __getitem__(self, i):
        if self.on_the_fly_label:
            label = []
        else:
            label = self.label_maps[i].type(torch.int64)
        moving_img, fixed_img, moving_lbl, fixed_lbl = self.generate_moving_fixed(label)
        return moving_img.unsqueeze(0), fixed_img.unsqueeze(0), \
               moving_lbl.unsqueeze(0), fixed_lbl.unsqueeze(0)

    def __len__(self):
        return len(self.label_maps)