import monai
import numpy as np
import torch
import torch.utils.data
import SimpleITK as sitk

from utils.utils import scaling_minmax

class Dataset(torch.utils.data.Dataset):
    def __init__(self, train_val_test, device='cuda', **kwargs):
        self.train_val_test = train_val_test
        self.device = device
        self.offsets = [0, 0, 0]
        self.scaling_function = scaling_minmax

        # parameters to deform moving-fixed images if synthetic
        self.augmenter = kwargs.pop('augmenter', None)
        self.augment_smod = False if self.augmenter is None else True

    @staticmethod
    def read_image_np(path):
        image_np = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype('float32')
        return image_np

    @staticmethod
    def read_image(path):
        image = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(path)).astype('float32'))
        return image

    @staticmethod
    def get_image_header(path):
        image = sitk.ReadImage(path)
        dim = np.array(image.GetSize())
        voxel_sp = np.array(image.GetSpacing())
        return dim[::-1], voxel_sp[::-1]

    def pad_torch(self, array, shape, pad_value):
        """
        Pads an array with the given padding value to a given shape. Returns the padded_array array and crop slices.
        """
        if array.squeeze().shape == tuple(shape):
            return array, ...
        array = array.squeeze().numpy()
        padded_array = pad_value * np.ones(shape, dtype=array.dtype)
        offsets = self.offsets
        try:
            slices = tuple([slice(offset, l + offset) for offset, l in zip(offsets, array.shape)])
            padded_array[slices] = array
        except:
            slices = tuple([slice(-1*offset, l + offset) for offset, l in zip(offsets, array.shape)])
            padded_array = array[slices]
        padded_array = torch.from_numpy(padded_array).unsqueeze(0)
        return padded_array, slices

    def adjust_shape(self, new_shape):
        self.offsets = [(shp - old_shp)//2 for (shp, old_shp) in zip(new_shape, self.shape)]
        self.shape = new_shape

    def get_shape(self):
        return self.shape

    def moving_fixed_paths(self, i):
        """Get the path to images and labels"""
        # Load in the moving- and fixed image/label
        moving_path, fixed_path = self.moving_img_path[i], self.fixed_img_path[i]
        return moving_path, fixed_path

    def subset(self, rand_indices):
        temp = [self.moving_img_path[i] for i in rand_indices]
        self.moving_img_path = temp
        temp = [self.fixed_img_path[i] for i in rand_indices]
        self.fixed_img_path = temp
        temp = [self.moving_lbl_path[i] for i in rand_indices]
        self.moving_lbl_path = temp
        temp = [self.fixed_lbl_path[i] for i in rand_indices]
        self.fixed_lbl_path = temp

    def __len__(self):
        return len(self.fixed_img_path)

    def __getitem__(self, idx):
        # Get image paths and load images
        moving_path, fixed_path = self.moving_fixed_paths(idx)
        moving_t = self.read_image(moving_path).unsqueeze(0)
        fixed_t = self.read_image(fixed_path).unsqueeze(0)
        return moving_t, fixed_t