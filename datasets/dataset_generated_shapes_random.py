import gc
import os

import numpy as np
import torch
import tqdm
from lightning import seed_everything
from matplotlib import pyplot as plt

from configs.config_paths import ROOT_DATA_SYNTHETIC
from datasets.dataset_generated import GeneratedDataset
from models.STM import SpatialTransformer
from models.VoxelMorph import VecInt, ResizeTransform
from utils.utils import set_seed, make_dir

def load_datasets(shape, root_data):
    augmentation_random = DeformationSynthMorph(inshape=shape)

    num_labels = 26
    intensities_m0 = np.array([0] + [25 / 225] * (num_labels - 1))
    intensities_m1 = np.array([0] + [1.0] * (num_labels - 1))
    intensities_s0 = np.array([0] + [5 / 225] * (num_labels - 1))
    intensities_s1 = np.array([25 / 225] * num_labels)

    config = {
             'shape': shape,
               'num_labels': 26,
              'scales': (32),
              'max_std': 1,
              'scales_svf': (32),
              'max_std_svf': 50,

              'intensities_m0': intensities_m0,
              'intensities_m1': intensities_m1,
              'intensities_s0': intensities_s0,
              'intensities_s1': intensities_s1,

              'augmenter': augmentation_random}

    dataset_train = GeneratedDatasetRandomGeometry(root_data=root_data,
                                                    train_val_test='train',
                                                    device='cpu',
                                                    **config)
    dataset_val = GeneratedDatasetRandomGeometry(root_data=root_data,
                                                    train_val_test='val',
                                                    device='cpu',
                                                    **config)

    return dataset_train, dataset_val


class GeneratedDatasetRandomGeometry(GeneratedDataset):
    def __init__(self,
                 root_data,
                 train_val_test,
                 device='cuda',
                 **kwargs):
        super(GeneratedDatasetRandomGeometry, self).__init__(root_data,
                                                             train_val_test,
                                                             device, **kwargs)
        self.label_generation = {'num_maps': kwargs.pop('num_maps', 100),
                                 'scales': kwargs.pop('scales', 32),
                                 'max_std': kwargs.pop('max_std', 1),
                                 'scales_svf': kwargs.pop('scales_svf', (32)),
                                 'max_std_svf': kwargs.pop('max_std_svf', 50),
                                 'shape': kwargs.pop('shape', [192] * 3)
                                 }

    @staticmethod
    def draw_perlin_torch(out_shape,
                          scales,
                          min_std=0,
                          max_std=1,
                          dtype=torch.float32):
        out_shape = np.asarray(out_shape, dtype=np.int32)
        if np.isscalar(scales):
            scales = [scales]

        out = torch.zeros(out_shape.tolist(), dtype=dtype)
        for scale in scales:
            sample_shape = np.ceil(out_shape[:-1] / scale)
            sample_shape = np.int32((*sample_shape, out_shape[-1]))

            # std = tf.random.uniform(
            #     shape=[], minval=min_std, maxval=max_std, dtype=dtype, seed=seed(),
            # )
            std = torch.rand([], dtype=dtype)
            std = (max_std - min_std) * std - min_std

            # gauss = tf.random.normal(sample_shape, stddev=std, dtype=dtype, seed=seed())
            gauss = torch.normal(mean=0, std=std, size=sample_shape.tolist())

            zoom = [o / s for o, s in zip(out_shape, sample_shape)]
            new_size = [int(z * shp) for z, shp in zip(zoom, gauss.shape)]
            new_size = new_size[:-1]
            out += gauss if scale == 1 else torch.nn.functional.interpolate(gauss.permute(3, 0, 1, 2).unsqueeze(0),
                                                                            size=new_size,
                                                                            mode='trilinear').squeeze().permute(1, 2, 3,
                                                                                                                0)
        return out

    @staticmethod
    def draw_perlin_torch_warp(out_shape,
                               scales,
                               min_std=0,
                               max_std=1,
                               dtype=torch.float32):
        '''
        Generate Perlin noise by drawing from Gaussian distributions at different
        resolutions, upsampling and summing. There are a couple of key differences
        between this function and the Neurite equivalent ne.utils.perlin_vol, which
        are not straightforwardly consolidated.

        Neurite function:
            (1) Iterates over scales in range(a, b) where a, b are input arguments.
            (2) Noise volumes are sampled at resolutions vol_shape / 2**scale.
            (3) Noise volumes are sampled uniformly in the interval [0, 1].
            (4) Volume weights are {1, 2, ...N} (normalized) where N is the number
                of scales, or sampled uniformly from [0, 1].

        This function:
            (1) Specific scales are passed as a list.
            (2) Noise volumes are sampled at resolutions vol_shape / scale.
            (3) Noise volumes are sampled normally, with SDs drawn uniformly from
                [min_std, max_std].

        Parameters:
            out_shape: List defining the output shape. In N-dimensional space, it
                should have N+1 elements, the last one being the feature dimension.
            scales: List of relative resolutions at which noise is sampled normally.
                A scale of 2 means half resolution relative to the output shape.
            min_std: Minimum standard deviation (SD) for drawing noise volumes.
            max_std: Maximum SD for drawing noise volumes.
            dtype: Output data type.
            seed: Integer for reproducible randomization. This may only have an
                effect if the function is wrapped in a Lambda layer.
        '''
        out_shape = np.asarray(out_shape, dtype=np.int32)
        if np.isscalar(scales):
            scales = [scales]
        out = torch.zeros(out_shape.tolist(), dtype=dtype)
        for scale in scales:
            sample_shape = np.ceil(out_shape[:-2] / scale)
            sample_shape = np.int32((*sample_shape, *out_shape[-2::]))

            # std = tf.random.uniform(
            #     shape=[], minval=min_std, maxval=max_std, dtype=dtype, seed=seed(),
            # )
            std = torch.rand([], dtype=dtype)
            std = (max_std - min_std) * std - min_std

            # gauss = tf.random.normal(sample_shape, stddev=std, dtype=dtype, seed=seed())
            gauss = torch.normal(mean=0, std=std, size=sample_shape.tolist())
            zoom = [o / s for o, s in zip(out_shape, sample_shape)]
            new_size = [int(z * shp) for z, shp in zip(zoom, gauss.shape)]
            new_size = new_size[:-2]
            out += gauss if scale == 1 else torch.nn.functional.interpolate(gauss.permute(4, 3, 0, 1, 2), size=new_size,
                                                                            mode='trilinear').permute(2, 3, 4, 1, 0)
        return out

    def generate_labels(self):
        shape = self.label_generation['shape']
        num_maps = self.label_generation['num_maps']
        scales = self.label_generation['scales']
        max_std = self.label_generation['max_std']
        scales_svf = self.label_generation['scales_svf']
        max_std_svf = self.label_generation['max_std_svf']

        stn = SpatialTransformer(shape=shape, unity=False)

        # Shape generation.
        if not os.path.exists(self.labels_path):
            label_maps = []
            for _ in tqdm.tqdm(range(num_maps)):
                im_torch = self.draw_perlin_torch(
                    out_shape=(*shape, self.num_labels),
                    scales=scales, max_std=max_std,
                )
                warp_torch = self.draw_perlin_torch_warp(
                    out_shape=(*shape, self.num_labels, len(shape)),
                    scales=scales_svf, max_std=max_std_svf,
                )
                im_torch_warped = torch.zeros(im_torch.shape)
                for i in range(self.num_labels):
                    svf = warp_torch.permute(4, 0, 1, 2, 3)[:, :, :, :, i]
                    im_temp = im_torch[:, :, :, i]
                    dvf = VecInt(inshape=shape, nsteps=5)(svf.unsqueeze(0)).squeeze(0)
                    im = stn.forward(src=im_temp.unsqueeze(0).unsqueeze(0), flow=dvf.unsqueeze(0)).squeeze()
                    im_torch_warped[:, :, :, i] = im

                lab_torch = torch.argmax(im_torch_warped, dim=-1)
                label_maps.append(np.uint8(lab_torch))

            make_dir(os.path.split(self.labels_path)[0])
            np.save(self.labels_path, label_maps)
        else:
            print('subject-level labels consist already: ')
            print(self.labels_path)

        label_maps_np = np.load(self.labels_path)
        self.label_maps = [torch.from_numpy(label_maps_np[i, :, :, :]) for i in range(label_maps_np.shape[0])]
        self.on_the_fly_label = False
        self.shape = self.label_maps[0].shape
        self.num_dim = len(self.shape)
        return self.labels_path


class DeformationSynthMorph():
    def __init__(self, inshape, **kwargs):
        self.inshape = inshape
        self.num_dim = len(inshape)
        self.vel_shape = (*(shp//2 for shp in inshape), self.num_dim)
        self.warp_res = kwargs.pop('warp_res', [8, 16, 32])
        self.min_std = kwargs.pop('warp_min_std', 0)
        self.max_std = kwargs.pop('warp_max_std', 3)
        self.vel_scale = np.asarray(self.warp_res) / 2
        self.vel_draw = lambda: self.draw_perlin_torch(
                                self.vel_shape, scales=self.vel_scale,
                                min_std=self.min_std,
                                max_std=self.max_std)
        self.transformer = SpatialTransformer(self.inshape)
        self.transformer_binary = SpatialTransformer(self.inshape, mode='nearest')

    @staticmethod
    def draw_perlin_torch(out_shape,
                          scales,
                          min_std=0,
                          max_std=3,
                          dtype=torch.float32):
        out_shape = np.asarray(out_shape, dtype=np.int32)
        if np.isscalar(scales):
            scales = [scales]

        out = torch.zeros(out_shape.tolist(), dtype=dtype)
        for scale in scales:
            sample_shape = np.ceil(out_shape[:-1] / scale)
            sample_shape = np.int32((*sample_shape, out_shape[-1]))

            std = torch.rand([], dtype=dtype)
            std = (max_std - min_std) * std - min_std

            gauss = torch.normal(mean=0, std=std, size=sample_shape.tolist())
            zoom = [o / s for o, s in zip(out_shape, sample_shape)]
            new_size = [int(z * shp) for z, shp in zip(zoom, gauss.shape)]
            new_size = new_size[:-1]
            out += gauss if scale == 1 else torch.nn.functional.interpolate(gauss.permute(3, 0, 1, 2).unsqueeze(0),
                                                                            size=new_size,
                                                                            mode='trilinear').squeeze().permute(1, 2, 3,
                                                                                                                0)
        return out

    def generate_DVF_moving_to_fixed(self, return_inverse=False):
        vel_field = self.vel_draw().type(torch.float32)
        vel_field = vel_field.permute(3, 0, 1, 2).unsqueeze(0)
        def_field = VecInt(inshape= self.vel_shape[:3], nsteps=5)(vel_field)
        def_field = ResizeTransform(0.5, ndims=3)(def_field).squeeze(0).permute(1, 2, 3, 0)

        if return_inverse:
            inverse_def_field = VecInt(inshape= self.vel_shape[:3], nsteps=5)(-vel_field)
            inverse_def_field = ResizeTransform(0.5, ndims=3)(inverse_def_field).squeeze(0).permute(1, 2, 3, 0)

        if return_inverse:
            return def_field.permute(3, 0, 1, 2).unsqueeze(0), inverse_def_field.permute(3, 0, 1, 2).unsqueeze(0)
        else:
            return def_field.permute(3, 0, 1, 2).unsqueeze(0)

if __name__ == '__main__':
    """ 
    With this script, you can generate your labels for synthetic data training. 
    When separate training and validation sets are created, make sure to set the
    random_seed to different values for each set 
    (ensuring that there are in fact different images in these sets).
    """
    gc.collect()
    show_intermediate = True
    num_labels = 26
    shape = [192, 192, 192]
    train_val_test = 'temp2'

    if train_val_test == 'train':
        set_seed(10)
        seed_everything(10)
        num_maps = 100
    elif train_val_test == 'val':
        set_seed(50)
        seed_everything(50)
        num_maps = 20
    elif train_val_test == 'temp2':
        set_seed(50)
        seed_everything(50)
        num_maps = 2

    dataset = load_dataset(shape, train_val_test, num_maps, root_data=ROOT_DATA_SYNTHETIC)
    dataset.generate_labels()

    if show_intermediate:
        moving_lbl_list, moving_img_list, fixed_lbl_list, dvf_list = list(), list(), list(), list()
        for batch_idx in range(2):
            moving_img, fixed_img, moving_lbl, fixed_lbl = dataset[batch_idx]

            fig, axs = plt.subplots(2, 2)
            for i, im in enumerate([moving_img, fixed_img]):
                axs[0, i].imshow(im.squeeze()[:, shape[1] // 2, :], cmap='gray', vmin=-1, vmax=1)
            for i, im in enumerate([moving_lbl, fixed_lbl]):
                axs[1, i].imshow(im.squeeze()[:, shape[1] // 2, :],
                                 interpolation="nearest", cmap='tab20c')
            for ax in axs.flat:
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                plt.setp(ax.spines.values(), visible=False)
            fig.tight_layout()
            fig.show()