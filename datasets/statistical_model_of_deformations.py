import os
from typing import Tuple

import pickle
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn import preprocessing

from models.STM import SpatialTransformer

plt.rcParams['image.cmap'] = 'gray'

# Bspline functions
def bspline_kernel_nd(t, order, dtype=float):
    tpowers = t ** torch.arange(order, 0 - 1, -1, dtype=dtype) # print(tpowers)
    if order == 1:
        return tpowers @ torch.tensor(((-1, 1), (1, 0)), dtype=dtype)
    elif order == 2:
        return (
            tpowers
            @ torch.tensor(((1, -2, 1), (-2, 2, 0), (1, 1, 0)), dtype=dtype)
            / 2.0
        )
    elif order == 3:
        return (
            tpowers
            @ torch.tensor(
                ((-1, 3, -3, 1), (3, -6, 3, 0), (-3, 0, 3, 0), (1, 4, 1, 0)),
                dtype=dtype,
            )
            / 6.0
        )


def bspline_convolution_kernel(upsampling_factors, order, dtype=float):
    ndim = len(upsampling_factors)
    for i, us_factor in enumerate(upsampling_factors):
        t = torch.linspace(1 - (1 / us_factor), 0, us_factor)
        ker1D = bspline_kernel_nd(t[:, None], order, dtype).T.flatten() #print(ker1D)
        shape = (1,) * i + ker1D.shape + (1,) * (ndim - 1 - i)
        try:
            kernel = kernel * ker1D.view(shape)
        except NameError:
            kernel = ker1D.view(shape)
    return kernel


class BsplineUpsampleBlock(torch.nn.Module):
    """
    third order bspline. upsampling via transposed convolutions.
    """

    def __init__(self, shape, upsampling_factors: Tuple[int], order: int = 3):
        super().__init__()
        self.upsampling_factors = upsampling_factors
        bspline_kernel = self.make_bspline_kernel(self.upsampling_factors, order=order)
        kernel_size = bspline_kernel.shape
        crop_size = tuple(int(el * 3 / 8) if el != 5 else 2 for el in kernel_size)
        upsampler = torch.nn.ConvTranspose3d(1, 1, kernel_size, stride=self.upsampling_factors, padding=crop_size, bias=False)
        upsampler.weight = torch.nn.Parameter(bspline_kernel[None, None], requires_grad=False)
        self.upsampler = upsampler
        self.output_shape = shape

    @staticmethod
    def make_bspline_kernel(upsampling_factors, order, dtype=torch.float32):
        # TODO: solve the issues with kernel shapes
        bspline_kernel = bspline_convolution_kernel(upsampling_factors, order=order, dtype=dtype)
        if (np.array(bspline_kernel.shape[::-1]) == 4).any() or (
                np.array(bspline_kernel.shape[::-1]) == 2
        ).any():  # hack to deal with 1 strides and kernel size of 4
            padding = list()
            for s in bspline_kernel.shape[::-1]:
                if s == 4 or s == 2:
                    padding.extend([1, 0])
                else:
                    padding.extend([0, 0])
            bspline_kernel = torch.nn.functional.pad(bspline_kernel, padding, mode="constant")
        return bspline_kernel

    def create_dvf(self, bspline_parameters, output_shape):
        shape = bspline_parameters.shape
        dvf = self.upsampler(
            bspline_parameters.view((shape[0] * 3, 1) + shape[2:]),
            output_size=output_shape,
        )
        newshape = dvf.shape
        return dvf.view((shape[0], 3) + newshape[2:])

    def forward(self, bspline_coefficients):
        dvf = self.create_dvf(bspline_coefficients, output_shape=self.output_shape)
        return dvf


class AugmentationDeformable():
    def __init__(self, inshape, max_deform=4):
        self.inshape = inshape
        self.max_deform = max_deform
        self.transformer = SpatialTransformer(self.inshape)
        self.transformer_binary = SpatialTransformer(self.inshape, mode='nearest') # .to(self.config.device)
        self.upsampler = torch.nn.Upsample(scale_factor=32, mode='trilinear')

    def rescale_DVF(self, DVF, reverse=False):
        """
        input:       DVF            torch tensor        containing the displacements in voxels
                    reverse         bool (optional)     When true the rescaling is reversed. From [-1, 1] to voxels.
        returns:    rescaled DVF    torch tensor        containing rescaled displacement between -1 and 1
        """
        delta = 2. / (torch.tensor(DVF.shape[:-1], dtype=DVF.dtype, device=DVF.device) - 1)
        if reverse:
            delta = 1. / delta
        return DVF * delta

    def generate_DVF_augment(self):
        grid = (2 * self.max_deform) * torch.rand([val // 32 for val in self.inshape] + [len(self.inshape)]) - self.max_deform
        DVF = self.upsampler(grid.permute(3, 0, 1, 2).unsqueeze(0))
        return DVF

    def generate_on_the_fly(self):
        DVF_augment = self.generate_DVF_augment()
        DVF_moving_to_fixed = self.generate_DVF_moving_to_fixed()
        return DVF_augment, DVF_moving_to_fixed

    def composed_transform(self, DVF1, DVF2):
        # Use this transformed grid to interpolated DVF1 (first transform)
        DVF1_transformed = self.transformer(src=DVF1, flow=DVF2)
        DVF = DVF1_transformed + DVF2
        return DVF

    def augment_full(self, image):
        DVF_aug, DVF_moving_to_fixed = self.generate_on_the_fly()
        DVF = self.composed_transform(DVF_aug, DVF_moving_to_fixed)
        augmented_image_mov = self.transformer(src=image,
                                           flow=DVF_aug)
        augmented_image_fix = self.transformer(src=image,
                                                flow=DVF)
        return augmented_image_mov, augmented_image_fix, (DVF_aug, DVF_moving_to_fixed, DVF)

class AugmentationSMOD(AugmentationDeformable):
    def __init__(self, inshape, components=None, sigma=[4000, 6000], grid_size_resp=(26, 23, 26)):
        super().__init__(inshape)
        self.inshape = inshape
        self.method = 'bspline'
        self.sigma_a = sigma[0]
        self.sigma_b = sigma[1]
        self.grid_size_resp = grid_size_resp
        if components is not None:
            components_mean = np.load(components + '_mean.npy', allow_pickle=True)
            components_Ud = np.load(components + '_variance.npy', allow_pickle=True)
            self.components_mean = torch.zeros((components_mean[0].size, 3))
            for i, array in enumerate(components_mean):
                self.components_mean[:, i] = torch.from_numpy(array)
            self.components_Ud = [torch.from_numpy(array) for array in components_Ud]
        else:
            self.components = None
        self.upsampling_factors = tuple([int(np.ceil(im_shp / (grd_shp - 3))) for im_shp, grd_shp in
                                         zip(inshape, grid_size_resp)])
        self.dvf_output_shape = [gridsize * factor for gridsize, factor in zip(grid_size_resp, self.upsampling_factors)]
        self.upsample_bspline = BsplineUpsampleBlock(self.dvf_output_shape, upsampling_factors=self.upsampling_factors)

    @staticmethod
    def plot_images_dvf(images, deformation_field, images_titles=['fixed', 'moving', 'warped'], dvf_ax=2):
        fig, axs = plt.subplots(3, len(images) + 1, figsize=((len(images) + 1) * 5, 15), dpi=80)
        plt.rc('font', size=20)
        for (j, im), title in zip(enumerate(images), images_titles):
            axs[0, j].imshow(im[im.shape[0] // 2, :, :])
            axs[0, j].set_title(title)
            axs[1, j].imshow(im[:, im.shape[1] // 2, :])
            axs[2, j].imshow(im[:, :, im.shape[2] // 2])

        shp = deformation_field.shape
        axs[0, len(images)].imshow(deformation_field[shp[0] // 2, :, :, dvf_ax], cmap='coolwarm')
        axs[1, len(images)].imshow(deformation_field[:, shp[1] // 2, :, dvf_ax], cmap='coolwarm')
        axs[2, len(images)].imshow(deformation_field[:, :, shp[2] // 2, dvf_ax], cmap='coolwarm')
        axs[0, len(images)].set_title('deformation (axis {})'.format(dvf_ax))

        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.show()

    # STEP 1 - registration of moving-fixed images in training set
    def registration(self, moving_image, fixed_image,
                     method, output_path):
        """Function that calls Elastix registration function
        """
        # Define parameter object
        import itk
        parameter_object = itk.ParameterObject.New()
        if os.path.isfile(method):
            parameter_object.AddParameterFile(method)  # e.g. Par007.txt
        else:
            parameter_object.SetParameterMap(parameter_object.GetDefaultParameterMap(method))

        # Registration
        warped_image, result_transform_parameters = itk.elastix_registration_method(
            fixed_image, moving_image,
            parameter_object=parameter_object,
            log_to_console=False, output_directory=output_path)

        # Deformation field
        grid_size = tuple([int(val) for val in list(result_transform_parameters.GetParameter(0, 'GridSize'))])
        bspline_params = result_transform_parameters.GetParameter(0, 'TransformParameters')
        bspline_params = [float(value) for value in bspline_params]
        bspline_params = np.array(bspline_params).reshape((3,) + grid_size).transpose(1, 2, 3, 0)

        deformation_field = bspline_params
        return [moving_image, fixed_image, warped_image], deformation_field, result_transform_parameters

    # STEP 2 - dimensionality reduction
    def dim_reduction(self, DVFs):
        """Apply PCA to get DVF values for expression of DVF generation
        Args:
            DVFs (list): list with DVF's either in shape (160, 128, 160, 3) or itk.itkImagePython.itkImageVF33
        Returns:
            DVF_mean (list with arrays): For each direction the mean of inputted DVFs
            DVF_Ud (list with arrays): For each direction the PCA components needed to generate the artificial DVFs
        """
        # Convert (160, 128, 160, 3) to (3276800, 3)
        deformation_grid_columns = [np.reshape(np.asarray(DVF), (-1, 3)) for DVF in DVFs]
        # Make empty lists for args to return
        num_components_list = []
        for i in range(3):
            # Select OndeDirection (x, y, or z) of displacement fields
            DVFs_OD = [DVF[:, i] for DVF in deformation_grid_columns]

            # Scale data
            standard_scalar = preprocessing.StandardScaler(with_mean=True, with_std=True)
            DVFs_OD = standard_scalar.fit_transform(DVFs_OD)

            # Fit PCA
            pca = PCA()
            pca.fit(DVFs_OD)
            # Determine the number of principal components for 90% variability
            explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
            num_components = np.argmax(explained_variance_ratio_cumsum >= 0.8) + 1
            num_components_list.append(num_components)
            # print(num_components, explained_variance_ratio_cumsum)

        num_components = max(num_components_list)
        deformation_grid_mean, deformation_grid_Ud = [], []
        print('number of componetns = {}'.format(num_components))
        for i in range(3):
            # Select OndeDirection (x, y, or z) of displacement fields
            DVFs_OD = [DVF[:, i] for DVF in deformation_grid_columns]
            # Average DVFs from one direction and add to list to return
            DVF_OD_mean = np.mean(DVFs_OD, axis=0)
            deformation_grid_mean.append(DVF_OD_mean)

            # Scale data
            standard_scalar = preprocessing.StandardScaler(with_mean=True, with_std=True)
            DVFs_OD = standard_scalar.fit_transform(DVFs_OD)

            # Fit PCA
            pca = PCA()
            pca.fit(DVFs_OD)
            # Calculate PCA variables to return
            U = pca.components_[:num_components, :]  # (2, 3276800)
            d = pca.explained_variance_[:num_components]  # (2,)
            # Scale eigenvectors with eigenvalues, that are scaled with the total amount of variance
            d_scaled = d / sum(pca.explained_variance_)  # sum_all=1
            Ud = U * d_scaled[:, np.newaxis]  # (p, 3276800)
            # Adding U*d to list with 3 dimensions
            deformation_grid_Ud.append(Ud)

        return deformation_grid_mean, deformation_grid_Ud

    # STEP 3 - generate DVF based on smod model
    def generate_DVF_moving_to_fixed(self):
        """Generate artificial DVFs from list with to_atlas_DVFs
        Args:
            DVFs_artificial_components (list with arrays): DVF_mean, DVF_Ud from dimreduction()
            sigma (int): random scaling component 100: visual deformations, 500: too much
            DVFs (list): list with DVF's either in shape (160, 128, 160, 3) or itk.itkImagePython.itkImageVF33
        Returns:
            DVFs_artificial (list with arrays): artificial DVFs with shape (160, 128, 160, 3)
        """
        # Unpack mean and PCA components from dimreduction()
        deformation_grid = torch.zeros((self.components_mean.shape))
        sigma = self.sigma_a + (self.sigma_b - self.sigma_a) * torch.rand(1)
        for j in range(3):
            # Paper: vg = Vmean + U*x*d
            x = (2 * sigma) * torch.rand(self.components_Ud[j].shape[0]).type(torch.float64) - sigma
            deformation_grid[:, j] = self.components_mean[:, j] + torch.matmul(self.components_Ud[j].T,
                                                                               x)  # (3276800,1)

        deformation_grid = deformation_grid.reshape(self.grid_size_resp + (3,))
        DVF_fullres = self.upsample_bspline.forward(
            deformation_grid.permute(3, 0, 1, 2).unsqueeze(0))
        diff_shapes = [shp_dvf - shp_im for shp_dvf, shp_im in zip(self.dvf_output_shape, self.inshape)]
        DVF_fullres = DVF_fullres[:, :,
                      diff_shapes[0] // 2:-diff_shapes[0] // 2,
                      diff_shapes[1] // 2:-diff_shapes[1] // 2,
                      diff_shapes[2] // 2:-diff_shapes[2] // 2]
        return DVF_fullres.flip([1])  # .to(self.config.device)

    def get_deformations(self, images_moving, images_fixed, method, output_path):
        dvf_file_path = os.path.join(output_path, 'DVF_list')
        if os.path.exists(dvf_file_path):
            with open(dvf_file_path, "rb") as fp:  # Unpickling
                DVFs_list = pickle.load(fp)
        else:
            #  register inhaled to exhaled with bspline to get the breathing motion
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            DVFs_list = []
            for f, m in zip(tqdm(images_fixed), images_moving):
                images, DVF, result_transform_parameters = self.registration(
                    moving_image=m,
                    fixed_image=f,
                    method=method,
                    output_path=output_path)
                DVFs_list.append(DVF)
            with open(dvf_file_path, "wb") as fp:  # Pickling
                pickle.dump(DVFs_list, fp)
        return DVFs_list

    def obtain_SMOD_model(self, images_moving, images_fixed, method, output_path):
        DVFs_list = self.get_deformations(images_moving, images_fixed, method, output_path)

        # generate artificial DVFs that model breathing motion from inhaled to exhaled
        DVF_mean, DVF_Ud = self.dim_reduction(DVFs=DVFs_list)
        np.save(os.path.join(output_path, 'DVF_components_mean.npy'), np.array(DVF_mean))
        np.save(os.path.join(output_path, 'DVF_components_variance.npy'), np.array(DVF_Ud))
        return DVF_mean, DVF_Ud

    def obtain_SMOD_model_subset(self, output_path, subset):
        dvf_file_path = os.path.join(output_path, 'DVF_list')
        if os.path.exists(dvf_file_path):
            with open(dvf_file_path, "rb") as fp:  # Unpickling
                DVFs_list = pickle.load(fp)

        temp = [DVFs_list[i] for i in subset]
        DVFs_list = temp

        if not os.path.exists(output_path +'_subset{}'.format(len(subset))):
            os.makedirs(output_path + '_subset{}'.format(len(subset)))

        # generate artificial DVFs that model breathing motion from inhaled to exhaled
        DVF_mean, DVF_Ud = self.dim_reduction(DVFs=DVFs_list)
        np.save(os.path.join(output_path + '_subset{}'.format(len(subset)), 'DVF_components_mean.npy'),
                np.array(DVF_mean))
        np.save(os.path.join(output_path + '_subset{}'.format(len(subset)), 'DVF_components_variance.npy'),
                np.array(DVF_Ud))
        return DVF_mean, DVF_Ud

def register(dataset, inshape, grid_size_resp):
    # dataset.subset(list(range(10)))
    images_moving, images_fixed = zip(
        *[(img_moving, img_fixed) for img_moving, img_fixed, _, _ in dataset])
    images_moving = [m.squeeze(0).numpy() for m in images_moving]
    images_fixed = [f.squeeze(0).numpy() for f in images_fixed]
    output_path = os.path.join(dataset.root_data, 'SMOD')
    # elastix_params = os.path.join(dataset.root_data, 'ELASTIX', 'Par0011.bspline2_ind.txt')
    elastix_params = 'bspline'
    print('Registering: obtaining DVFs for {} image pairs'.format(len(images_moving)))
    augmentation_smod = AugmentationSMOD(inshape=inshape, grid_size_resp=grid_size_resp)
    augmentation_smod.obtain_SMOD_model(images_moving, images_fixed,
                                        method=elastix_params,
                                        output_path=output_path)