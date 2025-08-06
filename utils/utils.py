import os

import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk

def make_dir(path):
    parent = os.path.split(path)[0]
    parent_parent = os.path.split(parent)[0]
    parent_parent_parent = os.path.split(parent_parent)[0]
    if not os.path.exists(parent_parent_parent):
        os.mkdir(parent_parent_parent)
    if not os.path.exists(parent_parent):
        os.mkdir(parent_parent)
    if not os.path.exists(parent):
        os.mkdir(parent)
    if not os.path.exists(path):
        os.mkdir(path)


def save_image(im, im_src_sitk, voxel_sp, image_path):
    # make sitk image
    im_sitk = sitk.GetImageFromArray(im)
    im_sitk.SetOrigin(im_src_sitk.GetOrigin())
    im_sitk.SetDirection(im_src_sitk.GetDirection())
    im_sitk.SetSpacing(voxel_sp)

    # write sitk image to file
    make_dir(os.path.split(image_path)[0])
    sitk.WriteImage(im_sitk, image_path)


def to_numpy(im):
    return im.squeeze().cpu().numpy()


def set_seed(seed_value, pytorch=True):
    """
    Set seed for deterministic behavior

    Parameters
    ----------
    seed_value : int
        Seed value.
    pytorch : bool
        Whether the torch seed should also be set. The default is True.

    Returns
    -------
    None.
    """
    import random
    random.seed(seed_value)
    np.random.seed(seed_value)
    if pytorch:
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True


def save_predictions(predictions, config):
    metrics = [pred[2] for pred in predictions]
    metrics_df = pd.DataFrame(metrics)
    metrics_df['dataset'] = config.dataset
    metrics_baseline = [pred[3] for pred in predictions]
    metrics_df_baseline = pd.DataFrame(metrics_baseline)
    metrics_df_baseline['dataset'] = config.dataset
    metrics_df = pd.concat([metrics_df_baseline, metrics_df], axis=1)
    csv_path = '{}/{}_{}_{}_{}.csv'.format(config.root_output,
                                           config.run_name,
                                           config.which,
                                           config.dataset,
                                           config.mode)
    metrics_df.to_csv(csv_path)
    metrics_df.to_pickle(csv_path.replace('.csv', '.pkl'))


def scaling_minmax(image):
    return ((image - image.min()) / (image.max() - image.min())) * 2 - 1
