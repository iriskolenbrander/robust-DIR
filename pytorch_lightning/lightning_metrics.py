import numpy as np
import scipy
import torch
from torchmetrics import Metric
import SimpleITK as sitk
from functools import partial
from SimpleITK import GetArrayViewFromImage as ArrayView
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric


class MyFolding(Metric):
    def __init__(self):
        super(MyFolding, self).__init__()
        self.add_state("folding", default=torch.tensor(0.), dist_reduce_fx="mean")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="mean")

    def jacobian(self, displacement_field):
        D_x = (displacement_field[:, :, 1:, :-1, :-1] - displacement_field[:, :, :-1, :-1, :-1])
        D_y = (displacement_field[:, :, :-1, 1:, :-1] - displacement_field[:, :, :-1, :-1, :-1])
        D_z = (displacement_field[:, :, :-1, :-1, 1:] - displacement_field[:, :, :-1, :-1, :-1])

        D1 = (D_x[:, 0, ...] + 1) * ((D_y[:, 1, ...] + 1) * (D_z[:, 2, ...] + 1) - D_z[:, 1, ...] * D_y[:, 2, ...])
        D2 = (D_x[:, 1, ...]) * (D_y[:, 0, ...] * (D_z[:, 2, ...] + 1) - D_y[:, 2, ...] * D_z[:, 0, ...])
        D3 = (D_x[:, 2, ...]) * (D_y[:, 0, ...] * D_z[:, 1, ...] - (D_y[:, 1, ...] + 1) * D_z[:, 0, ...])

        det = D1 - D2 + D3
        return det

    def update(self, displacement_field):
        det = self.jacobian(displacement_field)
        self.folding += 100 * (det <= 0).sum() / torch.prod(torch.tensor(det.shape))
        self.total += 1

    def compute(self):
        return self.folding.float() / self.total.float()


class MyFoldingSTD(Metric):
    def __init__(self):
        super(MyFoldingSTD, self).__init__()
        self.add_state("folding", default=torch.tensor(0.), dist_reduce_fx="mean")
        self.add_state("std_jacdet", default=torch.tensor(0.), dist_reduce_fx="mean")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="mean")

    def jacobian(self, displacement_field):
        D_x = (displacement_field[:, :, 1:, :-1, :-1] - displacement_field[:, :, :-1, :-1, :-1])
        D_y = (displacement_field[:, :, :-1, 1:, :-1] - displacement_field[:, :, :-1, :-1, :-1])
        D_z = (displacement_field[:, :, :-1, :-1, 1:] - displacement_field[:, :, :-1, :-1, :-1])

        D1 = (D_x[:, 0, ...] + 1) * ((D_y[:, 1, ...] + 1) * (D_z[:, 2, ...] + 1) - D_z[:, 1, ...] * D_y[:, 2, ...])
        D2 = (D_x[:, 1, ...]) * (D_y[:, 0, ...] * (D_z[:, 2, ...] + 1) - D_y[:, 2, ...] * D_z[:, 0, ...])
        D3 = (D_x[:, 2, ...]) * (D_y[:, 0, ...] * D_z[:, 1, ...] - (D_y[:, 1, ...] + 1) * D_z[:, 0, ...])

        det = D1 - D2 + D3
        return det

    def update(self, displacement_field):
        det = self.jacobian(displacement_field)
        self.folding += 100 * (det <= 0).sum() / torch.prod(torch.tensor(det.shape))
        self.std_jacdet += torch.std(torch.log(torch.clamp(det.flatten(start_dim=1, end_dim=-1),
                                                           min=0.000000001, max=1000000000)),
                                     dim=-1).nanmean()
        self.total += 1

    def compute(self):
        return self.folding.float() / self.total.float(), self.std_jacdet.float() / self.total.float()

class MyTargetRegistrationError(Metric):
    def __init__(self, voxel_spacing):
        super(MyTargetRegistrationError, self).__init__()
        self.voxel_spacing = voxel_spacing.copy()
        self.add_state("tre", default=torch.tensor(0.), dist_reduce_fx="mean")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, points_1, points_2):
        element_wise_difference = (points_1 - points_2) * torch.tensor(self.voxel_spacing).to(points_1.device)
        tre = torch.mean(torch.sqrt(torch.nansum(element_wise_difference ** 2, -1)))
        self.tre += tre
        self.total += 1

    def compute(self):
        return self.tre.float() / self.total.float()

class MyDiceMultiLabel(Metric):
    def __init__(self, max_label):
        super(MyDiceMultiLabel, self).__init__()
        self.max_label = max_label
        self.add_state("dice", default=torch.zeros((1, max_label + 1)), dist_reduce_fx="mean")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, prediction, ground_truth):
        epsilon = 1E-5
        dsc_multilabel = torch.zeros((self.max_label+1)).to(prediction.device)
        for class_num in range(1, self.max_label + 1):
            # Turn the prediction and ground truth into one-hot encoded Tensors with 1's at locations with a value class_num
            pred_one_hot = (prediction == class_num).type(torch.uint8).reshape(-1)
            gt_one_hot = (ground_truth == class_num).type(torch.uint8).reshape(-1)
            # Calculate the DSC per class and add it to the list
            overlap = (pred_one_hot * gt_one_hot).sum()
            dsc = (2. * overlap ) / (pred_one_hot.sum() + gt_one_hot.sum() + epsilon)
            dsc_multilabel[class_num] = dsc.item()
        self.dice += dsc_multilabel
        self.total += 1

    def compute(self):
        return self.dice.float() / self.total.float()

class MyHausdorffMultiLabel(Metric):
    def __init__(self, max_label, voxel_spacing):
        super(MyHausdorffMultiLabel, self).__init__()
        self.max_label = max_label
        self.voxel_spacing = voxel_spacing.copy()
        self.add_state("hd", default=torch.zeros((1, max_label + 1)), dist_reduce_fx="mean")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, prediction, ground_truth):
        hd_multilabel = torch.zeros((self.max_label+1)).to(prediction.device)
        con_sitk_1 = sitk.GetImageFromArray(prediction.squeeze(0).squeeze(0))
        con_sitk_1.SetSpacing(self.voxel_spacing)
        con_sitk_2 = sitk.GetImageFromArray(ground_truth.squeeze(0).squeeze(0))
        con_sitk_2.SetSpacing(self.voxel_spacing)

        for class_num in range(1, self.max_label+1):
            surface_1 = sitk.LabelContour(con_sitk_1 == class_num, False)
            surface_2 = sitk.LabelContour(con_sitk_2 == class_num, False)
            distance_map = partial(sitk.SignedMaurerDistanceMap, squaredDistance=False, useImageSpacing=True)

            # Get distance map for contours (the distance map computes the minimum distances)
            distance_map_1 = sitk.Abs(distance_map(surface_1))
            distance_map_2 = sitk.Abs(distance_map(surface_2))

            # Find the distances to surface points of the contour.  Calculate in both directions
            one_to_2 = ArrayView(distance_map_1)[ArrayView(distance_map_2) == 0]
            two_to_1 = ArrayView(distance_map_2)[ArrayView(distance_map_1) == 0]

            # Find the 95% Distance for each direction and average
            try:
                # hd_multilabel[class_num - 1] = (np.percentile(two_to_1, 95) + np.percentile(one_to_2, 95)).item() / 2.0
                hd_multilabel[class_num] = np.percentile(np.concatenate((np.array(two_to_1), np.array(one_to_2)), axis=0), 95).item()
            except:
                hd_multilabel[class_num] = np.nan

        self.hd += hd_multilabel.to(self.device)
        self.total += 1

    def compute(self):
        return self.hd.float() / self.total.float()


class MyHausdorffMultiLabelMonai(Metric):
    def __init__(self, max_label, voxel_spacing):
        super(MyHausdorffMultiLabelMonai, self).__init__()
        self.max_label = max_label
        self.voxel_spacing = voxel_spacing.copy()
        self.add_state("hd", default=torch.zeros((1, max_label + 1)), dist_reduce_fx="mean")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="mean")

        # MONAI HD metric instance
        self.hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95)

    def update(self, prediction, ground_truth):
        hd_multilabel = torch.zeros((self.max_label+1)).to(prediction.device)
        for class_num in range(1, self.max_label+1):
            surface_1 = prediction == class_num
            surface_2 = ground_truth == class_num

            # Find the 95% Distance for each direction and average
            try:
                hd_multilabel[class_num] = self.hd_metric(surface_1, surface_2, **{'spacing': self.voxel_spacing}).item()
            except:
                hd_multilabel[class_num] = np.nan

        self.hd += hd_multilabel.to(self.device)
        self.total += 1

    def compute(self):
        return self.hd.float() / self.total.float()


class MyMeanSurfaceDistanceMonai(Metric):
    def __init__(self, max_label, voxel_spacing):
        super(MyMeanSurfaceDistanceMonai, self).__init__()
        self.max_label = max_label
        self.voxel_spacing = voxel_spacing.copy()
        self.add_state("msd", default=torch.zeros((1, max_label + 1)), dist_reduce_fx="mean")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="mean")

        # MONAI HD metric instance
        self.msd_metric = SurfaceDistanceMetric(symmetric=True)

    def update(self, prediction, ground_truth):
        msd_multilabel = torch.zeros((self.max_label+1)).to(prediction.device)
        for class_num in range(1, self.max_label+1):
            surface_1 = prediction == class_num
            surface_2 = ground_truth == class_num

            try:
                msd_multilabel[class_num] = self.msd_metric(surface_1, surface_2, **{'spacing': self.voxel_spacing}).item()
            except:
                msd_multilabel[class_num] = np.nan

        self.msd += msd_multilabel.to(self.device)
        self.total += 1

    def compute(self):
        return self.msd.float() / self.total.float()