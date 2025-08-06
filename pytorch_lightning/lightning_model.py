import numpy as np
import torch
from monai.losses import GlobalMutualInformationLoss, DiceLoss
from monai.losses.dice import one_hot
from monai.metrics import SSIMMetric
from torch.optim.lr_scheduler import LinearLR

import lightning.pytorch as pl

from pytorch_lightning.lightning_metrics import MyFoldingSTD, MyFolding, MyTargetRegistrationError, MyDiceMultiLabel, \
    MyHausdorffMultiLabel, MyHausdorffMultiLabelMonai, MyMeanSurfaceDistanceMonai

class Grad:
    """
    N-D gradient loss.
    """
    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    @staticmethod
    def dvf_diff(input, dim):
        if dim == 0:
            diff = input[1:, :, :, :, :] - input[:-1, :, :, :, :]
        elif dim == 1:
            diff = input[:, 1:, :, :, :] - input[:, :-1, :, :, :]
        elif dim == 2:
            diff = input[:, :, 1:, :, :] - input[:, :, :-1, :, :]
        elif dim == 3:
            diff = input[:, :, :, 1:, :] - input[:, :, :, :-1, :]
        elif dim == 4:
            diff = input[:, :, :, :, 1:] - input[:, :, :, :, :-1]
        return diff

    def forward(self, dvf):
        dz = torch.abs(self.dvf_diff(dvf, dim=2))
        dy = torch.abs(self.dvf_diff(dvf, dim=3))
        dx = torch.abs(self.dvf_diff(dvf, dim=4))

        if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx
                dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0
        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class MyLightningModule(pl.LightningModule):
    def __init__(self, config, model, **kwargs):
        super(MyLightningModule, self).__init__()
        self.config = config
        self.model = model
        self.kwargs = kwargs
        self.max_label = kwargs.pop('max_label', 35)
        if 'dice' in kwargs:
            self.dice = MyDiceMultiLabel(max_label=self.max_label).to(self.config.device)
        if 'hausdorff' in kwargs:
            self.hausdorff = MyHausdorffMultiLabel(max_label=self.max_label,
                                                   voxel_spacing=config.voxel_spacing).to(self.config.device)
            self.hausdorff_monai = MyHausdorffMultiLabelMonai(max_label=self.max_label,
                                                   voxel_spacing=config.voxel_spacing).to(self.config.device)
        if 'msd' in kwargs:
            self.msd_monai = MyMeanSurfaceDistanceMonai(max_label=self.max_label,
                                                   voxel_spacing=config.voxel_spacing).to(self.config.device)


    def forward(self, source, target):
        return self.model.forward(source, target)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        return optimizer

    def val_test_step(self, batch, batch_idx, prefix, log=True, return_all=False):
        source, target, source_map, target_map = batch
        dvf, source_warped = self.forward(source, target)
        metrics_dict = dict()

        if 'dice' in self.kwargs:
            source_map_warped = self.model.stl_binary(source_map, dvf)
            dice = self.dice(source_map_warped, target_map)
            metrics_dict.update({f'{prefix}dice': dice.mean()})
            metrics_dict.update({f'{prefix}dice_median': dice.median()})

        if log:
            self.log_dict(metrics_dict, on_step=False, on_epoch=True, prog_bar=True)
        if return_all:
            return (source, target, source_warped), dvf, metrics_dict
        else:
            return

    def validation_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, prefix='val-')
        return

    def test_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, prefix='test-')
        return

class MyLightningModuleWeakSupervision(MyLightningModule):
    def __init__(self, config, model, **kwargs):
        super(MyLightningModuleWeakSupervision, self).__init__(config, model, **kwargs)
        self.reg_weight = config.reg_weight
        self.seg_weight = 1.0
        self.stl_segmentation_maps = model.stl

        if config.seg_loss == 'dice':
            self.seg_loss = DiceLoss(include_background=False,
                                     jaccard=True)
        elif config.seg_loss == 'mse':
            self.seg_loss = torch.nn.MSELoss(reduction='mean')
        self.smooth_loss = Grad(penalty='l2')

    def loss(self, source_map, target_map, dvf):
        loss_reg = self.reg_weight * self.smooth_loss.forward(dvf)
        loss_seg = self.seg_weight * self.seg_loss.forward(source_map, target_map)
        loss = loss_reg + loss_seg
        return loss

    def training_step(self, batch, batch_idx):
        source, target, source_map, target_map = batch
        num_classes = int(target_map.max()) + 1
        source_map = one_hot(source_map, num_classes=num_classes)
        target_map = one_hot(target_map, num_classes=num_classes)

        dvf, source_warped = self.forward(source, target)
        source_map_warped = self.stl_segmentation_maps(source_map, dvf)
        loss = self.loss(source_map_warped, target_map, dvf)
        self.log_dict({'train-loss': loss}, on_step=True, on_epoch=True, prog_bar=True)
        return loss
