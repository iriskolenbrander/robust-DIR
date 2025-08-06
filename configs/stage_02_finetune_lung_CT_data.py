import argparse
import os
from datetime import datetime

import lightning.pytorch as pl
import ml_collections
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import ConcatDataset
from lightning import seed_everything
from lightning.pytorch.loggers import WandbLogger

from configs.config_paths import ROOT_OUTPUT, ROOT_CHECKPOINTS, ROOT_DATA_LUNG_CT
from datasets.dataset_NLST import DatasetNLST
from datasets.statistical_model_of_deformations import AugmentationSMOD
from models.init_model import init_model
from utils.utils import set_seed

torch.backends.cudnn.benchmark = True


def parse_arguments():
    parser = argparse.ArgumentParser(description='Trainer - Finetuning or from-scratch training on lung CT data')
    parser.add_argument('--resume_id', type=str, metavar='', default='None', help='wandb run ID')
    parser.add_argument('--load_model', type=str, metavar='', default='None', help='')

    # Training configuration
    parser.add_argument('--seg_loss', type=str, metavar='', default='dice', help='Loss term: dice or mse')
    parser.add_argument('-lr', '--learning_rate', type=float, metavar='', default=1e-4, help='Learning rate')
    parser.add_argument("-rw", '--reg_weight', type=float, metavar='', default=0.75,
                        help='Regularization (smoothing) weight')
    parser.add_argument('-ep', '--epochs', type=int, metavar='', default=100, help='Nr of training epochs')
    parser.add_argument('-bs', '--batch_size', type=int, metavar='', default=1, help='Batch size')

    # Set remaining
    parser.add_argument('-seed', '--random_seed', type=int, metavar='', default=1000, help='random seed')
    parser.add_argument('-dev', '--device', type=str, metavar='', default='cuda', help='device / gpu used')
    parser.add_argument('-wandb', '--mode_wandb', type=str, metavar='', default='online',
                        help='online", "offline" or "disabled')

    config = parser.parse_args()

    config.root_output = ROOT_OUTPUT
    config.root_checkpoints = ROOT_CHECKPOINTS
    config.root_data = ROOT_DATA_LUNG_CT

    config.imgsize = (192, 192, 192)
    config = ml_collections.ConfigDict(dict(**vars(config)))
    print(config)

    # Set seed
    seed_everything(config.random_seed)
    set_seed(config.random_seed)
    return config


if __name__ == '__main__':
    now = datetime.now()
    date_time = now.strftime("%m%d-%H%M%S")
    config = parse_arguments()

    """ INIT DATA """
    augmentation_smod = AugmentationSMOD(inshape=(192, 192, 192),
                                         sigma=[4000, 6000],
                                         grid_size_resp=(23, 23, 23),
                                         components=os.path.join(config.root_data, 'SMOD/DVF_components'))
    dataset_config = dict(augmenter=augmentation_smod)

    train_dataset = DatasetNLST(train_val_test='train',
                                root_data=config.root_data,
                                device=config.device,
                                **dataset_config)
    val_dataset = DatasetNLST(train_val_test='val',
                              root_data=config.root_data,
                              device=config.device)
    config.metrics_dict = dict(dice=True, tre=True,
                               max_label=2 )

    config.voxel_spacing = train_dataset.voxel_spacing
    train_dataset.adjust_shape((192, 192, 192))
    val_dataset.adjust_shape((192, 192, 192))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=False)

    """ INIT MODEL AND LOGGER """
    lightning_model = init_model(config)
    lightning_model.train()
    if config.resume_id == 'None':
        config.run_name = date_time
        wandb_logger = WandbLogger(
            project="Robust-DIR",
            name=config.run_name,
            mode=config.mode_wandb,
            config=dict(**vars(config))['_fields'],
            save_dir=config.root_checkpoints,
            log_model=False,
        )
    else:
        wandb_logger = WandbLogger(
            project="Robust-DIR",
            mode=config.mode_wandb,
            config=dict(**vars(config))['_fields'],
            save_dir=config.root_checkpoints,
            log_model=False,
            version=config.resume_id
        )
        config.run_name = wandb_logger.experiment.name

    checkpoint_callback = ModelCheckpoint(monitor="val-dice", mode="max",
                                          dirpath=os.path.join(config.root_checkpoints, config.run_name),
                                          filename="{epoch:05d}",
                                          every_n_epochs=50,
                                          save_last=True,
                                          save_top_k=-1)

    """ TRAIN MODEL """
    trainer = pl.Trainer(max_epochs=config.epochs,
                         logger=[wandb_logger], callbacks=[checkpoint_callback],
                         devices=-1, accelerator=config.device,
                         check_val_every_n_epoch=5)
    if config.resume_id == 'None':
        trainer.fit(lightning_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    else:
        checkpoint_path = f"{config.root_checkpoints}/{config.run_name}/last.ckpt"
        trainer.fit(lightning_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
                    ckpt_path=checkpoint_path)