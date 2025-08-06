import torch

from models.STM import SpatialTransformer
from models.VoxelMorph import MultiStepVxM
from pytorch_lightning.lightning_model import MyLightningModuleWeakSupervision


def init_model(config):
    # Init model
    model = MultiStepVxM(inshape=config.imgsize,
                         int_steps=5,
                         nb_unet_features=[[128] * 4, [128] * 6])

    # Load pretrained model
    if config.load_model is None or config.load_model == 'None':
        print("No pretrained model loaded.")
    else:
        print(f"Loading pretrained model: {config.load_model}")
        checkpoint_path = f"{config.root_checkpoints}/{config.load_model}.ckpt"
        state_dict = torch.load(checkpoint_path, map_location=config.device)["state_dict"]
        state_dict_new = dict()
        for key in list(state_dict.keys()):
            state_dict_new[key.replace('model.', '', 1)] = state_dict.pop(key)  #
        try:
            model.load_state_dict(state_dict_new)
        except:
            model.stl_segmentation_maps = SpatialTransformer(model.img_size, mode='nearest')
            model.load_state_dict(state_dict_new)

    # Init pytorch lightning module
    lightning_model = MyLightningModuleWeakSupervision(config, model, **config.metrics_dict)
    return lightning_model