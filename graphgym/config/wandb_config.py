from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_wandb')
def set_cfg_wandb(cfg):
    """Weights & Biases tracker configuration.
    """

    # WandB group
    cfg.wandb = CN()

    # Use wandb or not
    cfg.wandb.use = False

    # Init for sweep (skip args like name and use those defined by the sweep agent)
    cfg.wandb.sweep_mode = False

    # Group to use for the sweep
    cfg.wandb.sweep_group = None

    # Wandb entity name, should exist beforehand
    cfg.wandb.entity = "gtransformers"

    # Wandb project name, will be created in your team if doesn't exist already
    cfg.wandb.project = "gtblueprint"

    # Optional run name
    cfg.wandb.name = ""
