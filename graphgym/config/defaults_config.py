from torch_geometric.graphgym.register import register_config


@register_config('overwrite_defaults')
def overwrite_defaults_cfg(cfg):
    """Overwrite the default config values that are first set by GraphGym in
    torch_geometric.graphgym.config.set_cfg

    WARNING: At the time of writing, the order in which custom config-setting
    functions like this one are executed is random; see the referenced `set_cfg`
    Therefore never reset here config options that are custom added, only change
    those that exist in core GraphGym.
    """

    # Training (and validation) pipeline mode
    cfg.train.mode = 'custom'  # 'standard' uses PyTorch-Lightning since PyG 2.1

    # Overwrite default dataset name
    cfg.dataset.name = 'none'

    # Overwrite default rounding precision
    cfg.round = 5

    # Default null dim in and out, but allow explicit setting of dims
    cfg.dim_in = None
    cfg.dim_out = None


@register_config('extended_cfg')
def extended_cfg(cfg):
    """General extended config options.
    """

    # Additional name tag used in `run_dir` and `wandb_name` auto generation.
    cfg.name_tag = ""

    # In training, if True (and also cfg.train.enable_ckpt is True) then
    # always checkpoint the current best model based on validation performance,
    # instead, when False, follow cfg.train.eval_period checkpointing frequency.
    cfg.train.ckpt_best = False

    # If set to some string value indicating the output directory, then dump
    # the test splits true/pred/batch_idx values under that directory, named
    # true.pt, pred.pt, and bidx.pt, respectively
    cfg.train.test_dump_ckpt_path = None
    cfg.train.test_dump_period = 10  # dump every x number of epochs

    # Record individual task scores in addition to the averaged scores if set
    # to True. Currently only support for regression tasks.
    cfg.train.record_individual_scores = False

    # What device to use for TorchMetrics. "default" means use the same value
    # as cfg.accelerator.
    cfg.val.accelerator = "default"

    # Number of node and graph targets, initialized with null value of -1.
    # Used by hybrid prediction head to set the number of node and graph heads.
    cfg.share.num_node_targets = -1
    cfg.share.num_graph_targets = -1
