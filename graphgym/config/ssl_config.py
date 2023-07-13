from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('ssl')
def set_cfg_ssl(cfg):
    r'''
    Configuration for SSL training using DIG.
    '''

    cfg.ssl = CN()

    # ----------------------------------------------------------------------- #
    # GNN encoder configs
    # ----------------------------------------------------------------------- #
    cfg.ssl.encoder = CN()

    # GNN type (gin, gcn, resgcn)
    cfg.ssl.encoder.type = "gin"

    # Hidden dimension of the GNN
    cfg.ssl.encoder.dim_inner = 128

    # Nuber of message passing layers in the GNN
    cfg.ssl.encoder.layers_mp = 5

    # Whether to use batch normalization at the end of each MP layer
    cfg.ssl.encoder.batchnorm = True

    # Graph pooling method
    cfg.ssl.encoder.agg = "sum"

    # ----------------------------------------------------------------------- #
    # SSL training configs
    # ----------------------------------------------------------------------- #
    cfg.ssl.model = CN()

    # SSL training method
    cfg.ssl.model.type = "GraphCL"

    # Augmentation for the first view
    cfg.ssl.model.aug1 = "none"

    # Augmentation for the second view
    cfg.ssl.model.aug2 = "GraphCL"

    # Agumentation ratio, e.g., percent of nodes to drop in the case of 'dropN'
    cfg.ssl.model.aug_ratio = 0.2

    # Temperature parameter for InfoNCE
    cfg.ssl.model.tau = 0.2

    # ----------------------------------------------------------------------- #
    # Trainer configs
    # ----------------------------------------------------------------------- #
    cfg.ssl.trainer = CN()

    # Training batch size
    cfg.ssl.trainer.batch_size = 256

    # Optimizer (only Adam supported by DIG; otherwise need to pass as callable)
    cfg.ssl.trainer.optimizer = "Adam"

    # Learning rate
    cfg.ssl.trainer.lr = 0.01

    # Weight decay
    cfg.ssl.trainer.weight_decay = 0.0

    # Max number of training epochs
    cfg.ssl.trainer.max_epoch = 20
