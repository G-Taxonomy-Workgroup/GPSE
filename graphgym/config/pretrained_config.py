from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_pretrained')
def set_cfg_pretrained(cfg):
    """Configuration options for loading a pretrained model.
    """

    # ----------------------------------------------------------------------- #
    # GPS pre-trained model configs (prediction heads are replaced)
    # ----------------------------------------------------------------------- #

    cfg.pretrained = CN()

    # Directory path to a saved experiment, if set, load the model from there
    # and fine-tune / run inference with it on a specified dataset.
    cfg.pretrained.dir = ""

    # Discard pretrained weights of the prediction head and reinitialize.
    cfg.pretrained.reset_prediction_head = True

    # Freeze the main pretrained 'body' of the model, learning only the new head
    cfg.pretrained.freeze_main = False

    # ----------------------------------------------------------------------- #
    # Generic pre-trained model with exact (even the prediction heads) configs.
    # This mode does not require explicit checking of the pre-trained model
    # config, and implicitly assumes that it is consistent with the new ones.
    # Useful for further training the model on a specific dataset.
    # ----------------------------------------------------------------------- #

    cfg.pretrained_exact = CN()

    # Directory path to the saved model, if set, load the model from there
    # and fine-tune on the current dataset.
    cfg.pretrained_exact.model_dir = ""
