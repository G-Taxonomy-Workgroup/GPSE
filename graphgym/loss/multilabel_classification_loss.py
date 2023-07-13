import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


@register_loss('multilabel_cross_entropy')
def multilabel_cross_entropy(pred, true):
    """Multilabel cross-entropy loss.
    """
    # NOTE: use 'multilabel' in the case of 2-class multi-label classification
    # to prevent overwritten by GraphGym into a single class binary
    # classification task
    # https://github.com/pyg-team/pytorch_geometric/blob/8b37ad571b6e08d700f344cd9965724939f4bd4c/torch_geometric/graphgym/model_builder.py#L77-L78
    if cfg.dataset.task_type in ['classification_multilabel', 'multilabel']:
        if cfg.model.loss_fun != 'cross_entropy':
            raise ValueError("Only 'cross_entropy' loss_fun supported with "
                             "'classification_multilabel' task_type.")
        bce_loss = nn.BCEWithLogitsLoss()
        is_labeled = true == true  # Filter our nans.
        return bce_loss(pred[is_labeled], true[is_labeled].float()), pred
