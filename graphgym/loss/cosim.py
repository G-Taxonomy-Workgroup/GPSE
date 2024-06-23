import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss
from torch_geometric.utils import to_dense_batch


def cosim_col_sep(pred, true, bidx):
    bidx = bidx + 1 if bidx.min() == -1 else bidx
    pred_dense = to_dense_batch(pred, bidx)[0]
    true_dense = to_dense_batch(true, bidx)[0]
    mask = (true_dense == 0).all(1)  # exclude trivial features from loss
    loss = 1 - F.cosine_similarity(pred_dense, true_dense, dim=1)[~mask].mean()
    return loss


@register_loss("cosine_similarity_loss")
def cosim_losses(pred, true, batch_idx=None):
    if cfg.model.loss_fun in ["cosim_col", "cosim_row"]:
        dim = 1 if cfg.model.loss_fun.endswith("row") else 0
        loss = 1 - F.cosine_similarity(pred, true, dim=dim).mean()
        return loss, pred
    if cfg.model.loss_fun == "cosim_col_sep":  # separate by graph instances
        loss = cosim_col_sep(pred, true, batch_idx)
        return loss, pred


@register_loss("mae_cosine_similarity_loss")
def mae_cosim_losses(pred, true, batch_idx=None):
    if cfg.model.loss_fun == "mae_cosim_col":
        mae_loss = F.l1_loss(pred, true)
        cosim_loss = 1 - F.cosine_similarity(pred, true, dim=0).mean()
        loss = mae_loss + cosim_loss
        return loss, pred
    if cfg.model.loss_fun == "mae_cosim_col_sep":
        if batch_idx is None:
            raise ValueError("mae_cosim_col_sep requires batch index as "
                             "input to distinguish different graphs.")
        mae_loss = F.l1_loss(pred, true)
        cosim_loss = cosim_col_sep(pred, true, batch_idx)
        loss = mae_loss + cosim_loss
        return loss, pred
