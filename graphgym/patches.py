from torch_geometric.graphgym.config import assert_cfg, cfg
from torch_geometric.loader import (
    ClusterLoader,
    DataLoader,
    GraphSAINTEdgeSampler,
    GraphSAINTNodeSampler,
    GraphSAINTRandomWalkSampler,
    NeighborLoader,
    RandomNodeLoader,
)

from torch_geometric.graphgym.loader import create_dataset


def get_loader(dataset, sampler, batch_size, node_split_name, shuffle=True):
    if sampler == "full_batch" or len(dataset) > 1:
        loader_train = DataLoader(dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=cfg.num_workers,
                                  pin_memory=True,
                                  persistent_workers=cfg.num_workers > 0)
    elif sampler == "neighbor":
        assert node_split_name, "NeighborLoader is only valid for node tasks"
        loader_train = NeighborLoader(
            dataset[0],
            num_neighbors=cfg.train.neighbor_sizes[:cfg.gnn.layers_mp],
            batch_size=batch_size, shuffle=shuffle,
            num_workers=cfg.num_workers, pin_memory=True,
            input_nodes=getattr(dataset[0], f"{node_split_name}_mask"))
    elif sampler == "random_node":
        loader_train = RandomNodeLoader(dataset[0],
                                        num_parts=cfg.train.train_parts,
                                        shuffle=shuffle,
                                        num_workers=cfg.num_workers,
                                        pin_memory=True)
    elif sampler == "saint_rw":
        loader_train = \
            GraphSAINTRandomWalkSampler(dataset[0],
                                        batch_size=batch_size,
                                        walk_length=cfg.train.walk_length,
                                        num_steps=cfg.train.iter_per_epoch,
                                        sample_coverage=0,
                                        shuffle=shuffle,
                                        num_workers=cfg.num_workers,
                                        pin_memory=True)
    elif sampler == "saint_node":
        loader_train = \
            GraphSAINTNodeSampler(dataset[0], batch_size=batch_size,
                                  num_steps=cfg.train.iter_per_epoch,
                                  sample_coverage=0, shuffle=shuffle,
                                  num_workers=cfg.num_workers,
                                  pin_memory=True)
    elif sampler == "saint_edge":
        loader_train = \
            GraphSAINTEdgeSampler(dataset[0], batch_size=batch_size,
                                  num_steps=cfg.train.iter_per_epoch,
                                  sample_coverage=0, shuffle=shuffle,
                                  num_workers=cfg.num_workers,
                                  pin_memory=True)
    elif sampler == "cluster":
        loader_train = \
            ClusterLoader(dataset[0],
                          num_parts=cfg.train.train_parts,
                          save_dir="{}/{}".format(cfg.dataset.dir,
                                                  cfg.dataset.name.replace(
                                                      "-", "_")),
                          batch_size=batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers,
                          pin_memory=True)

    else:
        raise NotImplementedError("%s sampler is not implemented!" % sampler)
    return loader_train


def create_loader():
    """
    Create data loader object

    Returns: List of PyTorch data loaders

    """
    dataset = create_dataset()
    # train loader
    if cfg.dataset.task == 'graph':
        id = dataset.data['train_graph_index']
        loaders = [
            get_loader(dataset[id], cfg.train.sampler, cfg.train.batch_size,
                       node_split_name=None, shuffle=True)
        ]
        delattr(dataset.data, 'train_graph_index')
    else:
        loaders = [
            get_loader(dataset, cfg.train.sampler, cfg.train.batch_size,
                       node_split_name="train", shuffle=True)
        ]

    # val and test loaders
    for i in range(cfg.share.num_splits - 1):
        if cfg.dataset.task == 'graph':
            split_names = ['val_graph_index', 'test_graph_index']
            id = dataset.data[split_names[i]]
            loaders.append(
                get_loader(dataset[id], cfg.val.sampler, cfg.train.batch_size,
                           node_split_name=None, shuffle=False))
            delattr(dataset.data, split_names[i])
        else:
            split_names = ['val', 'test']
            loaders.append(
                get_loader(dataset, cfg.val.sampler, cfg.train.batch_size,
                           node_split_name=split_names[i], shuffle=False))

    return loaders


def load_cfg(cfg, args):
    r"""
    Load configurations from file system and command line.

    This patch added the 'parser_drop_eq' option to enable compatibility with
    the wandb sweep, which specify cli args with equal sign. E.g., specifying
    'parser_drop_eq' turns 'param1=value1' to 'param1 value1', which can be
    readily parsed into the given graphgym cli parser.

    Note:
        The 'parser_drop_eq' MUST be specified as the first argument AFTER the
        predefined cli arguments, such as '--cfg'. Example:
        ``python main.py --cfg config.yaml parser_drop_eq param1=value1``

    Args:
        cfg (CfgNode): Configuration node
        args (ArgumentParser): Command argument parser

    """
    cfg.merge_from_file(args.cfg_file)
    if "parser_drop_eq" in args.opts:  # "param1=value1" -> "param1 value1"
        opts = []
        for opt in args.opts:
            if opt != "parser_drop_eq":
                opts += opt.split("=", 1)
    else:
        opts = args.opts
    cfg.merge_from_list(opts)
    assert_cfg(cfg)
