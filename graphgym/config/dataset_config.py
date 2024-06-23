from torch_geometric.graphgym.register import register_config


@register_config('dataset_cfg')
def dataset_cfg(cfg):
    """Dataset-specific config options.
    """

    # Input/output node encoder types (used to construct PE tasks)
    # Use "+" to cnocatenate multiple encoder types, e.g. "LapPE+RWSE"
    cfg.dataset.input_node_encoders = "none"
    cfg.dataset.output_node_encoders = "none"
    cfg.dataset.output_graph_encoders = "none"

    # If set to True, then combine the output encoders to
    # pestat_{output_node_encoders}+{output_graph_encoders}
    cfg.dataset.combine_output_pestat = False

    # The number of node types to expect in TypeDictNodeEncoder.
    cfg.dataset.node_encoder_num_types = 0

    # The number of edge types to expect in TypeDictEdgeEncoder.
    cfg.dataset.edge_encoder_num_types = 0

    # VOC/COCO Superpixels dataset version based on SLIC compactness parameter.
    cfg.dataset.slic_compactness = 10

    # Reduce the molecular graph dataset to only contain unique structured
    # graphs (ignoring atom and bond types)
    cfg.dataset.unique_mol_graphs = False
    cfg.dataset.umg_train_ratio = 0.8
    cfg.dataset.umg_val_ratio = 0.1
    cfg.dataset.umg_test_ratio = 0.1
    cfg.dataset.umg_random_seed = 0  # for random indexing
