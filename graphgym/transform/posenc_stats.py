from copy import deepcopy
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected, to_dense_adj)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add

from graphgym.transform.cycle_counts import count_cycles

EPS = 1e-6  # values below which we consider as zeros

PE_TYPES = [
    "ElstaticPE",
    "EquivStableLapPE",
    "HKdiagSE",
    "HKfullPE",
    "LapPE",
    "RWSE",
    "SignNet",
    "GPSE",
    "GraphLog",
    "CombinedPSE",
    "BernoulliRE",
    "NormalRE",
    "NormalFixedRE",
    "UniformRE",
]
RANDSE_TYPES = [
    "NormalSE",
    "UniformSE",
    "BernoulliSE",
]
GRAPH_ENC_TYPES = [
    "EigVals",
    "CycleGE",
    "RWGE",
]
ALL_ENC_TYPES = PE_TYPES + RANDSE_TYPES + GRAPH_ENC_TYPES


def _check_pe_types(pe_types: List[str]):
    wrong_types = []
    for t in pe_types:
        if t not in PE_TYPES:
            wrong_types.append(t)
    if wrong_types:
        raise ValueError(f"Unexpected PE selections {wrong_types} in {pe_types}")


def _check_randse_types(se_types: List[str]):
    wrong_types = []
    for t in se_types:
        if t not in RANDSE_TYPES:
            wrong_types.append(t)
    if wrong_types:
        raise ValueError(f"Unexpected RandomSE selections {wrong_types} in {se_types}")


def _check_all_types(se_types: List[str]):
    wrong_types = []
    for t in se_types:
        if t not in ALL_ENC_TYPES:
            wrong_types.append(t)
    if wrong_types:
        raise ValueError("Unexpected P/SE encoding selections: "
                         f"{wrong_types} in {se_types}")


def _combine_encs(
    data,
    in_node_encs: List[str],
    out_node_encs: List[str],
    out_graph_encs: List[str],
    cfg,
):
    combined_stats = []

    for name, pes in zip(["x", "y"], [in_node_encs, out_node_encs]):
        if pes == "none":
            continue

        _check_all_types(pe_types := pes.split("+"))
        pe_list: List[torch.Tensor] = []

        for pe_type in pe_types:
            if pe_type in ["LapPE", "EquivStableLapPE"]:
                if cfg[f"posenc_{pe_type}"].eigen.stack_eigval:
                    pe = torch.hstack((data.EigVecs, data.EigVals.squeeze(-1)))
                else:
                    pe = data.EigVecs
            elif pe_type == "SignNet":
                pe = torch.hstack((data.eigvecs_sn,
                                   data.eigvals_sn.squeeze(-1)))
            elif pe_type in RANDSE_TYPES:
                pe = getattr(data, name)
            else:
                pe = getattr(data, f"pestat_{pe_type}")
            pe_list.append(pe)

        combined_node_pe = torch.nan_to_num(torch.hstack(pe_list))

        if (name == "y") and cfg.dataset.combine_output_pestat:
            combined_stats.append(combined_node_pe)
        else:
            setattr(data, name, combined_node_pe)

    # Graph level encoding targets
    if out_graph_encs != "none":
        enc_list: List[torch.Tensor] = []
        _check_all_types(enc_types := out_graph_encs.split("+"))
        for enc_type in enc_types:
            if enc_type == "EigVals":
                enc = data.EigVals[0].T
            elif enc_type == "RWGE":
                enc = data.pestat_RWSE.mean(0, keepdim=True)
            else:
                enc = getattr(data, f"gestat_{enc_type}")
            enc_list.append(enc)

        combined_graph_pe = torch.nan_to_num(torch.hstack(enc_list))

        if cfg.dataset.combine_output_pestat:
            combined_stats.append(combined_graph_pe.repeat(data.x.shape[0], 1))
        else:
            data.y_graph = combined_graph_pe

    # Combined pestat
    if cfg.dataset.combine_output_pestat:
        data.pestat_CombinedPSE = torch.hstack(combined_stats)
        cfg.posenc_CombinedPSE._raw_dim = data.pestat_CombinedPSE.shape[1]


def compute_posenc_stats(data, pe_types, is_undirected, cfg):
    """Precompute positional encodings for the given graph.

    Supported PE statistics to precompute, selected by `pe_types`:
    'LapPE': Laplacian eigen-decomposition.
    'RWSE': Random walk landing probabilities (diagonals of RW matrices).
    'HKfullPE': Full heat kernels and their diagonals. (NOT IMPLEMENTED)
    'HKdiagSE': Diagonals of heat kernel diffusion.
    'ElstaticPE': Kernel based on the electrostatic interaction between nodes.

    Args:
        data: PyG graph
        pe_types: Positional encoding types to precompute statistics for.
            This can also be a combination, e.g. 'eigen+rw_landing'
        is_undirected: True if the graph is expected to be undirected
        cfg: Main configuration node

    Returns:
        Extended PyG Data object.
    """
    _check_all_types(pe_types)

    # Basic preprocessing of the input graph.
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    laplacian_norm_type = cfg.posenc_LapPE.eigen.laplacian_norm.lower()
    if laplacian_norm_type == 'none':
        laplacian_norm_type = None
    if is_undirected:
        undir_edge_index = data.edge_index
    else:
        undir_edge_index = to_undirected(data.edge_index)

    # Eigen values and vectors.
    evals, evects = None, None
    if 'LapPE' in pe_types or 'EquivStableLapPE' in pe_types:
        # Eigen-decomposition with numpy, can be reused for Heat kernels.
        L = to_scipy_sparse_matrix(
            *get_laplacian(undir_edge_index, normalization=laplacian_norm_type,
                           num_nodes=N)
        )
        evals, evects = np.linalg.eigh(L.toarray())

        if 'LapPE' in pe_types:
            max_freqs = cfg.posenc_LapPE.eigen.max_freqs
            eigvec_norm = cfg.posenc_LapPE.eigen.eigvec_norm
            skip_zero_freq = cfg.posenc_LapPE.eigen.skip_zero_freq
            eigvec_abs = cfg.posenc_LapPE.eigen.eigvec_abs
        elif 'EquivStableLapPE' in pe_types:
            max_freqs = cfg.posenc_EquivStableLapPE.eigen.max_freqs
            eigvec_norm = cfg.posenc_EquivStableLapPE.eigen.eigvec_norm
            skip_zero_freq = cfg.posenc_EquivStableLapPE.eigen.skip_zero_freq
            eigvec_abs = cfg.posenc_EquivStableLapPE.eigen.eigvec_abs

        data.EigVals, data.EigVecs = get_lap_decomp_stats(
            evals=evals, evects=evects,
            max_freqs=max_freqs,
            eigvec_norm=eigvec_norm,
            skip_zero_freq=skip_zero_freq,
            eigvec_abs=eigvec_abs)

    if 'SignNet' in pe_types:
        # Eigen-decomposition with numpy for SignNet.
        norm_type = cfg.posenc_SignNet.eigen.laplacian_norm.lower()
        if norm_type == 'none':
            norm_type = None
        L = to_scipy_sparse_matrix(
            *get_laplacian(undir_edge_index, normalization=norm_type,
                           num_nodes=N)
        )
        evals_sn, evects_sn = np.linalg.eigh(L.toarray())
        data.eigvals_sn, data.eigvecs_sn = get_lap_decomp_stats(
            evals=evals_sn, evects=evects_sn,
            max_freqs=cfg.posenc_SignNet.eigen.max_freqs,
            eigvec_norm=cfg.posenc_SignNet.eigen.eigvec_norm,
            skip_zero_freq=cfg.posenc_SignNet.eigen.skip_zero_freq,
            eigvec_abs=cfg.posenc_SignNet.eigen.eigvec_abs)
        data.EigVals = data.eigvals_sn

    # Random Walks.
    if 'RWSE' in pe_types or 'RWGE' in pe_types:
        if 'RWSE' in pe_types:
            kernel_param = cfg.posenc_RWSE.kernel
        elif 'RWGE' in pe_types:
            kernel_param = cfg.posenc_RWGE.kernel

        if len(kernel_param.times) == 0:
            raise ValueError("List of kernel times required for RW")
        rw_landing = get_rw_landing_probs(ksteps=kernel_param.times,
                                          edge_index=data.edge_index,
                                          num_nodes=N)
        data.pestat_RWSE = rw_landing

    # Heat Kernels.
    if 'HKdiagSE' in pe_types or 'HKfullPE' in pe_types:
        # Get the eigenvalues and eigenvectors of the regular Laplacian,
        # if they have not yet been computed for 'eigen'.
        if laplacian_norm_type is not None or evals is None or evects is None:
            L_heat = to_scipy_sparse_matrix(
                *get_laplacian(undir_edge_index, normalization=None, num_nodes=N)
            )
            evals_heat, evects_heat = np.linalg.eigh(L_heat.toarray())
        else:
            evals_heat, evects_heat = evals, evects
        evals_heat = torch.from_numpy(evals_heat)
        evects_heat = torch.from_numpy(evects_heat)

        # Get the full heat kernels.
        if 'HKfullPE' in pe_types:
            # The heat kernels can't be stored in the Data object without
            # additional padding because in PyG's collation of the graphs the
            # sizes of tensors must match except in dimension 0. Do this when
            # the full heat kernels are actually used downstream by an Encoder.
            raise NotImplementedError()
            # heat_kernels, hk_diag = get_heat_kernels(evects_heat, evals_heat,
            #                                   kernel_times=kernel_param.times)
            # data.pestat_HKdiagSE = hk_diag
        # Get heat kernel diagonals in more efficient way.
        if 'HKdiagSE' in pe_types:
            kernel_param = cfg.posenc_HKdiagSE.kernel
            if len(kernel_param.times) == 0:
                raise ValueError("Diffusion times are required for heat kernel")
            hk_diag = get_heat_kernels_diag(evects_heat, evals_heat,
                                            kernel_times=kernel_param.times,
                                            space_dim=0)
            data.pestat_HKdiagSE = hk_diag

    # Electrostatic interaction inspired kernel.
    if 'ElstaticPE' in pe_types:
        elstatic = get_electrostatic_function_encoding(undir_edge_index, N)
        data.pestat_ElstaticPE = elstatic

    if 'CycleGE' in pe_types:
        kernel_param = cfg.graphenc_CycleGE.kernel
        data.gestat_CycleGE = count_cycles(kernel_param.times, data)

    if 'NormalFixedRE' in pe_types:
        shape = (data.num_nodes, cfg.posenc_NormalFixedRE.dim_pe)
        data.pestat_NormalFixedRE = torch.normal(0, 1, shape)

    # Set up PE tasks if the input and output PEs are defined
    in_node_encs = cfg.dataset.input_node_encoders
    out_node_encs = cfg.dataset.output_node_encoders
    out_graph_encs = cfg.dataset.output_graph_encoders
    # Combine encodings if multiple encoding types are specified, sep by '+'
    if not all(item in ["none", "NormalSE", "UniformSE", "BernoulliSE"]
               for item in list(in_node_encs) + list(out_node_encs)):
        _combine_encs(data, in_node_encs, out_node_encs, out_graph_encs, cfg)

    return data


def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2',
                         skip_zero_freq: bool = True, eigvec_abs: bool = False):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
        skip_zero_freq: Start with first non-zero frequency eigenpairs if
            set to True. Otherwise, use first max_freqs eigenpairs.
        eigvec_abs: Use the absolute value of the eigenvectors if set to True.
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    N = len(evals)  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.
    offset = (abs(evals) < EPS).sum().clip(0, N) if skip_zero_freq else 0
    idx = evals.argsort()[offset:max_freqs + offset]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs + offset:
        EigVecs = F.pad(evects, (0, max_freqs + offset - N), value=float('nan'))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs + offset:
        EigVals = F.pad(evals, (0, max_freqs + offset - N),
                        value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)
    EigVecs = EigVecs.abs() if eigvec_abs else EigVecs

    return EigVals, EigVecs


def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, source, dim=0, dim_size=num_nodes)  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing


def get_heat_kernels_diag(evects, evals, kernel_times=[], space_dim=0):
    """Compute Heat kernel diagonal.

    This is a continuous function that represents a Gaussian in the Euclidean
    space, and is the solution to the diffusion equation.
    The random-walk diagonal should converge to this.

    Args:
        evects: Eigenvectors of the Laplacian matrix
        evals: Eigenvalues of the Laplacian matrix
        kernel_times: Time for the diffusion. Analogous to the k-steps in random
            walk. The time is equivalent to the variance of the kernel.
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the diffusion diagonal by a factor `t^(space_dim/2)`. In
            euclidean space, this correction means that the height of the
            gaussian stays constant across time, if `space_dim` is the dimension
            of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    heat_kernels_diag = []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2., dim=0)

        # Remove eigenvalues == 0 from the computation of the heat kernel
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]

        # Change the shapes for the computations
        evals = evals.unsqueeze(-1)  # lambda_{i, ..., ...}
        evects = evects.transpose(0, 1)  # phi_{i,j}: i-th eigvec X j-th node

        # Compute the heat kernels diagonal only for each time
        eigvec_mul = evects ** 2
        for t in kernel_times:
            # sum_{i>0}(exp(-2 t lambda_i) * phi_{i, j} * phi_{i, j})
            this_kernel = torch.sum(torch.exp(-t * evals) * eigvec_mul,
                                    dim=0, keepdim=False)

            # Multiply by `t` to stabilize the values, since the gaussian height
            # is proportional to `1/t`
            heat_kernels_diag.append(this_kernel * (t ** (space_dim / 2)))
        heat_kernels_diag = torch.stack(heat_kernels_diag, dim=0).transpose(0, 1)

    return heat_kernels_diag


def get_heat_kernels(evects, evals, kernel_times=[]):
    """Compute full Heat diffusion kernels.

    Args:
        evects: Eigenvectors of the Laplacian matrix
        evals: Eigenvalues of the Laplacian matrix
        kernel_times: Time for the diffusion. Analogous to the k-steps in random
            walk. The time is equivalent to the variance of the kernel.
    """
    heat_kernels, rw_landing = [], []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2., dim=0)

        # Remove eigenvalues == 0 from the computation of the heat kernel
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]

        # Change the shapes for the computations
        evals = evals.unsqueeze(-1).unsqueeze(-1)  # lambda_{i, ..., ...}
        evects = evects.transpose(0, 1)  # phi_{i,j}: i-th eigvec X j-th node

        # Compute the heat kernels for each time
        eigvec_mul = (evects.unsqueeze(2) * evects.unsqueeze(1))  # (phi_{i, j1, ...} * phi_{i, ..., j2})
        for t in kernel_times:
            # sum_{i>0}(exp(-2 t lambda_i) * phi_{i, j1, ...} * phi_{i, ..., j2})
            heat_kernels.append(
                torch.sum(torch.exp(-t * evals) * eigvec_mul,
                          dim=0, keepdim=False)
            )

        heat_kernels = torch.stack(heat_kernels, dim=0)  # (Num kernel times) x (Num nodes) x (Num nodes)

        # Take the diagonal of each heat kernel,
        # i.e. the landing probability of each of the random walks
        rw_landing = torch.diagonal(heat_kernels, dim1=-2, dim2=-1).transpose(0, 1)  # (Num nodes) x (Num kernel times)

    return heat_kernels, rw_landing


def get_electrostatic_function_encoding(edge_index, num_nodes):
    """Kernel based on the electrostatic interaction between nodes.
    """
    L = to_scipy_sparse_matrix(
        *get_laplacian(edge_index, normalization=None, num_nodes=num_nodes)
    ).todense()
    L = torch.as_tensor(L)
    Dinv = torch.eye(L.shape[0]) * (L.diag() ** -1)
    A = deepcopy(L).abs()
    A.fill_diagonal_(0)
    DinvA = Dinv.matmul(A)

    evals, evecs = torch.linalg.eigh(L)
    offset = (evals < EPS).sum().item()
    if offset == num_nodes:
        return torch.zeros(num_nodes, 7, dtype=torch.float32)

    electrostatic = evecs[:, offset:] / evals[offset:] @ evecs[:, offset:].T
    electrostatic = electrostatic - electrostatic.diag()
    green_encoding = torch.stack([
        electrostatic.min(dim=0)[0],  # Min of Vi -> j
        electrostatic.mean(dim=0),  # Mean of Vi -> j
        electrostatic.std(dim=0),  # Std of Vi -> j
        electrostatic.min(dim=1)[0],  # Min of Vj -> i
        electrostatic.std(dim=1),  # Std of Vj -> i
        (DinvA * electrostatic).sum(dim=0),  # Mean of interaction on direct neighbour
        (DinvA * electrostatic).sum(dim=1),  # Mean of interaction from direct neighbour
    ], dim=1)

    return green_encoding


def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """

    EigVals = EigVals.unsqueeze(0)

    if normalization in ["L1", "L2", "abs-max", "min-max"]:
        return normalizer(EigVecs, normalization, eps)

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / np.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs


def normalizer(x: torch.Tensor, normalization: str = "L2", eps: float = 1e-12):
    if normalization == "none":
        return x

    elif normalization == "L1":
        # L1 normalization: vec / sum(abs(vec))
        denom = x.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: vec / sqrt(sum(vec^2))
        denom = x.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: vec / max|vec|
        denom = torch.max(x.abs(), dim=0, keepdim=True).values

    elif normalization == "min-max":
        # MinMax normalization: (vec - min(vec)) / (max(vec) - min(vec))
        x = x - x.min(dim=0, keepdim=True).values
        denom = x.max(dim=0, keepdim=True).values

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    return x / denom.clamp_min(eps).expand_as(x)
