import logging
import time
from typing import Dict, List, Tuple

import numba
import numpy as np
import torch
from scipy.stats import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, \
    confusion_matrix
from torch_geometric.graphgym import get_current_gpu_usage
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.logger import infer_task, Logger
from torch_geometric.graphgym.utils.io import dict_to_json, dict_to_tb
from torchmetrics.functional import auroc

import graphgym.metrics_ogb as metrics_ogb
from graphgym.metric_wrapper import MetricWrapper
from graphgym.utils import get_device

EPS = 1e-6  # values below which we consider as zeros


def reformat_score_dict(score_dict: Dict[str, float], /):
    return {i: round(float(j), cfg.round) for i, j in score_dict.items()}


def accuracy_SBM(targets, pred_int):
    """Accuracy eval for Benchmarking GNN's PATTERN and CLUSTER datasets.
    https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/train/metrics.py#L34
    """
    S = targets
    C = pred_int
    CM = confusion_matrix(S, C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets == r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r, r] / float(cluster.shape[0])
            if CM[r, r] > 0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = np.sum(pr_classes) / float(nb_classes)
    return acc


class CustomLogger(Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Whether to run comparison tests of alternative score implementations.
        self.test_scores = False

    def reset(self):
        super().reset()
        self._batch_idx = []

    # basic properties
    def basic(self):
        stats = {
            'loss': round(self._loss / self._size_current, max(8, cfg.round)),
            'lr': round(self._lr, max(8, cfg.round)),
            'params': self._params,
            'time_iter': round(self.time_iter(), cfg.round),
        }
        gpu_memory = get_current_gpu_usage()
        if gpu_memory > 0:
            stats['gpu_memory'] = gpu_memory
        return stats

    # task properties
    def classification_binary(self):
        true = torch.cat(self._true).squeeze(-1)
        pred_score = torch.cat(self._pred)
        pred_int = self._get_pred_int(pred_score)

        if true.shape[0] < 1e7:  # AUROC computation for very large datasets is too slow.
            # TorchMetrics AUROC on GPU if available.
            device = get_device(cfg.val.accelerator, cfg.accelerator)
            auroc_score = auroc(pred_score.to(device),
                                true.to(device),
                                task='binary')
            if self.test_scores:
                # SK-learn version.
                try:
                    r_a_score = roc_auc_score(true.cpu().numpy(),
                                              pred_score.cpu().numpy())
                except ValueError:
                    r_a_score = 0.0
                assert np.isclose(float(auroc_score), r_a_score)
        else:
            auroc_score = 0.

        reformat = lambda x: round(float(x), cfg.round)
        res = {
            'accuracy': reformat(accuracy_score(true, pred_int)),
            'precision': reformat(precision_score(true, pred_int)),
            'recall': reformat(recall_score(true, pred_int)),
            'f1': reformat(f1_score(true, pred_int)),
            'auc': reformat(auroc_score),
        }
        if cfg.metric_best == 'accuracy-SBM':
            res['accuracy-SBM'] = reformat(accuracy_SBM(true, pred_int))
        return res

    def classification_multi(self):
        true, pred_score = torch.cat(self._true), torch.cat(self._pred)
        pred_int = self._get_pred_int(pred_score)
        reformat = lambda x: round(float(x), cfg.round)

        res = {
            'accuracy': reformat(accuracy_score(true, pred_int)),
            'f1': reformat(f1_score(true, pred_int,
                                    average='macro', zero_division=0)),
        }
        if cfg.metric_best == 'accuracy-SBM':
            res['accuracy-SBM'] = reformat(accuracy_SBM(true, pred_int))
        if true.shape[0] < 1e7:
            # AUROC computation for very large datasets runs out of memory.
            # TorchMetrics AUROC on GPU is much faster than sklearn for large ds
            device = get_device(cfg.val.accelerator, cfg.accelerator)
            res['auc'] = reformat(auroc(pred_score.to(device),
                                        true.to(device).squeeze(),
                                        task='multiclass',
                                        num_classes=pred_score.shape[1],
                                        average='macro'))

            if self.test_scores:
                # SK-learn version.
                sk_auc = reformat(roc_auc_score(true, pred_score.exp(),
                                                average='macro',
                                                multi_class='ovr'))
                assert np.isclose(sk_auc, res['auc'])

        return res

    def classification_multilabel(self):
        true, pred_score = torch.cat(self._true), torch.cat(self._pred)
        reformat = lambda x: round(float(x), cfg.round)

        # Send to GPU to speed up TorchMetrics if possible.
        device = get_device(cfg.val.accelerator, cfg.accelerator)
        true = true.to(device)
        pred_score = pred_score.to(device)
        # MetricWrapper will remove NaNs and apply the metric to each target dim
        acc = MetricWrapper(metric='accuracy',
                            target_nan_mask='ignore-mean-label',
                            task='binary',
                            cast_to_int=True)
        auroc = MetricWrapper(metric='auroc',
                              target_nan_mask='ignore-mean-label',
                              task='binary',
                              cast_to_int=True)
        # ap = MetricWrapper(metric='averageprecision',
        #                    target_nan_mask='ignore-mean-label',
        #                    task='binary',
        #                    cast_to_int=True)
        ogb_ap = reformat(metrics_ogb.eval_ap(true.cpu().numpy(),
                                              pred_score.cpu().numpy())['ap'])
        # Send to GPU to speed up TorchMetrics if possible.
        true = true.to(device)
        pred_score = pred_score.to(device)
        results = {
            'accuracy': reformat(acc(torch.sigmoid(pred_score), true)),
            'auc': reformat(auroc(pred_score, true)),
            # 'ap': reformat(ap(pred_score, true)),  # Slightly differs from sklearn.
            'ap': ogb_ap,
        }

        if self.test_scores:
            # Compute metric by OGB Evaluator methods.
            true = true.cpu().numpy()
            pred_score = pred_score.cpu().numpy()
            ogb = {
                'accuracy': reformat(metrics_ogb.eval_acc(
                    true, (pred_score > 0.).astype(int))['acc']),
                'ap': reformat(metrics_ogb.eval_ap(true, pred_score)['ap']),
                'auc': reformat(
                    metrics_ogb.eval_rocauc(true, pred_score)['rocauc']),
            }
            assert np.isclose(ogb['accuracy'], results['accuracy'])
            assert np.isclose(ogb['ap'], results['ap'])
            assert np.isclose(ogb['auc'], results['auc'])

        return results

    def subtoken_prediction(self):
        from ogb.graphproppred import Evaluator
        evaluator = Evaluator('ogbg-code2')

        seq_ref_list = []
        seq_pred_list = []
        for seq_pred, seq_ref in zip(self._pred, self._true):
            seq_ref_list.extend(seq_ref)
            seq_pred_list.extend(seq_pred)

        input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}
        result = evaluator.eval(input_dict)
        result['f1'] = result['F1']
        del result['F1']
        return result

    def regression(self):
        # NOTE: assumes that true / pred are 2d arrays
        true, pred = torch.cat(self._true).numpy(), torch.cat(self._pred).numpy()
        res = {
            'mae': mean_absolute_error(true, pred),
            'mse': mean_squared_error(true, pred),
            'rmse': mean_squared_error(true, pred, squared=False),
        }
        if (cfg.gnn.head.startswith("inductive_node")
                or cfg.gnn.head.startswith("inductive_hybrid")):
            # Enable computing graph-wide evaluated scores
            true, pred, batch_idx = self.combine_batch_idx(self._batch_idx,
                                                           true, pred)
        else:
            batch_idx = None
        record_individual = cfg.train.record_individual_scores
        res.update(eval_spearmanr(true, pred, batch_idx, record_individual))
        res.update(eval_pearsonr(true, pred, batch_idx, record_individual))
        res.update(eval_r2(true, pred, batch_idx, record_individual))
        return reformat_score_dict(res)

    @staticmethod
    def combine_batch_idx(
        batch_idx: List[torch.Tensor],
        true: np.ndarray,
        pred: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bidx = torch.cat(batch_idx)

        if bidx.min() == -1:
            # Move -1 (graph pred in hybrid task) bidx to the last 'batch'
            node_task_idx = torch.where(bidx >= 0)[0]
            graph_task_idx = torch.where(bidx == -1)[0]
            bidx[graph_task_idx] = bidx[node_task_idx[-1]] + 1
            idx = torch.concat([node_task_idx, graph_task_idx])
            true, pred, bidx = true[idx.numpy()], pred[idx.numpy()], bidx[idx]

        # Offset batch index so that it is monotonically increasing
        batch_split_points = torch.where(bidx[1:] < bidx[:-1])[0]
        offsets = (bidx[batch_split_points] + 1).cumsum(0)
        for i in range(num_split_points := len(batch_split_points)):
            start = batch_split_points[i] + 1
            if i == num_split_points - 1:
                end = len(bidx)
            else:
                end = batch_split_points[i + 1] + 1
            bidx[start:end] += offsets[i]
        if (bidx[:-1] > bidx[1:]).any():
            raise ValueError("Batch index is not monotonically increasing")

        return true, pred, bidx.numpy()

    def update_stats(self, true, pred, batch_idx, loss, lr, time_used, params,
                     dataset_name=None, **kwargs):
        if dataset_name == 'ogbg-code2':
            assert true['y_arr'].shape[1] == len(pred)  # max_seq_len (5)
            assert true['y_arr'].shape[0] == pred[0].shape[0]  # batch size
            batch_size = true['y_arr'].shape[0]

            # Decode the predicted sequence tokens, so we don't need to store
            # the logits that take significant memory.
            from graphgym.loader.ogbg_code2_utils import idx2vocab, \
                decode_arr_to_seq
            arr_to_seq = lambda arr: decode_arr_to_seq(arr, idx2vocab)
            mat = []
            for i in range(len(pred)):
                mat.append(torch.argmax(pred[i].detach(), dim=1).view(-1, 1))
            mat = torch.cat(mat, dim=1)
            seq_pred = [arr_to_seq(arr) for arr in mat]
            seq_ref = [true['y'][i] for i in range(len(true['y']))]
            pred = seq_pred
            true = seq_ref
        else:
            assert true.shape[0] == pred.shape[0]
            batch_size = true.shape[0]
        self._iter += 1
        self._true.append(true)
        self._pred.append(pred)
        self._batch_idx.append(batch_idx)
        self._size_current += batch_size
        self._loss += loss * batch_size
        self._lr = lr
        self._params = params
        self._time_used += time_used
        self._time_total += time_used
        for key, val in kwargs.items():
            if key not in self._custom_stats:
                self._custom_stats[key] = val * batch_size
            else:
                self._custom_stats[key] += val * batch_size

    def write_epoch(self, cur_epoch):
        start_time = time.perf_counter()
        basic_stats = self.basic()

        if self.task_type == 'regression':
            task_stats = self.regression()
        elif self.task_type == 'classification_binary':
            task_stats = self.classification_binary()
        elif self.task_type == 'classification_multi':
            task_stats = self.classification_multi()
        elif self.task_type in ['classification_multilabel', 'multilabel']:
            task_stats = self.classification_multilabel()
        elif self.task_type == 'subtoken_prediction':
            task_stats = self.subtoken_prediction()
        else:
            raise ValueError('Task has to be regression or classification')

        epoch_stats = {'epoch': cur_epoch,
                       'time_epoch': round(self._time_used, cfg.round)}
        eta_stats = {'eta': round(self.eta(cur_epoch), cfg.round),
                     'eta_hours': round(self.eta(cur_epoch) / 3600, cfg.round)}
        custom_stats = self.custom()

        if self.name == 'train':
            stats = {
                **epoch_stats,
                **eta_stats,
                **basic_stats,
                **task_stats,
                **custom_stats
            }
        else:
            stats = {
                **epoch_stats,
                **basic_stats,
                **task_stats,
                **custom_stats
            }

        # print
        logging.info('{}: {}'.format(self.name, stats))
        # json
        dict_to_json(stats, '{}/stats.json'.format(self.out_dir))
        # tensorboard
        if cfg.tensorboard_each_run:
            dict_to_tb(stats, self.tb_writer, cur_epoch)
        self.reset()
        if cur_epoch < 3:
            logging.info(f"...computing epoch stats took: "
                         f"{time.perf_counter() - start_time:.2f}s")
        return stats


def create_logger():
    """
    Create logger for the experiment

    Returns: List of logger objects

    """
    loggers = []
    names = ['train', 'val', 'test']
    for i, dataset in enumerate(range(cfg.share.num_splits)):
        loggers.append(CustomLogger(name=names[i], task_type=infer_task()))
    return loggers


def ensure_2d_array(x: np.ndarray, /) -> np.ndarray:
    return np.expand_dims(x, axis=1) if len(x.shape) == 1 else x


def _make_res_dict(
    name: str,
    scores: np.ndarray,
    record_individual: bool,
) -> Dict[str, float]:
    avg_score = np.nanmean(scores)
    res = {name: avg_score}
    if record_individual:
        for i, j in enumerate(scores):
            res[f"{name}_{i}"] = j
    return res


def _spearmanr(y_true, y_pred):
    res_list = []

    if y_true.ndim == 1:
        res_list.append(stats.spearmanr(y_true, y_pred)[0])
    else:
        for i in range(y_true.shape[1]):
            # ignore nan values
            is_labeled = ~np.isnan(y_true[:, i])
            res_list.append(stats.spearmanr(y_true[is_labeled, i],
                                            y_pred[is_labeled, i])[0])

    return np.array(res_list)


def eval_spearmanr(y_true, y_pred, batch_idx=None, record_individual=False):
    """Compute Spearman Rho averaged across tasks.
    """
    y_true, y_pred = ensure_2d_array(y_true), ensure_2d_array(y_pred)
    global_scores = _spearmanr(y_true, y_pred)
    res = _make_res_dict("spearmanr_global", global_scores, False)
    # FIX: Computing spearman for every graph is too expensive, disable for now
    #
    # res = {'spearmanr_global': _spearmanr(y_true, y_pred)}
    # if batch_idx is not None:
    #     instancewide_spearmanr = []
    #     for i in np.unique(batch_idx):
    #         ind = batch_idx == i
    #         instancewide_spearmanr.append(_spearmanr(y_true[ind], y_pred[ind]))
    #     res['spearmanr'] = np.nanmean(instancewide_spearmanr)

    #     if (num_nans := np.isnan(np.array(instancewide_spearmanr)).sum()) > 0:
    #         logging.warning(f"{num_nans} NaNs")
    return res


@numba.njit("f4[:](f4[:, :], f4[:, :])", nogil=True)
def pearsonr(y_true, y_pred):
    y_true_norm = y_true.T.copy()
    y_pred_norm = y_pred.T.copy()

    num_feat = y_true.shape[1]
    scores = np.zeros(num_feat, dtype=np.float32)
    for i in range(num_feat):
        std1 = y_true_norm[i].std()
        std2 = y_pred_norm[i].std()

        if std1 < EPS:  # skip constant targets
            scores[i] = np.nan
        elif std2 != 0:
            y_true_norm[i] = (y_true_norm[i] - y_true_norm[i].mean()) / std1
            y_pred_norm[i] = (y_pred_norm[i] - y_pred_norm[i].mean()) / std2
            scores[i] = (y_true_norm[i] * y_pred_norm[i]).mean()

    return scores


@numba.njit("Tuple((i8, u4[:]))(i8[:])", nogil=True)
def prepare_indptr(batch_idx):
    num_instances = batch_idx[-1] + 1
    indptr = np.zeros(num_instances + 1, dtype=np.uint32)

    current_index = 0
    for idx in batch_idx:
        if idx == (current_index + 1):
            current_index += 1
            indptr[current_index + 1] = indptr[current_index]
        elif idx != current_index:
            raise ValueError("Batch index array is not contiguous")

        indptr[current_index + 1] += 1

    return num_instances, indptr


@numba.njit("f4[:](f4[:, :], f4[:, :], i8[:])", nogil=True, parallel=True)
def pearsonr_split(y_true, y_pred, batch_idx):
    num_instances, indptr = prepare_indptr(batch_idx)
    num_feat = y_true.shape[1]
    raw_scores = np.zeros((num_instances, num_feat), dtype=np.float32)
    scores = np.zeros(num_feat, dtype=np.float32)
    for i in numba.prange(num_instances):
        start, end = indptr[i:i+2]
        raw_scores[i] = pearsonr(y_true[start:end], y_pred[start:end])
    for i in range(num_feat):
        scores[i] = np.nanmean(raw_scores[i])
    return scores


def eval_pearsonr(y_true, y_pred, batch_idx=None, record_individual=False):
    y_true, y_pred = ensure_2d_array(y_true), ensure_2d_array(y_pred)
    global_scores = pearsonr(y_true, y_pred)
    res = _make_res_dict("pearsonr_global", global_scores, False)
    if batch_idx is not None:
        split_scores = pearsonr_split(y_true, y_pred, batch_idx)
        res.update(_make_res_dict("pearsonr", split_scores, record_individual))
    return res


@numba.njit("f4[:](f4[:, :], f4[:, :])", nogil=True)
def r2(y_true, y_pred):
    y_true_t = y_true.T.copy()
    y_pred_t = y_pred.T.copy()

    num_feat = y_true.shape[1]
    scores = np.zeros(num_feat, dtype=np.float32)
    for i in range(num_feat):
        rss = ((y_true_t[i] - y_pred_t[i]) ** 2).sum()
        tss = ((y_true_t[i] - y_true_t[i].mean()) ** 2).sum()
        scores[i] = 1 - rss / tss if tss > EPS else np.nan

    return scores


@numba.njit("f4[:](f4[:, :], f4[:, :], i8[:])", nogil=True, parallel=True)
def r2_split(y_true, y_pred, batch_idx):
    num_instances, indptr = prepare_indptr(batch_idx)
    num_feat = y_true.shape[1]
    raw_scores = np.zeros((num_instances, num_feat), dtype=np.float32)
    scores = np.zeros(num_feat, dtype=np.float32)
    for i in numba.prange(num_instances):
        start, end = indptr[i:i+2]
        raw_scores[i] = r2(y_true[start:end], y_pred[start:end])
    for i in range(num_feat):
        scores[i] = np.nanmean(raw_scores[i])
    return scores


def eval_r2(y_true, y_pred, batch_idx=None, record_individual=False):
    y_true, y_pred = ensure_2d_array(y_true), ensure_2d_array(y_pred)
    global_scores = r2(y_true, y_pred)
    res = _make_res_dict("r2_global", global_scores, False)
    if batch_idx is not None:
        split_scores = r2_split(y_true, y_pred, batch_idx)
        res.update(_make_res_dict("r2", split_scores, record_individual))

    if False:  # for testing whether there are spuriously low variance targets
        low_var_idx_list = [
            i for i in range(batch_idx[-1] + 1)
            if any(0 < i < EPS for i in y_true[batch_idx == i].var(0))
        ]
        if low_var_idx_list:
            print(f"{low_var_idx_list=}")

    return res
