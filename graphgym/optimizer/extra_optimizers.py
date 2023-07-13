import logging
import math
from typing import Dict, Iterator, List, Tuple
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.optim as optim
from torch.nn import Parameter
from torch.optim import Adagrad, AdamW, Optimizer, RMSprop, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.optim import SchedulerConfig
import torch_geometric.graphgym.register as register


class FLAGAdamW(AdamW):

    model_modules_names_dict: Dict[str, List[str]] = {
        "gnn": ["encoder", "pre_mp", "mp", "post_mp"],
        "GPSModel": ["encoder", "layers", "post_mp"],
    }

    def __init__(
        self,
        params,
        base_lr: float,
        *,
        weight_decay: float = 0.0,
    ):
        if (model_type := cfg.model.type) not in self.model_modules_names_dict:
            raise NotADirectoryError(
                f"Model type {model_type!r} is not supported for "
                f"{self.__class__.__name__} yet. Please add the expected "
                f"module names to the class to enable support.")
        self.model_modules_names = self.model_modules_names_dict[model_type]
        self.steps = cfg.optim.flag_steps
        self.step_size = cfg.optim.flag_step_size

        super().__init__(params, lr=base_lr, weight_decay=weight_decay)

    @staticmethod
    def forward_with_pert(graphgym_gnn, batch, pert, module_names):
        batch = graphgym_gnn.encoder(batch.clone())
        batch.x += pert  # attack encoded features

        for module_name in module_names[1:]:  # skip encoder
            module = getattr(graphgym_gnn.model, module_name, None)
            if module is not None:
                # print(module_name)
                batch = module(batch)

        return batch

    @staticmethod
    @torch.no_grad()
    def infer_pert_shape(graphgym_gnn, batch) -> Tuple[int, int]:
        """Infer perturbation dimension from encoded features."""
        batch = graphgym_gnn.encoder(batch.clone())
        return tuple(batch.x.shape)

    def flag(self, graphgym_gnn, batch, loss_func):
        self.zero_grad()

        pert_shape = self.infer_pert_shape(graphgym_gnn, batch)
        pert = torch.zeros(*pert_shape, device=batch.edge_index.device)
        pert.uniform_(-self.step_size, self.step_size).requires_grad_()

        pred, true = self.forward_with_pert(graphgym_gnn, batch, pert,
                                            self.model_modules_names)
        loss, pred_score = loss_func(pred, true)
        loss /= self.steps

        for _ in range(self.steps - 1):
            loss.backward()

            sign = torch.sign(pert.grad.detach())
            pert_data = pert.detach() + self.step_size * sign
            pert.data = pert_data.data
            pert.grad[:] = 0

            pred, true = self.forward_with_pert(graphgym_gnn, batch, pert,
                                                self.model_modules_names)
            loss, pred_score = loss_func(pred, true)
            loss /= self.steps

        loss.backward()
        self.step()

        return loss, pred_score, true


@register.register_optimizer('FLAGAdamW')
def flagadamw_optimizer(params: Iterator[Parameter], base_lr: float,
                        weight_decay: float):
    return FLAGAdamW(params, base_lr, weight_decay=weight_decay)


class ASAM(SGD):
    """https://github.com/SamsungLabs/ASAM/blob/master/asam.py
    """

    def __init__(self, params, model, base_lr: float, *, rho=0.5, eta=0.01,
                 momentum=0.9, weight_decay=0.0):
        self.model = model
        self.params = params
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)
        super().__init__(params, lr=base_lr, weight_decay=weight_decay,
                         momentum=momentum)

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)
        self.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for p in self.params:
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        if cfg.optim.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.step()
        self.zero_grad()


@register.register_optimizer('ASAM')
def asam_optimizer(params: Iterator[Parameter], base_lr: float,
                   weight_decay: float):
    return ASAM(params, base_lr, weight_decay=weight_decay)


@register.register_optimizer('adagrad')
def adagrad_optimizer(params: Iterator[Parameter], base_lr: float,
                      weight_decay: float) -> Adagrad:
    return Adagrad(params, lr=base_lr, weight_decay=weight_decay)


@register.register_optimizer('adamW')
def adamW_optimizer(params: Iterator[Parameter], base_lr: float,
                    weight_decay: float) -> AdamW:
    return AdamW(params, lr=base_lr, weight_decay=weight_decay)


@register.register_optimizer('rmsprop')
def rmsprop_optimizer(params: Iterator[Parameter], base_lr: float,
                      weight_decay: float) -> AdamW:
    return RMSprop(params, lr=base_lr, weight_decay=weight_decay)


@dataclass
class ExtendedSchedulerConfig(SchedulerConfig):
    reduce_factor: float = 0.5
    schedule_patience: int = 15
    min_lr: float = 1e-6
    num_warmup_epochs: int = 10
    train_mode: str = 'custom'
    eval_period: int = 1


@register.register_scheduler('plateau')
def plateau_scheduler(optimizer: Optimizer, patience: int,
                      lr_decay: float) -> ReduceLROnPlateau:
    return ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay)


@register.register_scheduler('reduce_on_plateau')
def scheduler_reduce_on_plateau(optimizer: Optimizer, reduce_factor: float,
                                schedule_patience: int, min_lr: float,
                                train_mode: str, eval_period: int):
    if train_mode == 'standard':
        raise ValueError("ReduceLROnPlateau scheduler is not supported "
                         "by 'standard' graphgym training mode pipeline; "
                         "try setting config 'train.mode: custom'")

    if eval_period != 1:
        logging.warning("When config train.eval_period is not 1, the "
                        "optim.schedule_patience of ReduceLROnPlateau "
                        "may not behave as intended.")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=reduce_factor,
        patience=schedule_patience,
        min_lr=min_lr,
        verbose=True
    )
    if not hasattr(scheduler, 'get_last_lr'):
        # ReduceLROnPlateau doesn't have `get_last_lr` method as of current
        # pytorch1.10; we add it here for consistency with other schedulers.
        def get_last_lr(self):
            """ Return last computed learning rate by current scheduler.
            """
            return self._last_lr

        scheduler.get_last_lr = get_last_lr.__get__(scheduler)
        scheduler._last_lr = [group['lr']
                              for group in scheduler.optimizer.param_groups]

    def modified_state_dict(ref):
        """Returns the state of the scheduler as a :class:`dict`.
        Additionally modified to ignore 'get_last_lr', 'state_dict'.
        Including these entries in the state dict would cause issues when
        loading a partially trained / pretrained model from a checkpoint.
        """
        return {key: value for key, value in ref.__dict__.items()
                if key not in ['sparsifier', 'get_last_lr', 'state_dict']}

    scheduler.state_dict = modified_state_dict.__get__(scheduler)

    return scheduler


@register.register_scheduler('linear_with_warmup')
def linear_with_warmup_scheduler(optimizer: Optimizer,
                                 num_warmup_epochs: int, max_epoch: int):
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_epochs,
        num_training_steps=max_epoch
    )
    return scheduler


@register.register_scheduler('cosine_with_warmup')
def cosine_with_warmup_scheduler(optimizer: Optimizer,
                                 num_warmup_epochs: int, max_epoch: int):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_epochs,
        num_training_steps=max_epoch
    )
    return scheduler


def get_linear_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
        last_epoch: int = -1):
    """
    Implementation by Huggingface:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases linearly from the
    initial lr set in the optimizer to 0, after a warmup period during which it
    increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
        num_cycles: float = 0.5, last_epoch: int = -1):
    """
    Implementation by Huggingface:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases following the values
    of the cosine function between the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just
            decrease from the max value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
