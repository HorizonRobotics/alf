# Copyright (c) 2022 Horizon Robotics and ALF Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NeroPlus optimizer."""

import math
import torch
from torch.optim import Optimizer
from typing import Tuple, Optional
from .utils import get_opt_arg


def _norm(x):
    if x.ndim > 1:
        n = x.reshape(x.shape[0], -1).norm(dim=1)
        return n.reshape(x.shape[0], *([1] * (x.ndim - 1)))
    else:
        return x.norm()


def _normalize(p,
               max_norm: Optional[float] = 1,
               fixed_norm: bool = True,
               zero_mean: bool = True):
    if zero_mean:
        if p.ndim > 1:
            p.data -= p.mean(dim=tuple(range(1, p.ndim)), keepdim=True)
        elif p.ndim == 1:
            p.data -= p.mean()
    if max_norm != math.inf:
        scale = max_norm / (_norm(p) + 1e-30)
        if not fixed_norm:
            scale.clamp_(max=1)
        p.data *= scale


def _get_opt_args(p,
                  max_norm: float = 1,
                  fixed_norm: bool = True,
                  zero_mean: bool = True):
    if p.ndim > 1:
        max_norm = get_opt_arg(p, 'max_norm', max_norm)
        fixed_norm = get_opt_arg(p, 'fixed_norm', fixed_norm)
        zero_mean = get_opt_arg(p, 'zero_mean', zero_mean)
    elif p.ndim == 1:
        zero_mean = get_opt_arg(p, 'zero_mean', False)
        fixed_norm = get_opt_arg(p, 'fixed_norm', False)
        max_norm = get_opt_arg(p, 'max_norm', math.inf)
    return max_norm, fixed_norm, zero_mean


class NeroPlus(Optimizer):
    r"""NeroPlus Optimizer

    This is an enhanced version of the Nero optimizer described in the following
    paper:

    `Yang Liu et. al. Learning by Turning: Neural Architecture Aware Optimisation
    <https://arxiv.org/abs/2102.07227>`_

    The essence of this optimizer is to keep the norm of each parameter vector
    fixed and mean at zero during the optimization process. The parameter vector
    is defined as the part of parameter responsible for one dimension of the
    output. For example, ``FC(m, n)`` have two parameters, its weight of shape
    [m, n] and its bias of shape [n]. Its weight have m parameter vectors. Each
    of these m vectors is subject to the norm and mean constraint. For the bias,
    one element is responsible for one output dimension. So it is not subject to the
    norm and zero-mean constraint. Since the range of the output of a model should
    not be constrained, you should set opt_config for the output layers as
    `dict(fixed_norm=False, max_norm=math.inf, zero_mean=False)` or use a large finite ``max_norm``
    or ``weight_decay`` to introduce some regularization.

    For 2+ D parameter p, its parameter vectors are assumed to be p[0], p[1] ... p[-1].
    This is correct for many ALF layers (e.g. FC, Conv2D, TransformerBlock). But
    not all of ALF layers follows this rule. ``ParallelFC`` and other parallel
    layers are such examples. So you should not use NeroPlus if your model contains
    such layers.

    By default, 1D parameters are not subject to the constraint (i.e. max_norm=math.inf,
    fixed_norm=False, zero_mean=False). If the constraints are desired, they can
    be specified using ``opt_args`` attribute of ``Parameter``.

    The main enhancements compared to the original Nero optimizer include:

    1. Option for ADAM like update (normalizing_grad_by_norm=False)
    2. Upper bound constraint of weight norm (fixed_norm=False)
    3. Weight decay and L2 regularization.

    To use this optimizer, you should first use ``NeroPlus.initialize()`` to normalize
    the parameter of your model for the given constrains before actually using
    your model for training.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: learning rate
        betas: coefficients between 0 and 1. They are used for computing running
            averages of gradient and its squared norm or elementwise square.
            ``betas[0]`` can be zero, in which case no running average will be
            performed. ``betas[1]`` must be greater than 0.
        eps: term added to the denominator to improve numerical stability which
            corresponds to the epsilon_hat in the Adam paper.
        normalizing_grad_by_norm: whether to normalize the gradient by the running
            average of its squared norm or its elementwise square. Note that
            the original Nero optimizer uses ``True`` for this. However, we found
            the ADAM like behavior is better.
        max_norm: maximal norm of each parameter vector. A parameter vector is
            part of a parameter responsible for one output dimension.
        weight_decay: weight decay. This is same as the weight decay of AdamW,
            which is implemented as subtracting `lr * weight_deday * w` from
            parameter.
        l2_regularization: L2 penalty. This is same as the weight decay of Adam,
            which is implemented as adding `weight_decay * w` to gradient.
        fixed_norm: whether to fix the norm of the parameter vector. If True,
            the norm will be fixed at ``max_norm``.
        zero_mean: whether to enfoce the mean of a parameter vector is zero.

    ``lr``, ``weight_decay``, ``l2_regularization``, ``fixed_norm``, ``max_norm``, ``zero_mean`` can
    be set individually for each parameter using ``opt_args`` attributes of
    ``Parameter``. ``opt_args`` should be a dictionary. Additionally, ``lr_scale``
    which can be used to scale the global learning for a specific parameter.

    """

    def __init__(self,
                 params=[{
                     'params': []
                 }],
                 lr: float = 0.01,
                 betas: Tuple[float] = (0.9, 0.999),
                 eps: float = 1e-7,
                 normalizing_grad_by_norm=False,
                 max_norm: float = 1,
                 weight_decay: float = 0,
                 l2_regularization: float = 0,
                 fixed_norm: bool = True,
                 zero_mean: bool = True):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            normalizing_grad_by_norm=normalizing_grad_by_norm,
            weight_decay=weight_decay,
            l2_regularization=l2_regularization,
            max_norm=max_norm,
            fixed_norm=fixed_norm,
            zero_mean=zero_mean)
        super().__init__(params, defaults)
        assert 0 <= betas[0] < 1, ("Invalid value for betas[0]=%s" % betas[0])
        assert 0 < betas[1] < 1, ("Invalid value for betas[1]=%s" % betas[1])

    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        max_norm = param_group['max_norm']
        fixed_norm = param_group['fixed_norm']
        zero_mean = param_group['zero_mean']
        beta1, beta2 = param_group['betas']
        normalizing_grad_by_norm = param_group['normalizing_grad_by_norm']
        for p in param_group['params']:
            pmax_norm, pfixed_norm, pzero_mean = _get_opt_args(
                p, max_norm, fixed_norm, zero_mean)
            norm = _norm(p)
            if p.ndim > 1:
                if pzero_mean:
                    mean = p.mean(dim=tuple(range(1, p.ndim)))
                    i = mean.abs().argmax()
                    assert mean[i].abs() < 0.01, (
                        "Unnormalized parameter: mean()=%s"
                        "Model should be initialized using NeroPlus.initialize()"
                        % mean[i].item())
                if pmax_norm != math.inf:
                    diff = norm / pmax_norm - 1
                    if pfixed_norm:
                        diff.abs_()
                    i = diff.argmax()
                    assert diff[i] < 0.01, (
                        "Unnormalized parameter: norm=%s. "
                        "Model should be initialized using NeroPlus.initialize()"
                        % norm[i].item())
            state = self.state[p]
            state['step'] = 0
            if normalizing_grad_by_norm:
                state['exp_avg_sq'] = torch.zeros_like(norm)
            else:
                state['exp_avg_sq'] = torch.zeros_like(p)
            if beta1 > 0:
                state['exp_avg'] = torch.zeros_like(p)

    @staticmethod
    def initialize(model: torch.nn.Module,
                   max_norm: float = 1,
                   fixed_norm: bool = True,
                   zero_mean: bool = True):
        for p in model.parameters():
            pmax_norm, pfixed_norm, pzero_mean = _get_opt_args(
                p, max_norm, fixed_norm, zero_mean)
            _normalize(p, pmax_norm, pfixed_norm, pzero_mean)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            base_lr = group['lr']
            max_norm = group['max_norm']
            fixed_norm = group['fixed_norm']
            zero_mean = group['zero_mean']
            normalizing_grad_by_norm = group['normalizing_grad_by_norm']
            eps = group['eps']
            weight_decay = group['weight_decay']
            l2_regularization = group['l2_regularization']
            bias_correction1 = 1
            for p in group['params']:
                if p.grad is None:
                    continue
                pmax_norm, pfixed_norm, pzero_mean = _get_opt_args(
                    p, max_norm, fixed_norm, zero_mean)
                lr = get_opt_arg(p, 'lr', base_lr)
                lr_scale = get_opt_arg(p, 'lr_scale', 1)
                state = self.state[p]
                state['step'] += 1
                bias_correction2 = 1 - beta2**state['step']
                grad = p.grad
                pweight_decay = get_opt_arg(p, 'weight_decay', weight_decay)
                pl2_regularization = get_opt_arg(p, 'l2_regularization',
                                                 l2_regularization)
                if pl2_regularization != 0:
                    grad = grad.add(p, alpha=pl2_regularization)
                if beta1 > 0:
                    bias_correction1 = 1 - beta1**state['step']
                    exp_avg = state['exp_avg']
                    exp_avg.lerp_(grad, 1 - beta1)
                else:
                    exp_avg = grad
                if normalizing_grad_by_norm:
                    sq = _norm(grad)**2
                else:
                    sq = grad**2
                state['exp_avg_sq'].lerp_(sq, 1 - beta2)
                denom = state['exp_avg_sq'].sqrt().add_(eps)
                if pweight_decay > 0:
                    p.mul_(1 - lr_scale * lr * pweight_decay)
                # the exponential moving average of exp_avg and exp_avg_sq are not
                # unbiased estimate of the mean. Correct them using bas_correction1
                # and bias_correct2 as suggest by the original Adam paper.
                step_size = lr_scale * lr * math.sqrt(
                    bias_correction2) / bias_correction1
                # p <- p  - step_size * exp_avg / denom
                p.addcdiv_(exp_avg, denom, value=-step_size)
                _normalize(p, pmax_norm, pfixed_norm, pzero_mean)

        return loss
