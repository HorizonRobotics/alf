# Copyright (c) 2024 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import copy
import itertools
import torch
import torch.nn as nn


def _avg_fn(averaged_model_parameter, model_parameter, num_averaged):
    return torch.lerp(model_parameter, averaged_model_parameter,
                      1 / (num_averaged + 1))


class AveragedModel(nn.Module):
    # Note: torch.optim.swa_utils.AveragedModel in torch 1.13 has a bug of not
    # copying buffers for use_buffers=False. So we copy the implementation
    # from torch 2.2 here.
    r"""Implements averaged model for Stochastic Weight Averaging (SWA).

    Stochastic Weight Averaging was proposed in `Averaging Weights Leads to
    Wider Optima and Better Generalization`_ by Pavel Izmailov, Dmitrii
    Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson
    (UAI 2018).

    AveragedModel class creates a copy of the provided module :attr:`model`
    on the device :attr:`device` and allows to compute running averages of the
    parameters of the :attr:`model`.

    Args:
        model (torch.nn.Module): model to use with SWA
        device (torch.device, optional): if provided, the averaged model will be
            stored on the :attr:`device`
        avg_fn (function, optional): the averaging function used to update
            parameters; the function must take in the current value of the
            :class:`AveragedModel` parameter, the current value of :attr:`model`
            parameter and the number of models already averaged; if None,
            equally weighted average is used (default: None)
        use_buffers (bool): if ``True``, it will compute running averages for
            both the parameters and the buffers of the model. (default: ``False``)

    Example:
        >>> # xdoctest: +SKIP("undefined variables")
        >>> loader, optimizer, model, loss_fn = ...
        >>> swa_model = torch.optim.swa_utils.AveragedModel(model)
        >>> scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        >>>                                     T_max=300)
        >>> swa_start = 160
        >>> swa_scheduler = SWALR(optimizer, swa_lr=0.05)
        >>> for i in range(300):
        >>>      for input, target in loader:
        >>>          optimizer.zero_grad()
        >>>          loss_fn(model(input), target).backward()
        >>>          optimizer.step()
        >>>      if i > swa_start:
        >>>          swa_model.update_parameters(model)
        >>>          swa_scheduler.step()
        >>>      else:
        >>>          scheduler.step()
        >>>
        >>> # Update bn statistics for the swa_model at the end
        >>> torch.optim.swa_utils.update_bn(loader, swa_model)

    You can also use custom averaging functions with `avg_fn` parameter.
    If no averaging function is provided, the default is to compute
    equally-weighted average of the weights.

    Example:
        >>> # xdoctest: +SKIP("undefined variables")
        >>> # Compute exponential moving averages of the weights and buffers
        >>> ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (
        ...                 0.1 * averaged_model_parameter + 0.9 * model_parameter)
        >>> swa_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg, use_buffers=True)

    .. note::
        When using SWA with models containing Batch Normalization you may
        need to update the activation statistics for Batch Normalization.
        This can be done either by using the :meth:`torch.optim.swa_utils.update_bn`
        or by setting :attr:`use_buffers` to `True`. The first approach updates the
        statistics in a post-training step by passing data through the model. The
        second does it during the parameter update phase by averaging all buffers.
        Empirical evidence has shown that updating the statistics in normalization
        layers increases accuracy, but you may wish to empirically test which
        approach yields the best results in your problem.

    .. note::
        :attr:`avg_fn` is not saved in the :meth:`state_dict` of the model.

    .. note::
        When :meth:`update_parameters` is called for the first time (i.e.
        :attr:`n_averaged` is `0`) the parameters of `model` are copied
        to the parameters of :class:`AveragedModel`. For every subsequent
        call of :meth:`update_parameters` the function `avg_fn` is used
        to update the parameters.

    .. _Averaging Weights Leads to Wider Optima and Better Generalization:
        https://arxiv.org/abs/1803.05407
    .. _There Are Many Consistent Explanations of Unlabeled Data: Why You Should
        Average:
        https://arxiv.org/abs/1806.05594
    .. _SWALP: Stochastic Weight Averaging in Low-Precision Training:
        https://arxiv.org/abs/1904.11943
    .. _Stochastic Weight Averaging in Parallel: Large-Batch Training That
        Generalizes Well:
        https://arxiv.org/abs/2001.02312
    """

    def __init__(self, model, device=None, avg_fn=None, use_buffers=False):
        super().__init__()
        self.module = copy.deepcopy(model)
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer('n_averaged',
                             torch.tensor(0, dtype=torch.long, device=device))
        if avg_fn is None:

            def avg_fn(averaged_model_parameter, model_parameter,
                       num_averaged):
                return averaged_model_parameter + \
                    (model_parameter - averaged_model_parameter) / (num_averaged + 1)

        self.avg_fn = avg_fn
        self.use_buffers = use_buffers

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def update_parameters(self, model):
        self_param = (itertools.chain(self.module.parameters(),
                                      self.module.buffers())
                      if self.use_buffers else self.parameters())
        model_param = (itertools.chain(model.parameters(), model.buffers())
                       if self.use_buffers else model.parameters())
        for p_swa, p_model in zip(self_param, model_param):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(
                    self.avg_fn(p_swa.detach(), p_model_,
                                self.n_averaged.to(device)))
        if not self.use_buffers:
            # If not apply running averages to the buffers,
            # keep the buffers in sync with the source model.
            for b_swa, b_model in zip(self.module.buffers(), model.buffers()):
                b_swa.detach().copy_(b_model.detach().to(device))
        self.n_averaged += 1


class DoubleAveragedModel(nn.Module):
    """
    Use double average to compensate the limited float precision.
    """

    def __init__(self, model, device=None, use_buffers=False):
        super().__init__()
        self._averaged_model = AveragedModel(model, device, _avg_fn,
                                             use_buffers)
        self._double_averaged_model = AveragedModel(
            self._averaged_model, device, _avg_fn, use_buffers)
        self.register_buffer("_double_average_period",
                             torch.tensor(8, dtype=torch.int64))

    def forward(self, *args, **kwargs):
        if self._double_averaged_model.n_averaged == 0:
            return self._averaged_model(*args, **kwargs)
        else:
            return self._double_averaged_model(*args, **kwargs)

    def update_parameters(self, model):
        self._averaged_model.update_parameters(model)
        if self._averaged_model.n_averaged == self._double_average_period:
            self._double_averaged_model.update_parameters(self._averaged_model)
            self._averaged_model.n_averaged.copy_(0)
            if self._double_averaged_model.n_averaged == self._double_average_period:
                self._double_average_period *= 2
                self._double_averaged_model.n_averaged //= 2


def create_averaged_model(model, average_type, device=None, use_buffers=False):
    if average_type == "simple":
        return AveragedModel(
            model, device=device, avg_fn=_avg_fn, use_buffers=use_buffers)
    elif average_type == "double":
        return DoubleAveragedModel(
            model, device=device, use_buffers=use_buffers)
    else:
        assert average_type == "none"
        return model
