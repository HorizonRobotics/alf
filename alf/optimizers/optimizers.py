# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
import numpy as np
import torch
from typing import Callable

import alf
from alf.utils import common
from alf.utils import tensor_utils
from . import adam_tf


def _score_func(x, alpha=1e-5):
    r"""
    Compute the stein estimator of the score function
    :math:`\nabla\log q = -(K + \alpha I)^{-1}\nabla K`,
    used by gfsf_grad.

    Args:
        x (Tensor): set of N particles, shape (N x D), where D is the
            dimenseion of each particle
        alpha (float): weight of regularization for inverse kernel
            this parameter turns out to be crucial for convergence.

    Returns:
        :math:`\nabla\log q` (Tensor): the score function of shape (N x D)

    """
    N, D = x.shape
    diff = x.unsqueeze(1) - x.unsqueeze(0)  # [N, N, D]
    dist_sq = torch.sum(diff**2, -1)  # [N, N]
    h, _ = torch.median(dist_sq.view(-1), dim=0)
    h = h / np.log(N)

    kappa = torch.exp(-dist_sq / h)  # [N, N]
    kappa_inv = torch.inverse(kappa + alpha * torch.eye(N))  # [N, N]
    kappa_grad = torch.einsum('ij,ijk->jk', kappa, -2 * diff / h)  # [N, D]

    return -kappa_inv @ kappa_grad


def wrap_optimizer(cls):
    """A helper function to construct torch optimizers with
    params as [{'params': []}]. After construction, new parameter
    groups can be adde by using the add_param_group() method.

    This wrapper also clips gradients first before calling ``step()``.
    """
    NewClsName = cls.__name__ + "_"
    NewCls = type(NewClsName, (cls, ), {})
    NewCls.counter = 0

    @common.add_method(NewCls)
    def __init__(self,
                 gradient_clipping=None,
                 clip_by_global_norm=False,
                 gfsf_grad_weight=False,
                 name=None,
                 **kwargs):
        """
        Args:
            gradient_clipping (float): If not None, serve as a positive threshold
            clip_by_global_norm (bool): If True, use `tensor_utils.clip_by_global_norm`
                to clip gradient. If False, use `tensor_utils.clip_by_norms` for
                each grad.
            gfsf_grad_weight (float): If not None, denotes the weight of the repulsive
                gradient term for parameters with attribute ``gfsf``.
            name (str): the name displayed when summarizing the gradient norm. If
                None, then a global name in the format of "class_name_i" will be
                created, where "i" is the global optimizer id.
            kwargs: arguments passed to the constructor of the underline torch
                optimizer. If ``lr`` is given and it is a ``Callable``, it is
                treated as a learning rate scheduler and will be called everytime
                when ``step()`` is called to get the latest learning rate.
                Available schedulers are in ``alf.utils.schedulers``.
        """
        self._lr_scheduler = None
        if "lr" in kwargs:
            lr = kwargs["lr"]
            if isinstance(lr, Callable):
                self._lr_scheduler = lr
                kwargs["lr"] = float(lr())

        super(NewCls, self).__init__([{'params': []}], **kwargs)
        self._gradient_clipping = gradient_clipping
        self._clip_by_global_norm = clip_by_global_norm
        self._gfsf_grad_weight = gfsf_grad_weight
        self.name = name
        if name is None:
            self.name = NewClsName + str(NewCls.counter)
            NewCls.counter += 1

    @common.add_method(NewCls)
    def step(self, closure=None):
        """This function first clips the gradients if needed, then call the
        parent's ``step()`` function.
        """
        if self._lr_scheduler is not None:
            lr = float(self._lr_scheduler())
            for param_group in self.param_groups:
                param_group['lr'] = lr
        if self._gradient_clipping is not None:
            params = []
            for param_group in self.param_groups:
                params.extend(param_group["params"])
            grads = alf.nest.map_structure(lambda p: p.grad, params)
            if self._clip_by_global_norm:
                _, global_norm = tensor_utils.clip_by_global_norm(
                    grads, self._gradient_clipping, in_place=True)
                if alf.summary.should_record_summaries():
                    alf.summary.scalar("global_grad_norm/%s" % self.name,
                                       global_norm)
            else:
                tensor_utils.clip_by_norms(
                    grads, self._gradient_clipping, in_place=True)

        if self._gfsf_grad_weight is not None:
            for param_group in self.param_groups:
                if "gfsf_grad" in param_group:
                    params = param_group['params']
                    batch_size = params[0].shape[0]
                    params_tensor = torch.cat(
                        [p.view(batch_size, -1) for p in params],
                        dim=-1)  # [N, dim(params)]
                    logq_grad = _score_func(params_tensor)  # [N, dim(param)]
                    gfsf_grad = torch.split(
                        logq_grad,
                        [int(p.nelement() / batch_size) for p in params],
                        dim=-1)
                    for i in range(len(params)):
                        grad = params[i].grad.view(batch_size, -1)
                        grad.add_(self._gfsf_grad_weight * gfsf_grad[i])

        super(NewCls, self).step(closure=closure)

    @common.add_method(NewCls)
    def add_param_group(self, param_group):
        """This function first splits the input param_group into multiple
        param_groups according to their ``gfsf_group`` attributes, then 
        calls the parent's ``add_param_group()`` function to add each of
        them to the optimizer.
        """
        assert isinstance(param_group, dict), "param_group must be a dict"

        params = param_group["params"]
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('Please use a list instead.')
        else:
            param_group['params'] = list(params)

        len_params = len(param_group['params'])
        std_param_group = []
        gfsf_param_groups = [[] for i in range(len_params)]
        bs_gfsf_param_groups = torch.zeros(len_params)
        for param in param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " +
                                torch.typename(param))
            if hasattr(param, 'gfsf_group'):
                assert isinstance(param.gfsf_group,
                                  int), ("gfsf_group attribute mis-specified.")
                gfsf_group_id = param.gfsf_group % len_params
                if bs_gfsf_param_groups[gfsf_group_id] == 0:
                    bs_gfsf_param_groups[gfsf_group_id] = param.shape[0]
                else:
                    assert param.shape[0] == bs_gfsf_param_groups[
                        gfsf_group_id], (
                            "batch_sizes does not match that of the gfsf "
                            "param_group %d." % (gfsf_group_id))
                gfsf_param_groups[gfsf_group_id].append(param)
            else:
                std_param_group.append(param)

        if len(std_param_group) > 0:
            super(NewCls, self).add_param_group({'params': std_param_group})
        for gfsf_param_group in gfsf_param_groups:
            if len(gfsf_param_group) > 0:
                super(NewCls, self).add_param_group({
                    'params': gfsf_param_group,
                    'gfsf_grad': True
                })

    return NewCls


Adam = alf.configurable('Adam')(wrap_optimizer(torch.optim.Adam))

AdamW = alf.configurable('AdamW')(wrap_optimizer(torch.optim.AdamW))

SGD = alf.configurable('SGD')(wrap_optimizer(torch.optim.SGD))

AdamTF = alf.configurable('AdamTF')(wrap_optimizer(adam_tf.AdamTF))
