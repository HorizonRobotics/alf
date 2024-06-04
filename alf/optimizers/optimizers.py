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
from typing import Callable, Dict, List, Union

import alf
from alf.utils import common
from alf.utils import tensor_utils
from alf.utils.schedulers import as_scheduler, ConstantScheduler, Scheduler
from . import adam_tf, adamw, nero_plus
from .utils import get_opt_arg


def _rbf_func(x):
    r"""
    Compute the rbf kernel and its gradient w.r.t. first entry
    :math:`K(x, x), \nabla_x K(x, x)`, for computing ``svgd``_grad.

    Args:
        x (Tensor): set of N particles, shape (N x D), where D is the
            dimenseion of each particle

    Returns:
        :math:`K(x, x)` (Tensor): the RBF kernel of shape (N x N)
        :math:`\nabla_x K(x, x)` (Tensor): the derivative of RBF kernel of shape (N x N x D)

    """
    N, D = x.shape
    diff = x.unsqueeze(1) - x.unsqueeze(0)  # [N, N, D]
    dist_sq = torch.sum(diff**2, -1)  # [N, N]
    h, _ = torch.median(dist_sq.view(-1), dim=0)
    if h == 0.:
        h = torch.ones_like(h)
    else:
        h = h / max(np.log(N), 1.)

    kappa = torch.exp(-dist_sq / h)  # [N, N]
    kappa_grad = -2 * kappa.unsqueeze(-1) * diff / h  # [N, N, D]
    return kappa, kappa_grad


def _score_func(x, alpha=1e-5):
    r"""
    Compute the stein estimator of the score function
    :math:`\nabla\log q = -(K + \alpha I)^{-1}\nabla K`,
    for computing ``gfsf``_grad.

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
    if h == 0.:
        h = torch.ones_like(h)
    else:
        h = h / max(np.log(N), 1.)

    kappa = torch.exp(-dist_sq / h)  # [N, N]
    kappa_inv = torch.inverse(kappa + alpha * torch.eye(N))  # [N, N]
    kappa_grad = -2 * kappa.unsqueeze(-1) * diff / h  # [N, N, D]
    kappa_grad = kappa_grad.sum(0)  # [N, D]

    return -kappa_inv @ kappa_grad


def wrap_optimizer(cls):
    """A helper function to construct torch optimizers with
    params as [{'params': []}]. After construction, new parameter
    groups can be added by using the add_param_group() method.

    This wrapper also clips gradients first before calling ``step()``.
    """
    NewClsName = cls.__name__
    NewCls = type(NewClsName, (cls, ), {})
    NewCls.counter = 0

    @common.add_method(NewCls)
    def __init__(self,
                 *,
                 gradient_clipping=None,
                 clip_by_global_norm=False,
                 parvi=None,
                 repulsive_weight=1.,
                 capacity_ratio: Union[float, Scheduler] = 1.0,
                 min_capacity: int = 8192,
                 masked_out_value: Union[float, None] = None,
                 name=None,
                 **kwargs):
        """
        Some parameter specific optimization arguments can be set using
        `opt_args' attributes of ``Parameter``. If it is set, it should be a
        dictionary of optimization arguments. It can have the following
        arguments:

            fixed_norm (bool): if True, the Frobenius norm of the parameter will
                be kept constant.
            weight_decay (float): If set, a parameter specific weight decay will
                be used instead of the default weight_decay. This is only
                supported by ``AdamTF`` optimizer.

        Args:
            gradient_clipping (float): If not None, serve as a positive threshold
            clip_by_global_norm (bool): If True, use `tensor_utils.clip_by_global_norm`
                to clip gradient. If False, use `tensor_utils.clip_by_norms` for
                each grad.
            parvi (string): if not ``None``, paramters with attribute
                ``ensemble_group`` will be updated by particle-based vi algorithm
                specified by ``parvi``, options are [``svgd``, ``gfsf``],

                * Stein Variational Gradient Descent (SVGD)

                  Liu, Qiang, and Dilin Wang. "Stein Variational Gradient Descent:
                  A General Purpose Bayesian Inference Algorithm." NIPS. 2016.

                * Wasserstein Gradient Flow with Smoothed Functions (GFSF)

                  Liu, Chang, et al. "Understanding and accelerating particle-based
                  variational inference." ICML. 2019.

                To work with the ``parvi`` option, the parameters added to the
                optimizer (by ``add_param_group``) should have an (int) attribute
                ``ensemble_group``. See ``FCBatchEnsemble`` as an example.

            repulsive_weight (float): the weight of the repulsive gradient term
                for parameters with attribute ``ensemble_group``.
            capacity_ratio: For each parameter, `numel() * capacity_ratio`
                elements are turned on for training. The remaining elements
                are frozen. ``capacity_ratio`` can be a scheduler to control
                the capacity over the training process.
                Note that capacity_ratio scheduling does not support the mixed precision
                case and should not be used under that setting.
            min_capacity: For each parameter, at least so many elements
                are turned on for training.
            masked_out_value: the value to be set for the masked out parameters, i.e.,
                parameters whose mask value is True. If None, no operation will be applied.
                Otherwise, set the parameter values as the specified value.
            name (str): the name displayed when summarizing the gradient norm. If
                None, then a global name in the format of "class_name_i" will be
                created, where "i" is the global optimizer id.
            kwargs: arguments passed to the constructor of the underlying torch
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
        self._lr_schedulers = []

        self._capacity_ratio = alf.utils.schedulers.as_scheduler(
            capacity_ratio)
        self._random_number_generator = torch.Generator(
            alf.get_default_device())
        self._rng_state_device = self._random_number_generator.get_state(
        ).device

        super(NewCls, self).__init__([{'params': []}], **kwargs)
        if gradient_clipping is not None:
            self.defaults['gradient_clipping'] = gradient_clipping
            self.defaults['clip_by_global_norm'] = clip_by_global_norm
        self._gradient_clipping = gradient_clipping
        self._clip_by_global_norm = clip_by_global_norm
        self._parvi = parvi
        self._first_stepping_done = False  # whether done the first optimizer stepping
        self._min_capacity = min_capacity
        self._masked_out_value = masked_out_value
        self._norms = {}  # norm of each parameter
        if parvi is not None:
            assert parvi in ['svgd', 'gfsf'
                             ], ("parvi method %s is not supported." % (parvi))
            self.defaults['parvi'] = parvi
            self.defaults['repulsive_weight'] = repulsive_weight
            self._repulsive_weight = repulsive_weight
        self.name = name
        if name is None:
            self.name = NewClsName + "_" + str(NewCls.counter)
            NewCls.counter += 1

    @common.add_method(NewCls)
    def _clone_params(self, capacity_ratio: float):
        """clone the parameters

        Args:
            capacity_ratio: the capacity ration specifying the ratio of learnable parameters
                to the number of all parameters. Only copy if ``capacity_ratio < 1``
                and ``masked_out_value`` is None.
        Returns:
            an empty dictionary if ``self._masked_out_value`` is None; otherwise, return
            a dictionary with the parameter as the key and the current parameter value as
            the value of the dictionary.
        """
        param_values = {}
        if capacity_ratio < 1 and self._masked_out_value is None:
            for param_group in self.param_groups:
                for p in param_group['params']:
                    # only save previous param value if masked_out_value is unspecified
                    param_values[p] = p.data.clone()
        return param_values

    @common.add_method(NewCls)
    def _adjust_capacity(self, capacity_ratio: float, old_param_values: Dict):
        """adjust the capacity by apply reassigning old parameter values to the
        parameter according to the capacity mask.

        Args:
            capacity_ratio: the capacity ration specifying the ratio of learnable parameters
                to the number of all parameters
            old_param_values: a dictionary with the parameter as the key and the parameter
                value (e.g. from the previous iteration)  as the dictionary value.
                These values will be used to reset the corresponding parameter values
                after each gradient step if the corresonding capacity mask is True,
                effectively excluding it from learning.
        """
        if capacity_ratio < 1:
            common.warning_once(
                'Capacity scheduling is used. If using DDP with world size larger than '
                'one in training, suggest to explicitly set random_seed for training '
                'in order to make sure the capacity scheduling work as expected \n'
            )

            # To achieve this, we assign a random number for each element of
            # the parameter. An element is turned on if its assigned random number
            # is less than capacity_ratio. To save memory, we don't store the
            # random numbers. Instead, we save the random number generator state.
            for param_group in self.param_groups:
                for p in param_group['params']:
                    state = self.state[p]
                    old_param_val = old_param_values.get(p, None)
                    # get, save and set random number generator state
                    if 'rng_state' not in state:
                        # record random number generator state in ``self.state``
                        state[
                            'rng_state'] = self._random_number_generator.get_state(
                            )
                    else:
                        self._random_number_generator.set_state(
                            state['rng_state'])

                    # generate capacity mask using the same random number generator state
                    n = p.numel()
                    ratio = max(self._min_capacity / n, capacity_ratio)
                    mask = torch.rand(
                        p.shape,
                        generator=self._random_number_generator) >= ratio

                    if self._masked_out_value is None:
                        if old_param_val is not None:
                            # The following is faster than p.data[mask] = old_param[mask]
                            p.data.copy_(
                                torch.where(mask, old_param_val, p.data))
                            del old_param_val
                    else:
                        p.data[mask] = self._masked_out_value

    @common.add_method(NewCls)
    def _remove_customized_states(self,
                                  customized_keys: List[str] = ['rng_state']):
        """extract and remove the customized states (e.g. ``state['rng_state']``) from the
        optimizer's state attributes (``self.state``)

        Args:
            customized_keys: a list of keys for the customized states

        Returns:
            a dictionary with the parameter as the key and the customized states value
        """
        customized_state = {}
        for param_group in self.param_groups:
            for p in param_group['params']:
                state = self.state[p]
                customized_state[p] = {
                    key: state[key]
                    for key in customized_keys
                }
                [state.pop(key) for key in customized_keys]
        self._customized_state = customized_state
        return customized_state

    @common.add_method(NewCls)
    def _append_customized_states(self, customized_state: Dict):
        """append the customized states to the optimizer's state attributes (``self.state``)

        Args:
            customized_state: a dictionary of customized state
        """
        for param_group in self.param_groups:
            for p in param_group['params']:
                state = self.state[p]
                state.update(customized_state[p])

    @common.add_method(NewCls)
    def step(self, closure=None):
        """This function first clips the gradients if needed, then call the
        parent's ``step()`` function.
        """
        if self._lr_scheduler is not None:
            lr = float(self._lr_scheduler())
            for i, (lr_scheduler, param_group) in enumerate(
                    zip(self._lr_schedulers, self.param_groups)):
                if lr_scheduler is not None:
                    param_group['lr'] = lr_scheduler()
                    if alf.summary.should_record_summaries():
                        alf.summary.scalar("lr/%s/%s" % (self.name, i), lr)
                else:
                    param_group['lr'] = lr
            if alf.summary.should_record_summaries():
                alf.summary.scalar("lr/%s" % self.name, lr)
        params = []
        for param_group in self.param_groups:
            params.extend(param_group["params"])

        if not isinstance(self, NeroPlus):
            for param in params:
                if (get_opt_arg(param, 'fixed_norm', False)
                        and param not in self._norms):
                    self._norms[param] = param.norm()

        if self._gradient_clipping is not None:
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

        if self._parvi is not None:
            self._parvi_step()

        capacity_ratio = self._capacity_ratio()

        param_values = self._clone_params(capacity_ratio)

        if not self._first_stepping_done and capacity_ratio < 1:
            # if no stepping done yet, remove customized states to avoid any impacts on
            # some specific optimizers' internal procedures, e.g. lazy state initialization
            # https://github.com/pytorch/pytorch/blob/b2e0f8d82d721f6598113f64a408b0017bb90cb2/torch/optim/adam.py#L102-L103
            customized_states = self._remove_customized_states()
        else:
            customized_states = {}

        super(NewCls, self).step(closure=closure)

        if len(customized_states):
            self._append_customized_states(customized_states)

        if not isinstance(self, NeroPlus):
            for param in params:
                if param.grad is not None and get_opt_arg(
                        param, 'fixed_norm', False):
                    param.data.mul_(
                        self._norms[param] / (param.norm() + 1e-30))

        self._adjust_capacity(capacity_ratio, param_values)

        if capacity_ratio < 1:
            if alf.summary.should_record_summaries():
                alf.summary.scalar("capacity_ratio", capacity_ratio)

        self._first_stepping_done = True

    @common.add_method(NewCls)
    def _parvi_step(self):
        for param_group in self.param_groups:
            if "parvi_grad" in param_group:
                params = param_group['params']
                batch_size = params[0].shape[0]
                params_tensor = torch.cat(
                    [p.view(batch_size, -1) for p in params],
                    dim=-1)  # [N, D], D=dim(params)
                if self._parvi == 'svgd':
                    # [N, N], [N, N, D]
                    kappa, kappa_grad = _rbf_func(params_tensor)
                    grads_tensor = torch.cat(
                        [p.grad.view(batch_size, -1) for p in params],
                        dim=-1).detach()  # [N, D]
                    kernel_logp = torch.matmul(kappa,
                                               grads_tensor) / batch_size
                    svgd_grad = torch.split(
                        kernel_logp -
                        self._repulsive_weight * kappa_grad.mean(0),
                        [p.nelement() // batch_size for p in params],
                        dim=-1)
                    for i in range(len(params)):
                        grad = params[i].grad.view(batch_size, -1)
                        grad.copy_(svgd_grad[i])
                else:
                    logq_grad = _score_func(params_tensor)  # [N, D]
                    gfsf_grad = torch.split(
                        logq_grad,
                        [p.nelement() // batch_size for p in params],
                        dim=-1)
                    for i in range(len(params)):
                        grad = params[i].grad.view(batch_size, -1)
                        grad.add_(self._repulsive_weight * gfsf_grad[i])

    @common.add_method(NewCls)
    def add_param_group(self, param_group):
        """This function first splits the input param_group into multiple
        param_groups according to their ``ensemble_group`` attributes, then
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

        lr_scheduler = param_group.get('lr_scheduler', None)
        if isinstance(lr_scheduler, Callable):
            self._lr_schedulers.append(lr_scheduler)
            param_group['lr'] = lr_scheduler()
        else:
            self._lr_schedulers.append(None)

        len_params = len(param_group['params'])
        std_param_group = []
        ensemble_param_groups = [[] for i in range(len_params)]
        group_batch_sizes = [0] * len_params
        for param in param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " +
                                torch.typename(param))
            if hasattr(param, 'ensemble_group'):
                assert isinstance(
                    param.ensemble_group,
                    int), ("ensemble_group attribute mis-specified.")
                ensemble_group_id = param.ensemble_group
                if group_batch_sizes[ensemble_group_id] == 0:
                    group_batch_sizes[ensemble_group_id] = param.shape[0]
                else:
                    assert param.shape[0] == group_batch_sizes[
                        ensemble_group_id], (
                            "batch_size of params does not match that of the "
                            "ensemble param_group %d." % (ensemble_group_id))
                ensemble_param_groups[ensemble_group_id].append(param)
            else:
                std_param_group.append(param)

        if len(alf.nest.flatten(ensemble_param_groups)) > 0:
            if len(std_param_group) > 0:
                super(NewCls, self).add_param_group({
                    'params': std_param_group
                })
            for ensemble_param_group in ensemble_param_groups:
                if len(ensemble_param_group) > 0:
                    super(NewCls, self).add_param_group({
                        'params': ensemble_param_group,
                        'parvi_grad': True
                    })
        else:
            super(NewCls, self).add_param_group(param_group)

        capacity_ratio = self._capacity_ratio()
        self._adjust_capacity(capacity_ratio, {})

    @common.add_method(NewCls)
    def load_state_dict(self, state_dict):
        """
        Byte type is required for ``rng_state`` but within optimizer.load_state_dict,
        it is converted to float.
        This function first call the parent's ``load_state_dict()`` function, and
        then make sure the ``rng_state`` is in correct data type.
        """
        super(NewCls, self).load_state_dict(state_dict)
        if len(self.state) > 0:
            for param_group in self.param_groups:
                for p in param_group['params']:
                    state = self.state[p]
                    if 'rng_state' in state:
                        state['rng_state'] = state['rng_state'].to(
                            self._rng_state_device).byte()

    @common.add_method(NewCls)
    def __setstate__(self, state):
        for param_group in self.param_groups:
            for p in param_group['params']:
                p_state = self.state[p]
                if 'step' not in p_state:
                    # Some of the optimizers such as Adam/AdamW/ASGD etc in higher version
                    # torch require 1) the presence of 'step' in state, and 2) it appears
                    # as the first key of the state dictionary. Therefore we explicitly
                    # create it if it does not exist and make it the key that comes first
                    self.state[p] = {'step': 0, **p_state}

        super(NewCls, self).__setstate__(state)

    return NewCls


Adam = alf.repr_wrapper(
    alf.configurable('Adam')(wrap_optimizer(torch.optim.Adam)))

if torch.__version__ >= '1.8.1':
    AdamW = alf.configurable('AdamW')(wrap_optimizer(torch.optim.AdamW))
else:
    AdamW = alf.repr_wrapper(
        alf.configurable('AdamW')(wrap_optimizer(adamw.AdamW)))

SGD = alf.repr_wrapper(
    alf.configurable('SGD')(wrap_optimizer(torch.optim.SGD)))

AdamTF = alf.repr_wrapper(
    alf.configurable('AdamTF')(wrap_optimizer(adam_tf.AdamTF)))

NeroPlus = alf.repr_wrapper(
    alf.configurable('NeroPlus')(wrap_optimizer(nero_plus.NeroPlus)))
