# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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
"""Utility functions for generate summary."""
from absl import logging
import functools
import numpy as np
import os
import time
import torch
import torch.distributions as td

import alf
from alf.data_structures import LossInfo
from alf.nest import is_namedtuple, is_nested, py_map_structure_with_path, map_structure
from alf.utils import dist_utils
from alf.summary import should_record_summaries, get_global_counter
from typing import List, Optional

DEFAULT_BUCKET_COUNT = 30


def _summary_wrapper(summary_func):
    """Summary wrapper

    Wrapper summary function to reduce cost for data computation
    """

    @functools.wraps(summary_func)
    def wrapper(*args, **kwargs):
        if should_record_summaries():
            summary_func(*args, **kwargs)

    return wrapper


@_summary_wrapper
def histogram_discrete(name, data, bucket_min, bucket_max, step=None):
    """histogram for discrete data.

    Args:
        name (str): name for this summary
        data (Tensor): A ``Tensor`` integers of any shape.
        bucket_min (int): represent bucket min value
        bucket_max (int): represent bucket max value
            bucket count is calculate as ``bucket_max - bucket_min + 1``
            and output will have this many buckets.
        step (None|Tensor): step value for this summary. this defaults to
            ``alf.summary.get_global_counter()``
    """
    bins = torch.arange(bucket_min, bucket_max + 1).cpu()
    # For N bins, there should be N+1 bin edges
    bin_edges = bins.to(torch.float32) - 0.5
    bin_edges = torch.cat([bin_edges, bin_edges[-1:] + 1.])
    alf.summary.histogram(name, data, step=step, bins=bin_edges)


@_summary_wrapper
def histogram_continuous(name,
                         data,
                         bucket_min=None,
                         bucket_max=None,
                         bucket_count=DEFAULT_BUCKET_COUNT,
                         edge_inclusive=True,
                         step=None):
    """histogram for continuous data.

    Args:
        name (str): name for this summary
        data (Tensor): A ``Tensor`` of any shape.
        bucket_min (float|None): represent bucket min value,
            if None value, ``data.min()`` will be used
        bucket_max (float|None): represent bucket max value,
            if None value, ``data.max()`` will be used
        bucket_count (int):  positive ``int``. The output will have this many buckets.
        edge_inclusive (bool): whether to include a data point that lies exactly
            on the leftmost/rightmost edge.
        step (None|Tensor): step value for this summary. this defaults to
            ``alf.summary.get_global_counter()``
    """
    data = data.to(torch.float64)
    if bucket_min is None:
        bucket_min = data.min()
    else:
        bucket_min = torch.as_tensor(bucket_min)
    if bucket_max is None:
        bucket_max = data.max()
    else:
        bucket_max = torch.as_tensor(bucket_max)
    bins = (
        bucket_min +
        (torch.arange(bucket_count + 1, dtype=torch.float64) / bucket_count) *
        (bucket_max - bucket_min))
    if edge_inclusive:
        bins[0] -= 1e-6
        bins[-1] += 1e-6
    data = data.clamp(bucket_min, bucket_max)
    alf.summary.histogram(name, data, step=step, bins=bins.cpu())


@_summary_wrapper
@alf.configurable
def summarize_variables(name_and_params, with_histogram=True):
    """Add summaries for variables.

    Args:
        name_and_params (list[(str, Parameter)]): A list of ``(name, Parameter)``
            tuples.
        with_histogram (bool): If True, generate histogram.
    """
    for var_name, var in name_and_params:
        var_values = var
        if with_histogram and torch.all(torch.isfinite(var_values)):
            # Need to make sure all values are finite to avoid the histogram range
            # error
            alf.summary.histogram(
                name='summarize_vars/' + var_name + '_value', data=var_values)
        alf.summary.scalar(
            name='summarize_vars/' + var_name + '_value_norm',
            data=var_values.norm())


@_summary_wrapper
@alf.configurable
def summarize_gradients(name_and_params, with_histogram=True):
    """Add summaries for gradients.

    Args:
        name_and_params (list[(str, Parameter)]): A list of ``(name, Parameter)``
            tuples.
        with_histogram (bool): If True, generate histogram.
    """
    for var_name, var in name_and_params:
        if var.grad is None:
            continue
        grad_values = var.grad
        if with_histogram:
            if torch.all(grad_values.isfinite()):
                alf.summary.histogram(
                    name='summarize_grads/' + var_name + '_gradient',
                    data=grad_values)
        alf.summary.scalar(
            name='summarize_grads/' + var_name + '_gradient_norm',
            data=grad_values.norm())


alf.summary.histogram = _summary_wrapper(alf.summary.histogram)


@_summary_wrapper
def add_nested_summaries(prefix, data):
    """Add summary of a nest of data.

    Args:
        prefix (str): the prefix of the names of the summaries
        data (dict or namedtuple): data to be summarized
    """

    def _summarize(path, x):
        if isinstance(x, torch.Tensor):
            alf.summary.scalar(prefix + '/' + path, x)

    py_map_structure_with_path(_summarize, data)


@_summary_wrapper
@alf.configurable
def summarize_per_category_loss(loss_info: LossInfo,
                                summarize_count: bool = False,
                                label_names: Optional[List[str]] = None):
    """Add summary about each category of the unaggregated ``loss_info.loss``
    of the shape (T, B), or (B, ) by partitioning it according to
    ``loss_info.batch_label``, which has the same shape as ``loss_info.loss``.
    It also creates summarization of the number of samples encountered
    for each category.

    Args:
        loss_info: do per-category summarization if
        ``loss_info.batch_label`` is present, and skip otherwise
        summarize_count: whether to summarize the number of samples
            for each category as well
        label_names: the names of each category to be used
            in tensorboard summary. The category number will be used if
            ``label_names`` is None.
    """

    if loss_info.batch_label != ():
        assert loss_info.batch_label.shape == loss_info.loss.shape, (
            "shape mismatch between batch_label shape {} and loss "
            "shape {}".format(loss_info.batch_label.shape,
                              loss_info.loss.shape))

        # (T, B) -> (T * B, )
        loss = loss_info.loss.reshape(-1)
        batch_label = loss_info.batch_label.int().reshape(-1)
        labels = torch.unique(batch_label)
        labels = labels.tolist()

        for label in labels:
            subset_indices = (batch_label == label)
            subset_loss = loss[subset_indices]
            if label_names is None:
                label_str = label
            else:
                label_str = label_names[label]

            alf.summary.scalar(
                'loss/loss_for_category_{}'.format(label_str),
                data=subset_loss.mean())
            if summarize_count:
                alf.summary.scalar(
                    'loss/sample_count_for_category_{}'.format(label_str),
                    data=subset_indices.sum())
    else:
        return


@_summary_wrapper
def summarize_loss(loss_info: LossInfo):
    """Add summary about ``loss_info``

    Args:
        loss_info (LossInfo): ``loss_info.extra`` must be a namedtuple
    """
    if not isinstance(loss_info.loss, tuple):
        alf.summary.scalar('loss', data=loss_info.loss)
    if loss_info.gns != ():
        alf.summary.scalar('gradient_noise_scale', data=loss_info.gns)
    if not loss_info.extra:
        return
    # Support extra as namedtuple or dict (more flexible)
    if is_namedtuple(loss_info.extra) or isinstance(loss_info.extra, dict):
        add_nested_summaries('loss', loss_info.extra)


@_summary_wrapper
def summarize_nest(prefix, nest):
    def _summarize(path, tensor):
        add_mean_hist_summary(prefix + "/" + path, tensor)

    alf.nest.py_map_structure_with_path(_summarize, nest)


@_summary_wrapper
def summarize_action(actions, action_specs, name="action"):
    """Generate histogram summaries for actions.

    Actions whose rank is more than 1 will be skipped.

    Args:
        actions (nested Tensor): actions to be summarized
        action_specs (nested TensorSpec): spec for the actions
        name (str): name of the summary
    """

    def _summarize_one(path, action, action_spec):
        if len(action_spec.shape) > 1:
            return

        if action_spec.is_discrete:
            histogram_discrete(
                name="%s/%s" % (name, path),
                data=action,
                bucket_min=int(action_spec.minimum),
                bucket_max=int(action_spec.maximum))
        else:
            if len(action_spec.shape) == 0:
                action_dim = 1
            else:
                action_dim = action_spec.shape[-1]
            action = torch.reshape(action, (-1, action_dim))

            def _get_val(a, i):
                return a if len(a.shape) == 0 else a[i]

            for a in range(action_dim):
                histogram_continuous(
                    name="%s/%s/%s/value" % (name, path, a),
                    data=action[:, a],
                    bucket_min=_get_val(action_spec.minimum, a),
                    bucket_max=_get_val(action_spec.maximum, a))
                alf.summary.scalar("%s/%s/%s/mean" % (name, path, a),
                                   action[:, a].mean())

    py_map_structure_with_path(_summarize_one, actions, action_specs)


@_summary_wrapper
def summarize_distribution(name, distributions):
    """Generate summary for distributions.

    Currently the following types of distributions are supported:

    * Normal, StableCauchy, Beta: mean and std of each dimension will be summarized
    * Above distribution wrapped by Independent and TransformedDistribution:
      the base distribution is summarized
    * MixtureSameFamily: the mixture weights and the most likely component distribution
        are summarized
    * Tensor: each dimension dist[..., a] will be summarized

    Note that unsupported distributions will be ignored (no error reported).

    Args:
        name (str): name of the summary
        distributions (nested td.distribuation.Distribution): distributions to
            be summarized.
    """

    def _summarize_one(path, dist):
        if isinstance(dist, torch.Tensor):
            # dist might be a Tensor
            action_dim = dist.shape[-1]
            for a in range(action_dim):
                add_mean_hist_summary("%s_loc/%s/%s" % (name, path, a),
                                      dist[..., a])
        else:
            ind = None
            if isinstance(dist, td.MixtureSameFamily):
                probs = dist.mixture_distribution.probs
                n = probs.shape[-1]
                if n <= 10:  # 10 is arbitrarily chosen to avoid too many summaries
                    for i in range(n):
                        add_mean_hist_summary(
                            "%s_probs/%s/%s" % (name, path, i), probs[..., i])
                else:
                    entropy = -torch.xlogy(probs, probs).sum(-1)
                    add_mean_hist_summary("%s_cond_entropy/%s" % (name, path),
                                          entropy)
                    probs = probs.reshape(-1, probs.shape[-1]).mean(0)
                    entropy = -torch.xlogy(probs, probs).sum()
                    alf.summary.scalar("%s_entropy/%s" % (name, path), entropy)

                ind = dist_utils.get_mode(dist.mixture_distribution)
                dist = dist.component_distribution
            dist = dist_utils.get_base_dist(dist)
            if isinstance(dist, (td.Normal, dist_utils.StableCauchy,
                                 dist_utils.TruncatedDistribution)):
                loc = dist.loc
                log_scale = dist.scale.log()
            elif isinstance(dist, td.Beta):
                loc = dist.mean
                log_scale = 0.5 * dist.variance.log()
            else:
                return

            if ind is not None:
                # This is for handling mixture distribution. The component with
                # the highest probability is selected for summary.
                if len(ind.shape) == 1:
                    i0 = torch.arange(ind.shape[0])
                    loc = loc[i0, ind]
                    log_scale = log_scale[i0, ind]
                elif len(ind.shape) == 2:
                    i0 = torch.arange(ind.shape[0]).unsqueeze(-1)
                    i1 = torch.arange(ind.shape[1])
                    loc = loc[i0, i1, ind]
                    log_scale = log_scale[i0, i1, ind]
                else:
                    return

            action_dim = loc.shape[-1]
            for a in range(action_dim):
                add_mean_hist_summary("%s_log_scale/%s/%s" % (name, path, a),
                                      log_scale[..., a])
                add_mean_hist_summary("%s_loc/%s/%s" % (name, path, a),
                                      loc[..., a])

    py_map_structure_with_path(_summarize_one, distributions)


def add_mean_hist_summary(name, value):
    """Generate mean and histogram summary of ``value``.

    Args:
        name (str): name of the summary
        value (Tensor): tensor to be summarized
    """
    alf.summary.histogram(name + "/value", value)
    add_mean_summary(name + "/mean", value)


def safe_mean_hist_summary(name, value, mask=None):
    """Generate mean and histogram summary of ``value``.

    It skips the summary if ``value`` is empty.

    Args:
        name (str): name of the summary
        value (Tensor): tensor to be summarized
        mask (bool Tensor): optional mask to indicate which element of value
            to use. Its shape needs to be same as that of ``value``
    """
    if mask is not None:
        value = value[mask]
    if np.prod(value.shape) > 0:
        add_mean_hist_summary(name, value)


def add_mean_summary(name, value):
    """Generate mean summary of ``value``.

    Args:
        name (str): name of the summary
        value (Tensor): tensor to be summarized
    """
    if not value.dtype.is_floating_point:
        value = value.to(torch.float32)
    alf.summary.scalar(name, value.mean())


def safe_mean_summary(name, value, mask=None):
    """Generate mean summary of ``value``.

    It skips the summary if ``value`` is empty.

    Args:
        name (str): name of the summary
        value (Tensor): tensor to be summarized
        mask (bool Tensor): optional mask to indicate which element of value
            to use. Its shape needs to be same as that of ``value``
    """
    if mask is not None:
        value = value[mask]
    if np.prod(value.shape) > 0:
        add_mean_summary(name, value)


_contexts = {}


class record_time(object):
    """A context manager for record the time.

    It records the average time spent under the context between
    two summaries.

    Example:

    .. code-block:: python

        with record_time("time/calc"):
            long_function()
    """

    def __init__(self, tag):
        """Create a context object for recording time.

        By default, record_time will do cuda.synchronize() before entering and
        after leaving the context to measure the time accurately. This behavior
        can be disabled by setting environment variable ALF_RECORD_TIME_SYNC to 0
        if you suspect synchronization slow down your code. See
        https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution.

        Args:
            tag (str): the summary tag for the the time.
        """
        sync = os.environ.get("ALF_RECORD_TIME_SYNC", "1") != "0"
        self._tag = tag
        self._sync = sync
        caller = logging.get_absl_logger().findCaller()
        # token is a string of filename:lineno:tag
        token = caller[0] + ':' + str(caller[1]) + ':' + tag
        if token not in _contexts:
            _contexts[token] = {'time': 0., 'c0': int(get_global_counter())}
        self._counter = _contexts[token]

    def __enter__(self):
        if self._sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.time()

    def __exit__(self, type, value, traceback):
        if self._sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._counter['time'] += time.time() - self._t0
        if should_record_summaries():
            c0 = self._counter['c0']
            c1 = int(get_global_counter())
            if c1 > c0:
                alf.summary.scalar(self._tag,
                                   self._counter['time'] / (c1 - c0))
                self._counter['time'] = .0
                self._counter['c0'] = c1


def summarize_tensor_gradients(name, tensor, batch_dims=1, clone=False):
    """Summarize the gradient of ``tensor`` during backward.

    Args:
        name (str): name of the summary
        tensor (nested Tensor): tensor of which the gradient is to be summarized.
        batch_dims (int): first so many dimensions are treated as batch dimensions
        clone (bool): If True, ``tensor`` will first be cloned. This is useful
            if ``tensor`` is used in multiple places and you only want to summarize
            the gradient from one place. If False, the gradient will be the sum
            from all gradients backpropped to ``tensor``.
    Returns:
        ``tensor`` or cloned ``tensor``: the cloned ``tensor`` should be used for
            the downstream calculations.
    """

    def _hook(grad, name):
        if grad is not None:
            norm = grad.reshape(*grad.shape[0:batch_dims], -1).norm(dim=-1)
            alf.summary.scalar(name + '/max_norm', norm.max())
            alf.summary.scalar(name + '/avg_norm', norm.mean())

    def _register_hook1(tensor, name):
        if tensor.requires_grad:
            if clone:
                tensor = tensor.clone()
            tensor.register_hook(functools.partial(_hook, name=name))
        return tensor

    if not torch.is_grad_enabled():
        return tensor
    name = '/' + alf.summary.scope_name() + name
    if not is_nested(tensor):
        return _register_hook1(tensor, name)
    else:

        def _register_hook(path, x):
            return _register_hook1(x, name + '/' + path)

        tensor = py_map_structure_with_path(_register_hook, tensor)
        return tensor


def summarize_distribution_gradient(name,
                                    distribution,
                                    batch_dims=1,
                                    clone=False):
    """Summarize the gradient of the parameters of ``distribution`` during backward.

    Args:
        name (str): name of the summary
        distribution (nested Distribution): distribution of which the gradient is to be summarized.
        batch_dims (int): first so many dimensions are treated as batch dimensions
        clone (bool): If True, ``distribution`` will first be cloned. This is useful
            if ``distribution`` is used in multiple places and you only want to summarize
            the gradient from one place. If False, the gradient will be the sum
            from all gradients backpropped to ``distribution``.
    Returns:
        ``distribution`` or cloned ``distribution``: the cloned ``distribution``
            should be used for the downstream calculations.
    """
    dist_params = dist_utils.distributions_to_params(distribution)
    if clone:
        spec = dist_utils.extract_spec(distribution)
        dist_params = map_structure(torch.clone, dist_params)
        distribution = dist_utils.params_to_distributions(dist_params, spec)
    summarize_tensor_gradients(
        name, dist_params, batch_dims=batch_dims, clone=False)
    return distribution
