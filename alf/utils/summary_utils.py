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
import gin
import numpy as np
from tensorboard.plugins.histogram import metadata
import time
import torch
import torch.distributions as td

import alf
from alf.data_structures import LossInfo
from alf.nest import is_namedtuple
from alf.utils import dist_utils
from alf.summary import should_record_summaries

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
        data (Tensor): A `Tensor` integers of any shape.
        bucket_min (int): represent bucket min value
        bucket_max (int): represent bucket max value
            bucket count is calculate as `bucket_max - bucket_min + 1`
            and output will have this many buckets.
        step (None|Tensor): step value for this summary. this defaults to
            `alf.summary.get_global_counter()`
    """
    alf.summary.histogram(
        name,
        data,
        step=step,
        bins=torch.arange(bucket_min, bucket_max + 1).cpu())


@_summary_wrapper
def histogram_continuous(name,
                         data,
                         bucket_min=None,
                         bucket_max=None,
                         bucket_count=DEFAULT_BUCKET_COUNT,
                         step=None):
    """histogram for continuous data .

    Args:
        name (str): name for this summary
        data (Tensor): A `Tensor` of any shape.
        bucket_min (float|None): represent bucket min value,
            if None value of tf.reduce_min(data) will be used
        bucket_max (float|None): represent bucket max value,
            if None value tf.reduce_max(data) will be used
        bucket_count (int):  positive `int`. The output will have this many buckets.
        step (None|Tensor): step value for this summary. this defaults to
            `alf.summary.get_global_counter()`
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
    data = data.clamp(bucket_min, bucket_max)
    alf.summary.histogram(name, data, step=step, bins=bins.cpu())


@_summary_wrapper
@gin.configurable
def summarize_variables(name_and_params, with_histogram=True):
    """Add summaries for variables.

    Args:
        name_and_params (list[(str, Parameter)]): A list of (name, Parameter)
            tuples.
        with_histogram (bool): If True, generate histogram.
    """
    for var_name, var in name_and_params:
        var_values = var
        if with_histogram:
            alf.summary.histogram(
                name='summarize_vars/' + var_name + '_value', data=var_values)
        alf.summary.scalar(
            name='summarize_vars/' + var_name + '_value_norm',
            data=var_values.norm())


@_summary_wrapper
@gin.configurable
def summarize_gradients(name_and_params, with_histogram=True):
    """Add summaries for gradients.

    Args:
        name_and_params (list[(str, Parameter)]): A list of (name, Parameter)
            tuples.
        with_histogram (bool): If True, generate histogram.
    """
    for var_name, var in name_and_params:
        if var.grad is None:
            continue
        grad_values = var.grad
        if with_histogram:
            alf.summary.histogram(
                name='summarize_grads/' + var_name + '_gradient',
                data=grad_values)
        alf.summary.scalar(
            name='summarize_grads/' + var_name + '_gradient_norm',
            data=grad_values.norm())


alf.summary.histogram = _summary_wrapper(alf.summary.histogram)


def add_nested_summaries(prefix, data):
    """Add summary about loss_info

    Args:
        prefix (str): the prefix of the names of the summaries
        data (dict or namedtuple): data to be summarized
    """
    fields = data.keys() if isinstance(data, dict) else data._fields
    for field in fields:
        elem = data[field] if isinstance(data, dict) else getattr(data, field)
        name = prefix + '/' + field
        if isinstance(elem, dict) or is_namedtuple(elem):
            add_nested_summaries(name, elem)
        elif isinstance(elem, torch.Tensor):
            alf.summary.scalar(name, elem)


def summarize_loss(loss_info: LossInfo):
    """Add summary about loss_info

    Args:
        loss_info (LossInfo): loss_info.extra must be a namedtuple
    """
    alf.summary.scalar('loss', data=loss_info.loss)
    if not loss_info.extra:
        return
    # Support extra as namedtuple or dict (more flexible)
    if is_namedtuple(loss_info.extra) or isinstance(loss_info.extra, dict):
        add_nested_summaries('loss', loss_info.extra)


def summarize_action(actions, action_specs, name="action"):
    """Generate histogram summaries for actions.

    Actions whose rank is more than 1 will be skipped.

    Args:
        actions (nested Tensor): actions to be summarized
        action_specs (nested TensorSpec): spec for the actions
    """
    action_specs = alf.nest.flatten(action_specs)
    actions = alf.nest.flatten(actions)

    for i, (action, action_spec) in enumerate(zip(actions, action_specs)):
        if len(action_spec.shape) > 1:
            continue

        if action_spec.is_discrete:
            histogram_discrete(
                name="%s/%s" % (name, i),
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
                    name="%s/%s/%s" % (name, i, a),
                    data=action[:, a],
                    bucket_min=_get_val(action_spec.minimum, a),
                    bucket_max=_get_val(action_spec.maximum, a))


def summarize_action_dist(action_distributions,
                          action_specs,
                          name="action_dist"):
    """Generate summary for action distributions.

    Args:
        action_distributions (nested td.distribuation.Distribution):
            distributions to be summarized
        action_specs (nested BoundedTensorSpec): specs for the actions
        name (str): name of the summary
    """
    action_specs = alf.nest.flatten(action_specs)
    actions = alf.nest.flatten(action_distributions)

    for i, (dist, action_spec) in enumerate(zip(actions, action_specs)):
        if isinstance(dist, torch.Tensor):
            # dist might be a Tensor
            action_dim = action_spec.shape[-1]
            for a in range(action_dim):
                alf.summary.histogram(
                    name="%s_loc/%s/%s" % (name, i, a), data=dist[..., a])
        else:
            dist = dist_utils.get_base_dist(dist)
            action_dim = action_spec.shape[-1]
            log_scale = dist.scale.log()
            for a in range(action_dim):
                alf.summary.histogram(
                    name="%s_log_scale/%s/%s" % (name, i, a),
                    data=log_scale[..., a])
                alf.summary.histogram(
                    name="%s_loc/%s/%s" % (name, i, a), data=dist.loc[..., a])


def add_mean_hist_summary(name, value):
    """Generate mean and histogram summary of `value`.

    Args:
        name (str): name of the summary
        value (Tensor): tensor to be summarized
    Returns:
        None
    """
    alf.summary.histogram(name + "/value", value)
    add_mean_summary(name + "/mean", value)


def safe_mean_hist_summary(name, value, mask):
    """Generate mean and histogram summary of `value`.

    It skips the summary if `value` is empty.

    Args:
        name (str): name of the summary
        value (Tensor): tensor to be summarized
        mask (bool Tensor): optional mask to indicate which element of value
            to use. Its shape needs to be same as that of `value`
    Returns:
        None
    """
    if mask is not None:
        value = value[mask]
    if np.prod(value.shape) > 0:
        add_mean_hist_summary(name, value)


def add_mean_summary(name, value):
    """Generate mean summary of `value`.

    Args:
        name (str): name of the summary
        value (Tensor): tensor to be summarized
    Returns:
        None
    """
    if not value.dtype.is_floating_point:
        value = value.to(torch.float32)
    alf.summary.scalar(name, value.mean())


def safe_mean_summary(name, value):
    """Generate mean summary of `value`.

    It skips the summary if `value` is empty.

    Args:
        name (str): name of the summary
        value (Tensor): tensor to be summarized
    Returns:
        None
    """
    if np.prod(value.shape) > 0:
        add_mean_summary(name, value)


_contexts = {}


class record_time(object):
    """A context manager for record the time.

    It records the average time spent under the context between
    two summaries.

    Example:
    ```python
    with record_time("time/calc"):
        long_function()
    ```
    """

    def __init__(self, tag):
        """Create a context object for recording time.

        Args:
            tag (str): the summary tag for the the time.
        """
        self._tag = tag
        caller = logging.get_absl_logger().findCaller()
        # token is a string of filename:lineno:tag
        token = caller[0] + ':' + str(caller[1]) + ':' + tag
        if token not in _contexts:
            _contexts[token] = {'time': 0., 'n': 0}
        self._counter = _contexts[token]

    def __enter__(self):
        self._t0 = time.time()

    def __exit__(self, type, value, traceback):
        self._counter['time'] += time.time() - self._t0
        self._counter['n'] += 1
        if should_record_summaries():
            alf.summary.scalar(self._tag,
                               self._counter['time'] / self._counter['n'])
            self._counter['time'] = .0
            self._counter['n'] = 0
