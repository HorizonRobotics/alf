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
import time

import tensorflow as tf
from tensorboard.plugins.histogram import metadata
from tensorflow.python.ops import summary_ops_v2

from alf.utils.conditional_ops import run_if

DEFAULT_BUCKET_COUNT = 30


def _summary_wrapper(summary_func):
    """Summary wrapper

    Wrapper summary function to reduce cost for data computation
     """

    def wrapper(*args, **kwargs):
        from alf.utils.common import run_if
        return run_if(summary_ops_v2._should_record_summaries_v2(), lambda:
                      summary_func(*args, **kwargs))

    return wrapper


@_summary_wrapper
def histogram_discrete(name,
                       data,
                       bucket_min,
                       bucket_max,
                       step=None,
                       description=None):
    """histogram for discrete data.

    Args:
        name (str): name for this summary
        data (Tensor): A `Tensor` integers of any shape.
        bucket_min (int): represent bucket min value
        bucket_max (int): represent bucket max value
            bucket count is calculate as `bucket_max - bucket_min + 1`
            and output will have this many buckets.
        step (None|tf.Variable):  step value for this summary. this defaults to
            `tf.summary.experimental.get_step()`
        description (str): Optional long-form description for this summary
    """
    summary_metadata = metadata.create_summary_metadata(
        display_name=None, description=description)
    summary_scope = (getattr(tf.summary.experimental, 'summary_scope', None)
                     or tf.summary.summary_scope)
    with summary_scope(
            name, 'histogram_summary',
            values=[data, bucket_min, bucket_max, step]) as (tag, _):
        with tf.name_scope('buckets'):
            bucket_count = bucket_max - bucket_min + 1
            data = data - bucket_min
            one_hots = tf.one_hot(
                tf.reshape(data, shape=[-1]), depth=bucket_count)
            bucket_counts = tf.cast(
                tf.reduce_sum(input_tensor=one_hots, axis=0), tf.float64)
            edge = tf.cast(tf.range(bucket_count), tf.float64)
            # histogram can not draw when left_edge == right_edge
            left_edge = edge - 1e-12
            right_edge = edge + 1e-12
            tensor = tf.transpose(
                a=tf.stack([left_edge, right_edge, bucket_counts]))

        return tf.summary.write(
            tag=tag, tensor=tensor, step=step, metadata=summary_metadata)


@_summary_wrapper
def histogram_continuous(name,
                         data,
                         bucket_min=None,
                         bucket_max=None,
                         bucket_count=DEFAULT_BUCKET_COUNT,
                         step=None,
                         description=None):
    """histogram for continuous data .

    Args:
        name (str): name for this summary
        data (Tensor): A `Tensor` of any shape.
        bucket_min (float|None): represent bucket min value,
            if None value of tf.reduce_min(data) will be used
        bucket_max (float|None): represent bucket max value,
            if None value tf.reduce_max(data) will be used
        bucket_count (int):  positive `int`. The output will have this many buckets.
        step (None|tf.Variable):  step value for this summary. this defaults to
            `tf.summary.experimental.get_step()`
        description (str): Optional long-form description for this summary
    """
    summary_metadata = metadata.create_summary_metadata(
        display_name=None, description=description)
    summary_scope = (getattr(tf.summary.experimental, 'summary_scope', None)
                     or tf.summary.summary_scope)
    with summary_scope(
            name,
            'histogram_summary',
            values=[data, bucket_min, bucket_max, bucket_count, step]) as (tag,
                                                                           _):
        with tf.name_scope('buckets'):
            data = tf.cast(tf.reshape(data, shape=[-1]), tf.float64)
            if bucket_min is None:
                bucket_min = tf.reduce_min(data)
            if bucket_max is None:
                bucket_max = tf.reduce_min(data)
            range_ = bucket_max - bucket_min
            bucket_width = range_ / tf.cast(bucket_count, tf.float64)
            offsets = data - bucket_min
            bucket_indices = tf.cast(
                tf.floor(offsets / bucket_width), dtype=tf.int32)
            clamped_indices = tf.clip_by_value(bucket_indices, 0,
                                               bucket_count - 1)
            one_hots = tf.one_hot(clamped_indices, depth=bucket_count)
            bucket_counts = tf.cast(
                tf.reduce_sum(input_tensor=one_hots, axis=0), dtype=tf.float64)
            edges = tf.linspace(bucket_min, bucket_max, bucket_count + 1)
            edges = tf.concat([edges[:-1], [float(bucket_max)]], 0)
            edges = tf.cast(edges, tf.float64)
            left_edges = edges[:-1]
            right_edges = edges[1:]
            tensor = tf.transpose(
                a=tf.stack([left_edges, right_edges, bucket_counts]))
        return tf.summary.write(
            tag=tag, tensor=tensor, step=step, metadata=summary_metadata)


def unique_var_names(vars):
    """Generate unique names for `vars`

    Variable names may not be not unique, which can create problems for summary.
    This function add a suffix when the names duplicate.

    Args:
        vars (iterable of Varaible): the list of Variables
    Returns:
        iterator of the unique variable names in the same order as vars.
    """
    count = {}
    for var in vars:
        var_name = var.name.replace(':', '_')
        if var_name in count:
            count[var_name] += 1
            var_name += "_" + str(count[var_name])
        else:
            count[var_name] = 0
        yield var_name


@_summary_wrapper
def add_variables_summaries(grads_and_vars, step=None):
    """Add summaries for variables.

    Args:
      grads_and_vars (list): A list of (gradient, variable) pairs.
      step (tf.Variable): Variable to use for summaries.
    """
    if not grads_and_vars:
        return
    vars = [v for g, v in grads_and_vars]
    for var, var_name in zip(vars, unique_var_names(vars)):
        if isinstance(var, tf.IndexedSlices):
            var_values = var.values
        else:
            var_values = var
        tf.summary.histogram(
            name='summarize_vars/' + var_name + '_value',
            data=var_values,
            step=step)
        tf.summary.scalar(
            name='summarize_vars/' + var_name + '_value_norm',
            data=tf.linalg.global_norm([var_values]),
            step=step)


@_summary_wrapper
def add_gradients_summaries(grads_and_vars, step=None):
    """Add summaries to gradients.

    Args:
      grads_and_vars (list): A list of gradient to variable pairs (tuples).
      step (tf.Variable): Variable to use for summaries.
    """
    if not grads_and_vars:
        return
    grads, vars = zip(*grads_and_vars)
    for grad, var_name in zip(grads, unique_var_names(vars)):
        if grad is not None:
            if isinstance(grad, tf.IndexedSlices):
                grad_values = grad.values
            else:
                grad_values = grad
            tf.summary.histogram(
                name='summarize_grads/' + var_name + '_gradient',
                data=grad_values,
                step=step)
            tf.summary.scalar(
                name='summarize_grads/' + var_name + '_gradient_norm',
                data=tf.linalg.global_norm([grad_values]),
                step=step)


tf.summary.histogram = _summary_wrapper(tf.summary.histogram)


def summarize_action_dist(action_distributions,
                          action_specs,
                          name="action_dist"):
    """Generate summary for action distributions.

    Args:
        action_distributions (nested tfp.distribuations.Distribution):
            distributions to be summarized
        action_specs (nested BoundedTensorSpec): specs for the actions
        name (str): name of the summary
    """
    import tensorflow_probability as tfp
    from tf_agents.distributions.utils import SquashToSpecNormal
    action_specs = tf.nest.flatten(action_specs)
    actions = tf.nest.flatten(action_distributions)

    for i, (dist, action_spec) in enumerate(zip(actions, action_specs)):
        if isinstance(dist, SquashToSpecNormal):
            dist = dist.input_distribution
        if not isinstance(dist, tfp.distributions.Normal):
            # Only support Normal currently
            continue
        action_dim = action_spec.shape[-1]
        log_scale = tf.math.log(dist.scale)
        for a in range(action_dim):
            tf.summary.histogram(
                name="%s_log_scale/%s/%s" % (name, i, a),
                data=log_scale[..., a])
            tf.summary.histogram(
                name="%s_loc/%s/%s" % (name, i, a), data=dist.loc[..., a])


def get_current_scope():
    """Returns the current name scope in the default_graph.

    For example:
    ```python
    with tf.name_scope('scope1'):
        with tf.name_scope('scope2'):
            print(get_current_scope())
    ```
    would print the string `scope1/scope2/`.

    Returns:
        A string representing the current name scope.
    """
    with tf.name_scope("foo") as scope:
        # With the above example, scope is "scope1/scope2/foo_1/".
        # We want to return "scope1/scope2/".
        return scope[:scope.rfind('/', 0, -1) + 1]


def add_mean_hist_summary(name, value):
    """Generate mean and histogram summary of `value`.

    Args:
        name (str): name of the summary
        value (Tensor): tensor to be summarized
    Returns:
        None
    """
    tf.summary.histogram(name + "/value", value)
    add_mean_summary(name + "/mean", value)


def safe_mean_hist_summary(name, value):
    """Generate mean and histogram summary of `value`.

    It skips the summary if `value` is empty.

    Args:
        name (str): name of the summary
        value (Tensor): tensor to be summarized
    Returns:
        None
    """
    current_scope = get_current_scope()
    # The reason of prefixing name with current_scope is that inside run_if,
    # somehow get_current_scope() is '', which makes summary tag unscoped.
    run_if(
        tf.reduce_prod(tf.shape(value)) >
        0, lambda: add_mean_hist_summary(current_scope + name, value))


def add_mean_summary(name, value):
    """Generate mean summary of `value`.

    Args:
        name (str): name of the summary
        value (Tensor): tensor to be summarized
    Returns:
        None
    """
    if not value.dtype.is_floating:
        value = tf.cast(value, tf.float32)
    tf.summary.scalar(name, tf.reduce_mean(value))


def safe_mean_summary(name, value):
    """Generate mean summary of `value`.

    It skips the summary if `value` is empty.

    Args:
        name (str): name of the summary
        value (Tensor): tensor to be summarized
    Returns:
        None
    """
    current_scope = get_current_scope()
    # The reason of prefixing name with current_scope is that inside run_if,
    # somehow get_current_scope() is '', which makes summary tag unscoped.
    run_if(
        tf.reduce_prod(tf.shape(value)) >
        0, lambda: add_mean_summary(current_scope + name, value))


class record_time(object):
    """A context manager for record the time.

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

    def __enter__(self):
        self._t0 = time.time()

    def __exit__(self, type, value, traceback):
        t = time.time() - self._t0
        tf.summary.scalar(self._tag, t)
