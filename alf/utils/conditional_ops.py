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
"""Conditional operations."""

import tensorflow as tf

from alf.utils.scope_utils import get_current_scope


def _gather_nest(nest, indices):
    return tf.nest.map_structure(lambda t: tf.gather(t, indices, axis=0), nest)


def select_from_mask(data, mask):
    """Select the items from data based on mask.

    data[i,...] will be selected to form a new tensor if mask[i] is True or
    non-zero

    Args:
        data (nested Tensor): source tensor
        mask (Tensor): 1D Tensor mask.shape[0] should be same as data.shape[0]
    Returns:
        nested Tensor with the same structure as data
    """
    scatter_indices = tf.where(mask)
    gather_indices = tf.squeeze(scatter_indices, 1)
    return _gather_nest(data, gather_indices)


def conditional_update(target, cond, func, *args, **kwargs):
    """Update target according to cond mask

    Compute result as an update of `target` based on `cond`. To be specific,
    result[row] is func(*args[row], **kwargs[row]) if cond[row] is True,
    otherwise result[row] will be target[row]. Note that target will not be
    changed.

    If you simply want to do some conditional computation without actually
    returning any results. You can use conditional_update in the following way:
    ```
    # func needs to return an empty tuple ()
    conditional_update((), cond, func, *args, **kwargs)
    ```

    Args:
        target (nested Tensor): target to be updated
        func (Callable): a function with arguments (*args, **kwargs) and returning
            a nest with same structure as target
        cond (Tensor): 1d bool Tensor with shape[0] == target.shape[0]
    Returns:
        nest with the same structure and shape as target.
    """
    # shape of indices from where() is [batch_size,1], which is what scatter_nd
    # needs
    scatter_indices = tf.where(cond)
    scope = get_current_scope()

    def _update_subset():
        gather_indices = tf.squeeze(scatter_indices, 1)
        selected_args = _gather_nest(args, gather_indices)
        selected_kwargs = _gather_nest(kwargs, gather_indices)
        with tf.name_scope(scope):
            # tf.case loses the original name scope. Need to restore it.
            updates = func(*selected_args, **selected_kwargs)
        return tf.nest.map_structure(
            lambda tgt, updt: tf.tensor_scatter_nd_update(
                tgt, scatter_indices, updt), target, updates)

    total = tf.shape(cond)[0]
    n = tf.shape(scatter_indices)[0]
    return tf.case(((n == 0, lambda: target),
                    (n == total, lambda: func(*args, **kwargs))),
                   default=_update_subset)


def run_if(cond, func):
    """Run a function if `cond` Tensor is True.

    This function is useful for conditionally executing a function only when
    a condition given by a tf Tensor is True. It is equivalent to the following
    code if `cond` is a python bool value:
    ```python
    if cond:
        func()
    ```
    However, when `cond` is tf bool scalar tensor, the above code does not
    always do what we want because tensorflow does not allow bool scalar tensor
    to be used in the same way as python bool. So we have to use tf.cond to do
    the job.

    Args:
        cond (tf.Tensor): scalar bool Tensor
        func (Callable): function to be run
    Returns:
        None
    """
    scope = get_current_scope()

    def _if_true():
        # The reason of this line is that inside tf.cond, somehow
        # get_current_scope() is '', which makes operations and summaries inside
        # func unscoped. We need this line to restore the original name scope.
        with tf.name_scope(scope):
            func()
            return tf.constant(True)

    tf.cond(cond, _if_true, lambda: tf.constant(False))
