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

import torch

import alf
import alf.utils.common as common


def _gather_nest(nest, indices):
    return alf.nest.map_structure(lambda t: t[indices], nest)


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
    gather_indices = torch.where(mask)[0]
    return _gather_nest(data, gather_indices)


def conditional_update(target, cond, func, *args, **kwargs):
    """Update target according to cond mask

    Compute result as an update of ``target`` based on ``cond``. To be specific,
    result[row] is ``func(*args[row], **kwargs[row])`` if cond[row] is True,
    otherwise result[row] will be target[row]. Note that ``target`` will not be
    changed.

    If you simply want to do some conditional computation without actually
    returning any results. You can use conditional_update in the following way:

    .. code-block:: python

        # func needs to return an empty tuple ()
        conditional_update((), cond, func, *args, **kwargs)


    Args:
        target (nested Tensor): target to be updated
        func (Callable): a function with arguments (*args, **kwargs) and returning
            a nest with same structure as target
        cond (Tensor): 1d bool Tensor with shape[0] == target.shape[0]
    Returns:
        nest with the same structure and shape as target.
    """
    # the return of torch.where() is a tuple (indices, )
    gather_indices = torch.where(cond)[0]

    def _update_subset():
        selected_args = _gather_nest(args, gather_indices)
        selected_kwargs = _gather_nest(kwargs, gather_indices)
        updates = func(*selected_args, **selected_kwargs)

        def _update(tgt, updt):
            scatter_indices = common.expand_dims_as(gather_indices, updt)
            scatter_indices = scatter_indices.expand_as(updt)
            return tgt.scatter(0, scatter_indices, updt)

        return alf.nest.map_structure(_update, target, updates)

    total = cond.shape[0]
    n = gather_indices.shape[0]
    if n == 0:
        return target
    elif n == total:
        return func(*args, **kwargs)
    else:
        return _update_subset()
