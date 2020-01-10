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
"""Functions for handling nest."""
import wrapt

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.util.nest import map_structure_up_to
from tf_agents.specs.distribution_spec import DistributionSpec


def is_namedtuple(value):
    """Whether the value is a namedtuple instance

    Args:
         value (Object):
    Returns:
        True if the value is a namedtuple instance
    """

    return isinstance(value, tuple) and hasattr(value, '_fields')


def nest_list_to_tuple(nest):
    """Convert the lists in a nest to tuples.

    Some tf-agents function (e.g. ReplayBuffer) cannot accept nest containing
    list. So we need some utility to convert back and forth.

    Args:
        nest (a nest): a nest structure
    Returns:
        nest with the same content as the input but lists are changed to tuples
    """
    # TF may wrap tuple, which causes construction of tuple to fail.
    # So we use the original object instead.
    if isinstance(nest, wrapt.ObjectProxy):
        nest = nest.__wrapped__
    if isinstance(nest, tuple):
        new_nest = tuple(nest_list_to_tuple(item) for item in nest)
        if is_namedtuple(nest):
            # example is a namedtuple
            new_nest = type(nest)(*new_nest)
        return new_nest
    elif isinstance(nest, list):
        return tuple(nest_list_to_tuple(item) for item in nest)
    elif isinstance(nest, dict):
        new_nest = {}
        for k, v in nest.items():
            new_nest[k] = nest_list_to_tuple(v)
        return new_nest
    else:
        return nest


def nest_contains_list(nest):
    """Whether the nest contains list.

    Args:
        nest (nest): a nest structure
    Returns:
        bool: True if nest contains one or more list
    """
    if isinstance(nest, list):
        return True
    elif isinstance(nest, tuple):
        for item in nest:
            if nest_contains_list(item):
                return True
    elif isinstance(nest, dict):
        for _, item in nest.items():
            if nest_contains_list(item):
                return True
    return False


def nest_tuple_to_list(nest, example):
    """Convert the tuples in a nest to list according to example

    If a tuple whole corresponding structure in the example is a list, it will
    be convert to a list.

    Some tf-agents function (e.g. ReplayBuffer) cannot accept nest containing
    list. So we need some utility to convert back and forth.

    Args:
        nest (a nest): a nest structure without list
        example (a nest): the example structure that nest will be converted to
    Returns:
        nest with the same content as the input but some tuples are changed to
        lists
    """
    # TF may wrap tuple, which causes construction of tuple to fail.
    # So we use the original object instead.
    if isinstance(nest, wrapt.ObjectProxy):
        nest = nest.__wrapped__
    if isinstance(example, wrapt.ObjectProxy):
        example = example.__wrapped__
    if isinstance(nest, tuple):
        new_nest = tuple(
            nest_tuple_to_list(nst, exp) for nst, exp in zip(nest, example))
        if is_namedtuple(example):
            # example is a namedtuple
            new_nest = type(example)(*new_nest)
        elif isinstance(example, list):
            # type(example) might be ListWrapper
            new_nest = type(example)(list(new_nest))
        return new_nest
    elif isinstance(nest, list):
        raise ValueError("list is not expected in nest %s" % nest)
    elif isinstance(nest, dict):
        new_nest = {}
        for k, v in nest.items():
            new_nest[k] = nest_tuple_to_list(v, example[k])
        return new_nest
    else:
        return nest


def get_nest_batch_size(nest, dtype=None):
    """Get the batch_size of a nest."""
    batch_size = tf.shape(tf.nest.flatten(nest)[0])[0]
    if dtype is not None:
        batch_size = tf.cast(batch_size, dtype)
    return batch_size


def to_distribution_param_spec(nest):
    def _to_param_spec(spec):
        if isinstance(spec, DistributionSpec):
            return spec.input_params_spec
        elif isinstance(spec, tf.TensorSpec):
            return spec
        else:
            raise ValueError("Only TensorSpec or DistributionSpec is allowed "
                             "in nest, got %s. nest is %s" % (spec, nest))

    return tf.nest.map_structure(_to_param_spec, nest)


def params_to_distributions(nest, nest_spec):
    """Convert distribution parameters to Distribution, keep Tensors unchanged.

    Args:
        nest (nested tf.Tensor): nested Tensor and dictionary of the Tensor
            parameters of Distribution. Typically, `nest` is obtained using
            `distributions_to_params()`
        nest_spec (nested DistributionSpec and TensorSpec): The distribution
            params will be converted to Distribution according to the
            corresponding DistributionSpec in nest_spec
    Returns:
        nested Distribution/Tensor
    """

    def _to_dist(spec, params):
        if isinstance(spec, DistributionSpec):
            return spec.build_distribution(**params)
        elif isinstance(spec, tf.TensorSpec):
            return params
        else:
            raise ValueError(
                "Only DistributionSpec or TensorSpec is allowed "
                "in nest_spec, got %s. nest_spec is %s" % (spec, nest_spec))

    return map_structure_up_to(nest_spec, _to_dist, nest_spec, nest)


def distributions_to_params(nest):
    """Convert distributions to its parameters, keep Tensors unchanged.

    Only returns parameters that have tf.Tensor values.

    Args:
        nest (nested Distribution and Tensor): Each Distribution will be
            converted to dictionary of its Tensor parameters.
    Returns:
        A nest of Tensor/Distribution parameters. Each leaf is a Tensor or a
        dict corresponding to one distribution, with keys as parameter name and
        values as tensors containing parameter values.
    """

    def _to_params(dist_or_tensor):
        if isinstance(dist_or_tensor, tfp.distributions.Distribution):
            params = dist_or_tensor.parameters
            return {
                k: params[k]
                for k in params if isinstance(params[k], tf.Tensor)
            }
        elif isinstance(dist_or_tensor, tf.Tensor):
            return dist_or_tensor
        else:
            raise ValueError(
                "Only Tensor or Distribution is allowed in nest, ",
                "got %s. nest is %s" % (dist_or_tensor, nest))

    return tf.nest.map_structure(_to_params, nest)
