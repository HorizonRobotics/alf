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

import numpy as np
import gin

import torch


def is_namedtuple(value):
    """Whether the value is a namedtuple instance

    Args:
         value (Object):
    Returns:
        True if the value is a namedtuple instance
    """

    return isinstance(value, tuple) and hasattr(value, '_fields')


def is_unnamedtuple(value):
    """Whether the value is an unnamedtuple instance"""
    return isinstance(value, tuple) and not is_namedtuple(value)


def extract_fields_from_nest(nest):
    """Extract fields and the corresponding values from a nest if it's either
    a `namedtuple` or `dict`.

    Returns:
        An iterator that generates (field, value) pairs.
    """
    assert is_namedtuple(nest) or isinstance(nest, dict), \
        "Nest {} must be a dict or namedtuple!".format(nest)
    fields = nest.keys() if isinstance(nest, dict) else nest._fields
    for field in fields:
        value = nest[field] if isinstance(nest, dict) else getattr(nest, field)
        yield field, value


def is_nested(value):
    """Returns true if the input is one of: `list`, `unnamedtuple`, `dict`, or
    `namedtuple`. Note that this definition is different from tf's is_nested
    where all types that are collections.abc.Sequence are defined to be nested.
    """
    return isinstance(value, (list, tuple, dict))


def flatten(nest):
    """Returns a flat list from a given nested structure."""
    if not is_nested(nest):
        # any other data type will be returned as it is
        return [nest]
    flattened = []
    if isinstance(nest, list) or is_unnamedtuple(nest):
        for value in nest:
            flattened.extend(flatten(value))
    else:
        for _, value in extract_fields_from_nest(nest):
            flattened.extend(flatten(value))
    return flattened


def assert_same_structure(nest1, nest2):
    """Asserts that two structures are nested in the same way."""
    # When neither is nested, the assertion won't fail
    if is_nested(nest1) or is_nested(nest2):
        assert type(nest1) == type(nest2)
        if isinstance(nest1, list) or is_unnamedtuple(nest1):
            for value1, value2 in zip(nest1, nest2):
                assert_same_structure(value1, value2)
        else:
            fields_and_values1 = sorted(
                list(extract_fields_from_nest(nest1)), key=lambda fv: fv[0])
            fields_and_values2 = sorted(
                list(extract_fields_from_nest(nest2)), key=lambda fv: fv[0])
            for fv1, fv2 in zip(fields_and_values1, fields_and_values2):
                assert fv1[0] == fv2[0], \
                    "Keys are different {} <-> {}".format(fv1[0], fv2[0])
                assert_same_structure(fv1[1], fv2[1])


def map_structure(func, *nests):
    """Applies func to each entry in structure and returns a new structure."""
    assert nests, "There should be at least one input nest!"
    for nest in nests[1:]:
        assert_same_structure(nests[0], nest)

    def _map(*nests):
        if not is_nested(nests[0]):
            return func(*nests)
        if isinstance(nests[0], list) or is_unnamedtuple(nests[0]):
            ret = type(nests[0])([_map(*values) for values in zip(*nests)])
        else:
            ret = {}
            for fields_and_values in zip(
                    *[extract_fields_from_nest(nest) for nest in nests]):
                field = fields_and_values[0][0]
                values = map(lambda fv: fv[1], fields_and_values)
                ret[field] = _map(*values)
            ret = type(nests[0])(**ret)
        return ret

    return _map(*nests)


def pack_sequence_as(nest, flat_seq):
    """Returns a given flattened sequence packed into a given structure."""
    assert len(flatten(nest)) == len(flat_seq), \
        "The two structures have a different number of elements!"

    def _pack(nest, flat_seq):
        if not is_nested(nest):
            return flat_seq.pop(0)
        if isinstance(nest, list) or is_unnamedtuple(nest):
            ret = type(nest)([_pack(value, flat_seq) for value in nest])
        else:
            ret = {}
            for field, value in extract_fields_from_nest(nest):
                ret[field] = _pack(value, flat_seq)
            ret = type(nest)(**ret)
        return ret

    return _pack(nest, flat_seq)


def get_nest_batch_size(nest, dtype=None):
    """Get the batch_size of a nest.

    Args:
        nest (nest): a nested structure
        dtype : a python data type

    Returns:
        batch_size
    """
    flat_seq = flatten(nest)
    assert len(flat_seq) > 0, "Zero element in the nest!"
    batch_size = flat_seq[0].size()[0]
    if dtype is not None:
        batch_size = dtype(batch_size)
    return batch_size


def find_field(nest, name, ignore_empty=True):
    """Find fields with given name.

    Examples:
    ```python
    nest = dict(a=1, b=dict(a=dict(a=2, b=3), b=2))
    find_filed(nest, 'a')
    # you would get [1, {"a": 2, "b": 3}]
    ```

    Args:
        nest (nest): a nest structure
        name (str): name of the field
        ignore_empty (bool): ignore the field if it is None or empty.
    Returns:
        list
    """
    ret = []
    if isinstance(nest, list) or is_unnamedtuple(nest):
        for elem in nest:
            if isinstance(elem, (dict, tuple, list)):
                ret = ret + find_field(elem, name)
    elif isinstance(nest, dict) or is_namedtuple(nest):
        for field, elem in extract_fields_from_nest(nest):
            if field == name:
                if ((elem is not None and elem != () and elem != [])
                        or not ignore_empty):
                    ret.append(elem)
            elif isinstance(elem, (dict, tuple, list)):
                ret = ret + find_field(elem, name)
    return ret


@gin.configurable
def nest_concatenate(nest, dim=-1):
    """Concatenate all elements in a nest along the specified axis. It assumes
    that all elements have the same tensor shape. Can be used as a preprocessing
    combiner in `EncodingNetwork`.

    Args:
        nest (nest): a nested structure
        dim (int): the dim along which the elements are concatenated

    Returns:
        tensor (torch.Tensor): the concat result
    """
    return torch.cat(flatten(nest), dim=dim)


def nest_reduce_sum(nest):
    """Add all elements in a nest together. It assumes that all elements have
    the same tensor shape. Can be used as a preprocessing combiner in
    `EncodingNetwork`.

    Args:
        nest (nest): a nested structure

    Returns:
        tensor (torch.Tensor):
    """
    return torch.sum(torch.stack(flatten(nest), dim=0), dim=0)
