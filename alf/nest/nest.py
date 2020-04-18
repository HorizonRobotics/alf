# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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

from absl import logging

import torch


def assert_same_type(value1, value2):
    assert type(value1) == type(value2), \
        "Different types! {} <-> {}".format(type(value1), type(value2))


def assert_same_length(seq1, seq2):
    assert len(seq1) == len(seq2), \
        "Different lengths! {} <-> {}".format(len(seq1), len(seq2))


def is_namedtuple(value):
    """Whether the value is a namedtuple instance.

    Args:
        value (Object):
    Returns:
        ``True`` if the value is a namedtuple instance.
    """

    return isinstance(value, tuple) and hasattr(value, '_fields')


def is_unnamedtuple(value):
    """Whether the value is an unnamedtuple instance."""
    return isinstance(value, tuple) and not is_namedtuple(value)


def extract_fields_from_nest(nest):
    """Extract fields and the corresponding values from a nest if it's either
    a ``namedtuple`` or ``dict``.

    Args:
        nest (nest): a nested structure

    Returns:
        Iterable: an iterator that generates ``(field, value)`` pairs. The fields
        are sorted before being returned.

    Raises:
        AssertionError: if the nest is neither ``namedtuple`` nor ``dict``.
    """
    assert is_namedtuple(nest) or isinstance(nest, dict), \
        "Nest {} must be a dict or namedtuple!".format(nest)
    fields = nest.keys() if isinstance(nest, dict) else nest._fields
    for field in sorted(fields):
        value = nest[field] if isinstance(nest, dict) else getattr(nest, field)
        yield field, value


def extract_any_leaf_from_nest(nest):
    """Extract an arbitrary leaf from a nest. Should be faster than doing
    ``flatten(nest)[0]`` because this function has short circuit.

    Args:
        nest (nest): a nested structure

    Returns:
        A ``Tensor`` of there exists a leaf; otherwise ``None``.
    """
    if not is_nested(nest):
        return nest
    if isinstance(nest, list) or is_unnamedtuple(nest):
        for value in nest:
            ret = extract_any_leaf_from_nest(value)
            if ret is not None:
                return ret
    else:
        for _, value in extract_fields_from_nest(nest):
            ret = extract_any_leaf_from_nest(value)
            if ret is not None:
                return ret


def is_nested(value):
    """Returns true if the input is one of: ``list``, ``unnamedtuple``, ``dict``,
    or ``namedtuple``. Note that this definition is different from tf's is_nested
    where all types that are ``collections.abc.Sequence`` are defined to be nested.
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
        try:
            assert_same_type(nest1, nest2)
            assert_same_length(nest1, nest2)
        except AssertionError as e:
            logging.error(str(e))
            raise AssertionError(
                "assert_same_structure() fails between {} and {}".format(
                    nest1, nest2))

        if isinstance(nest1, list) or is_unnamedtuple(nest1):
            for value1, value2 in zip(nest1, nest2):
                assert_same_structure(value1, value2)
        else:
            for fv1, fv2 in zip(
                    extract_fields_from_nest(nest1),
                    extract_fields_from_nest(nest2)):
                assert fv1[0] == fv2[0], \
                    "Keys are different !{} <-> {}".format(fv1[0], fv2[0])
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


def map_structure_up_to(shallow_nest, func, *nests):
    """
    Applies a function to ``nests`` up to the depths of ``shallow_nest``. Every
    sub-nest of each of ``nests`` beyond the depth of the corresponding sub-nest
    in ``shallow_nest`` will be treated as a leaf and input to ``func``.

    Examples (taken from ``tensorflow.nest.map_structure_up_to``):

        .. code-block:: python

            shallow_nest = [None, None]
            inp_val = [[1], 2]
            out = map_structure_up_to(shallow_nest, lambda x: 2 * x, inp_val)
            # Output is: [[1, 1], 4]

            ab_tuple = collections.namedtuple("ab_tuple", "a, b")
            op_tuple = collections.namedtuple("op_tuple", "add, mul")
            inp_val = ab_tuple(a=2, b=3)
            inp_ops = ab_tuple(a=op_tuple(add=1, mul=2), b=op_tuple(add=2, mul=3))
            out = map_structure_up_to(inp_val, lambda val, ops: (val + ops.add) * ops.mul,
                                        inp_val, inp_ops)
            # Output is: ab_tuple(a=6, b=15)

            data_list = [[2, 4, 6, 8], [[1, 3, 5, 7, 9], [3, 5, 7]]]
            name_list = ['evens', ['odds', 'primes']]
            out = map_structure_up_to(
                name_list,
                lambda name, sec: "first_{}_{}".format(len(sec), name),
                name_list, data_list)
            # Output is: ['first_4_evens', ['first_5_odds', 'first_3_primes']]

    Args:
        shallow_nest (nest): a shallow nested structure.
        func (Callable): callable which will be applied to ``nests``.
        *nests (nest): a variable length of nested structures.

    Returns:
        nest: a result nested structure that has the same depths with
        ``shallow_nest``.
    """
    assert nests, "There should be at least one input nest!"

    def _map(shallow_nest, *nests):
        if not is_nested(shallow_nest):
            return func(*nests)
        for nest in nests:
            assert_same_type(shallow_nest, nest)
            assert_same_length(shallow_nest, nest)
        nests = [shallow_nest] + list(nests)
        if isinstance(shallow_nest, list) or is_unnamedtuple(shallow_nest):
            ret = [_map(values[0], *values[1:]) for values in zip(*nests)]
            ret = type(shallow_nest)(ret)
        else:
            ret = {}
            for fields_and_values in zip(
                    *[extract_fields_from_nest(nest) for nest in nests]):
                fields = list(map(lambda fv: fv[0], fields_and_values))
                assert fields.count(fields[0]) == len(fields), \
                    "Fields are not all the same {}".format(fields)
                values = list(map(lambda fv: fv[1], fields_and_values))
                ret[fields[0]] = _map(values[0], *values[1:])
            ret = type(shallow_nest)(**ret)
        return ret

    try:
        return _map(shallow_nest, *nests)
    except AssertionError as e:
        logging.error(str(e))
        raise AssertionError(
            "map_structure_up_to() fails for a shallow_nest {} with nests {}".
            format(shallow_nest, nests))


def fast_map_structure_flatten(func, structure, *flat_structure):
    """Applies func to each entry in structure and returns a flattened structure."""
    entries = zip(*flat_structure)
    return pack_sequence_as(structure, [func(*x) for x in entries])


def fast_map_structure(func, *structure):
    """map_structure using ``pack_sequence_as()``."""
    flat_structure = [flatten(s) for s in structure]
    entries = zip(*flat_structure)

    return pack_sequence_as(structure[0], [func(*x) for x in entries])


def pack_sequence_as(nest, flat_seq):
    """Returns a given flattened sequence packed into a given structure."""
    assert_same_length(flatten(nest), flat_seq)
    flat_seq = list(flat_seq)

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

    flat_seq = list(flat_seq)
    return _pack(nest, flat_seq)


def batch_nested_tensor(nested_tensor):
    """Unsqueeze a zero (batch) dimention for each entry in ``nested_tensor``."""
    return map_structure(lambda x: torch.unsqueeze(x, dim=0), nested_tensor)


def unbatch_nested_tensor(nested_tensor):
    """Squeeze the first (batch) dimension of each entry in ``nested_tensor``."""
    return map_structure(lambda x: torch.squeeze(x, dim=0), nested_tensor)


def get_nest_shape(nest):
    """Get the shape of a nest leaf. It assumes that all nodes of the nest share
    the same shape. For efficiency we don't do a check here.

    Args:
        nest (nest): a nested structure

    Returns:
        torch.Size:
    """
    leaf = extract_any_leaf_from_nest(nest)
    assert leaf is not None, "Zero element in the nest!"
    return leaf.shape


def get_nest_size(nest, dim):
    """Get the size of dimension ``dim`` from a nest.
    It assumes that all nodes of the nest share the same size.

    Args:
        nest (nest): a nested structure
        dim (int): the dimension to get the size for

    Returns:
        int: size of ``dim``
    """
    return get_nest_shape(nest)[dim]


def get_nest_batch_size(nest):
    """Get the batch size (dim=0) of a nest, assuming batch-major.

    Args:
        nest (nest): a nested structure
    Returns:
        int: batch size
    """
    return get_nest_size(nest, dim=0)


def find_field(nest, name, ignore_empty=True):
    """Find fields with given name.

    Examples:

        .. code-block:: python

            nest = dict(a=1, b=dict(a=dict(a=2, b=3), b=2))
            find_filed(nest, 'a')
            # you would get [1, {"a": 2, "b": 3}]

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


def prune_nest_like(nest, slim_nest, value_to_match=None):
    """Prune a nested structure referring to another slim nest. Generally, for
    every corrsponding node, we only keep the fields that're contained in
    ``slim_nest``. In addition, if a field of ``slim_nest`` contains a value of
    ``value_to_match``, then the corresponding field of ``nest`` will also be
    updated to this value.

    .. note::

        If a node is a ``list`` or ``unnamedtuple``, then we require their
        lengths are equal.

    Examples:

        .. code-block:: python

            x = dict(a=1, b=2)
            y = dict(a=TensorSpec(()))
            z = prune_nest_like(x, y) # z is dict(a=1)

            y2 = dict(a=TensorSpec(()), b=())
            z2 = prune_nest_like(x, y2, value_to_match=()) # z2 is dict(a=1, b=())

    Args:
        nest (nest): a nested structure
        slim_nest (nest): a slim nested structure. It's required that at every
            node, its fields is a subset of those of ``nest``.
        value_to_match (nest): a value that indicates the paired field of
            ``slim_nest`` should be updated in ``nest``. Can be set to the default
            value of a ``namedtuple``.

    Returns:
        nest: the pruned nest that has the same set of fields with ``slim_nest``.
    """

    def _prune(nest, slim_nest):
        if is_nested(nest) or is_nested(slim_nest):
            assert_same_type(nest, slim_nest)
            if isinstance(nest, list) or is_unnamedtuple(nest):
                assert len(nest) == len(slim_nest), \
                    "{} should have the same length with {}".format(
                        nest, slim_nest)
                ret = type(nest)([
                    sn if sn == value_to_match else _prune(n, sn)
                    for n, sn in zip(nest, slim_nest)
                ])
            else:
                ret = {}
                nest_fields_values = dict(extract_fields_from_nest(nest))
                for field, slim_nest_value in extract_fields_from_nest(
                        slim_nest):
                    if field not in nest_fields_values:
                        raise ValueError("Field '%s' not in nest!" % field)
                    nest_value = nest_fields_values[field]
                    if slim_nest_value != value_to_match:
                        ret[field] = _prune(nest_value, slim_nest_value)
                    else:
                        ret[field] = slim_nest_value

                ret = type(nest)(**ret)
            return ret
        else:
            return nest

    try:
        return _prune(nest, slim_nest)
    except AssertionError as e:
        logging.error(str(e))
        raise AssertionError(
            "prune_nest_like() fails between {} and {}".format(
                nest, slim_nest))
