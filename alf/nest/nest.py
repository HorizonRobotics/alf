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
"""Functions for handling nest."""
from typing import Union, List, Tuple, Dict

from absl import logging

import cnest

import torch

from typing import Any

# For easier type annotation with nests
Nest = Any

# yapf: disable
NestedTensor = Union[
    torch.Tensor,
    List['NestedTensor'],
    # An empty tuple is also considered a NestedTensor
    Tuple[()],
    # Though Tuple['NestedTensor', ...] is not the tightest specification, it is here
    # to cover the case of "(named) tuple of NestedTensor".
    Tuple['NestedTensor', ...],
    Dict[str, 'NestedTensor']
]
# yapf: enable


def flatten(nest):
    """(C++) Returns a flat list from a given nested structure.

    Note that the order of the values is the same as the alphabetical order of the
    fields.
    """
    try:
        return cnest.flatten(nest)
    except Exception as e:
        logging.error("flatten() fails for {}. Error message: '{}'".format(
            nest, str(e)))
        raise e


def assert_same_structure(nest1, nest2):
    """(C++) Asserts that two structures are nested in the same way."""
    try:
        cnest.assert_same_structure(nest1, nest2)
    except Exception as e:
        paths = tuple(_get_all_paths(nst) for nst in (nest1, nest2))
        logging.error(
            "assert_same_structure() fails for {} and {}. Error message: '{}'"
            "nest1 has paths {}. nest2 has paths {}.".format(
                nest1, nest2, str(e), paths[0], paths[1]))
        raise e


def map_structure(func, *nests):
    """(C++) Applies func to each entry in structure and returns a new structure."""
    try:
        return cnest.map_structure(func, *nests)
    except Exception as e:
        paths = tuple(_get_all_paths(nst) for nst in nests)
        logging.error("map_structure() fails for {}. Error message: '{}'. "
                      "The paths in nests are {}.".format(
                          nests, str(e), paths))
        raise e


def map_structure_without_check(func, *nests):
    """(C++) Applies func to each entry in structure and returns a new structure.
    This function doesn't do any check for efficiency.
    """
    try:
        return cnest.map_structure_without_check(func, *nests)
    except Exception as e:
        logging.error(
            "map_structure_without_check() fails for {}. Error message: '{}'".
            format(nests, str(e)))
        raise e


def pack_sequence_as(nest, flat_seq):
    """(C++) Returns a given flattened sequence packed into a given structure."""
    try:
        return cnest.pack_sequence_as(nest, flat_seq)
    except Exception as e:
        logging.error(
            "pack_sequence_as() fails for {} and {}. Error message: '{}'".
            format(nest, flat_seq, str(e)))
        raise e


def flatten_up_to(shallow_nest, nest):
    """(C++) Flatten ``nests`` up to the depths of ``shallow_nest``. Every
    sub-nest of each of ``nests`` beyond the depth of the corresponding sub-nest
    in ``shallow_nest`` will be treated as a leaf that stops flattening downwards.
    """
    try:
        return cnest.flatten_up_to(shallow_nest, nest)
    except Exception as e:
        logging.error(
            "flatten_up_to() fails for {} and {}. Error message: '{}'".format(
                shallow_nest, nest, str(e)))
        raise e


def map_structure_up_to(shallow_nest, func, *nests):
    """(C++)
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
    try:
        return cnest.map_structure_up_to(shallow_nest, func, *nests)
    except Exception as e:
        logging.error(
            ("map_structure_up_to() fails for a shallow_nest {} with nests {}."
             " Error message: '{}'").format(shallow_nest, nests, str(e)))
        raise e


def assert_same_structure_up_to(shallow_nest, deep_nest):
    """(C++)
    Asserts that ``deep_nest`` has same structure as ``shallow_nest`` up the
    depths of ``shallow_nest``.  Every sub-nest of each of ``nests`` beyond the
    depth of the corresponding sub-nest in ``shallow_nest`` will be treated as a
    leaf.

    Examples:

    .. code-block:: python

        assert_same_structure_up_to(([2], None), ([1], [1, 2, 3]))
        # success

        assert_same_structure_up_to(([2], []), ([1], [1, 2, 3]))
        # failure

    Args:
        shallow_nest (nest): a shallow nested structure.
        deep_nest (nest): a variable length of nested structures.
    """
    try:
        cnest.map_structure_up_to(shallow_nest, lambda _: None, deep_nest)
    except Exception as e:
        logging.error(
            "assert_same_structure_up_to() fails for {} and {}. Error message: "
            "'{}'".format(shallow_nest, deep_nest, str(e)))
        raise e


def prune_nest_like(nest, slim_nest, value_to_match=None):
    """(C++)
    Prune a nested structure referring to another slim nest. Generally, for
    every corresponding node, we only keep the fields that're contained in
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
    try:
        return cnest.prune_nest_like(nest, slim_nest, value_to_match)
    except Exception as e:
        logging.error(
            "prune_nest_like() fails between {} and {}. Error message: '{}'".
            format(nest, slim_nest, str(e)))
        raise e


def assert_same_type(value1, value2):
    assert (type(value1) == type(value2)
            or (isinstance(value1, dict) and isinstance(value2, dict))), (
                "Different types! {} <-> {}".format(
                    type(value1), type(value2)))


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


def extract_fields_from_nest(nest, keep_order=False):
    """Extract fields and the corresponding values from a nest if it's either
    a ``namedtuple`` or ``dict``.

    Args:
        nest (nest): a nested structure
        keep_order: if True, do not sort the fields

    Returns:
        Iterable: an iterator that generates ``(field, value)`` pairs. The fields
        are sorted before being returned.

    Raises:
        AssertionError: if the nest is neither ``namedtuple`` nor ``dict``.
    """
    assert is_namedtuple(nest) or isinstance(nest, dict), \
        "Nest {} must be a dict or namedtuple!".format(nest)
    fields = nest.keys() if isinstance(nest, dict) else nest._fields
    if not keep_order:
        fields = sorted(fields)
    for field in fields:
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


def py_flatten(nest, keep_fields_order=False):
    """Returns a flat list from a given nested structure.

    Args:
        nest: a nested structure
        keep_fields_order: if true, do not sort the fields when flattening for
            namedtuple or dict. For example,
            py_flatten({'b': 1, 'a': 2}, False) -> [2, 1]
            py_flatten({'b': 1, 'a': 2}, True)  -> [1, 2]
    """
    if not is_nested(nest):
        # any other data type will be returned as it is
        return [nest]
    flattened = []
    if isinstance(nest, list) or is_unnamedtuple(nest):
        for value in nest:
            flattened.extend(py_flatten(value))
    else:
        for _, value in extract_fields_from_nest(
                nest, keep_order=keep_fields_order):
            flattened.extend(py_flatten(value))
    return flattened


def py_flatten_up_to(shallow_nest, nest):
    """Flatten ``nests`` up to the depths of ``shallow_nest``. Every sub-nest of
    each of ``nests`` beyond the depth of the corresponding sub-nest in
    ``shallow_nest`` will be treated as a leaf that stops flattening downwards.
    """
    if not is_nested(shallow_nest):
        return [nest]

    try:
        assert_same_type(shallow_nest, nest)
        assert_same_length(shallow_nest, nest)
    except AssertionError as e:
        logging.error(str(e))
        raise AssertionError(
            "Different types or lengths between {} and {}".format(
                shallow_nest, nest))

    flattened = []
    if isinstance(shallow_nest, list) or is_unnamedtuple(shallow_nest):
        for sn, n in zip(shallow_nest, nest):
            flattened.extend(py_flatten_up_to(sn, n))
    else:
        for fv1, fv2 in zip(
                extract_fields_from_nest(shallow_nest),
                extract_fields_from_nest(nest)):
            assert fv1[0] == fv2[0], \
                "Keys are different !{} <-> {}".format(fv1[0], fv2[0])
            flattened.extend(py_flatten_up_to(fv1[1], fv2[1]))
    return flattened


def py_assert_same_structure(nest1, nest2):
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
                py_assert_same_structure(value1, value2)
        else:
            for fv1, fv2 in zip(
                    extract_fields_from_nest(nest1),
                    extract_fields_from_nest(nest2)):
                assert fv1[0] == fv2[0], \
                    "Keys are different !{} <-> {}".format(fv1[0], fv2[0])
                py_assert_same_structure(fv1[1], fv2[1])


def py_map_structure_with_path(func, *nests):
    """Applies func to each entry in structure and returns a new structure.
    This function gives func access to one additional parameter as its first argument:
    the symbolic string of the path to the element currently supplied.
    List elements will be indexed by the ordinal position of the element in the list.
    """
    assert nests, "There should be at least one input nest!"
    for nest in nests[1:]:
        py_assert_same_structure(nests[0], nest)

    def _map(*nests, path=""):
        if not is_nested(nests[0]):
            return func(path, *nests)
        if isinstance(nests[0], list) or is_unnamedtuple(nests[0]):
            ret = type(nests[0])([
                _map(
                    *values[:-1],
                    path=path + ("." if path else "") + str(values[-1]))
                for values in zip(*nests, range(len(nests[0])))
            ])
        else:
            ret = {}
            for fields_and_values in zip(
                    *[extract_fields_from_nest(nest) for nest in nests]):
                field = fields_and_values[0][0]
                values = map(lambda fv: fv[1], fields_and_values)
                ret[field] = _map(
                    *values, path=path + ("." if path else "") + field)
            ret = type(nests[0])(**ret)
        return ret

    return _map(*nests, path="")


def py_map_structure(func, *nests):
    """Applies func to each entry in structure and returns a new structure."""
    assert nests, "There should be at least one input nest!"
    for nest in nests[1:]:
        py_assert_same_structure(nests[0], nest)

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


def py_map_structure_up_to(shallow_nest, func, *nests):
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
    """Applies func to entries of ``flat_structure`` and returns a packed
    structure according to ``structure``."""
    entries = zip(*flat_structure)
    return pack_sequence_as(structure, [func(*x) for x in entries])


def fast_map_structure(func, *structure):
    """map_structure using ``pack_sequence_as()``."""
    flat_structure = [flatten(s) for s in structure]
    entries = zip(*flat_structure)

    return pack_sequence_as(structure[0], [func(*x) for x in entries])


def py_pack_sequence_as(nest, flat_seq, keep_fields_order=False):
    """Returns a given flattened sequence packed into a given structure.

    Args:
        nest: a nested structure
        flat_seq: a flattened sequence to be packed into the same structure as
            ``nest``
        keep_fields_order: if true, do not sort the fields when packing for
            namedtuple or dict. For example,
            py_pack_sequence_as({'b': 1, 'a': 2}, [3, 4], False) -> {'a': 3, 'b': 4}
            py_pack_sequence_as({'b': 1, 'a': 2}, [3, 4], True)  -> {'b': 3, 'a': 4}
    """
    assert_same_length(py_flatten(nest), flat_seq)
    counter = [0]

    def _pack(nest, flat_seq):
        if not is_nested(nest):
            ret = flat_seq[counter[0]]
            counter[0] += 1
            return ret

        if isinstance(nest, list) or is_unnamedtuple(nest):
            ret = type(nest)([_pack(value, flat_seq) for value in nest])
        else:
            ret = {}
            for field, value in extract_fields_from_nest(
                    nest, keep_order=keep_fields_order):
                ret[field] = _pack(value, flat_seq)
            ret = type(nest)(**ret)
        return ret

    return _pack(nest, list(flat_seq))


def batch_nested_tensor(nested_tensor):
    """Unsqueeze a zero (batch) dimension for each entry in ``nested_tensor``."""
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

    def _is_empty(x):
        return isinstance(x, (tuple, list)) and len(x) == 0

    ret = []
    if isinstance(nest, list) or is_unnamedtuple(nest):
        for elem in nest:
            if isinstance(elem, (dict, tuple, list)):
                ret = ret + find_field(elem, name)
    elif isinstance(nest, dict) or is_namedtuple(nest):
        for field, elem in extract_fields_from_nest(nest):
            if field == name:
                if ((elem is not None and not _is_empty(elem))
                        or not ignore_empty):
                    ret.append(elem)
            elif isinstance(elem, (dict, tuple, list)):
                ret = ret + find_field(elem, name)
    return ret


def py_prune_nest_like(nest, slim_nest, value_to_match=None):
    """Prune a nested structure referring to another slim nest. Generally, for
    every corresponding node, we only keep the fields that're contained in
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


def _get_all_paths(nested):
    """Get all paths in nested."""
    return flatten(py_map_structure_with_path(lambda path, x: path, nested))


def get_field(nested, field):
    """Get the field from nested.

    ``field`` is a string separated by ".". ``get_field(nested, "a.b")`` is equivalent
    to ``nested.a.b`` if ``nested`` is constructed using namedtuple or ``nests['a']['b']``
    if nested is constructed using dict. If nested is constructed using list or
    unnamed tuple, ``get_field(nested, "1.2")`` is equivalent to ``nested[1][2]``.

    Args:
        nested (nest): a nested structure
        field (str): indicate the path to the field with '.' separating the field
            name at different level. ``None`` or '' means the whole nest.
    Returns:
        nest: value of the field corresponding to ``field``
    Raises:
        LookupError: if field cannot be found.
    """

    def _traverse(nested, levels):
        if not levels:
            return nested
        level = levels[0]
        if is_namedtuple(nested):
            return _traverse(nested=getattr(nested, level), levels=levels[1:])
        elif isinstance(nested, dict):
            return _traverse(nested=nested[level], levels=levels[1:])
        elif isinstance(nested, (tuple, list)):
            return _traverse(nested=nested[int(level)], levels=levels[1:])
        else:
            raise LookupError()

    try:
        return _traverse(
            nested=nested, levels=field.split('.') if field else [])
    except (AttributeError, LookupError, ValueError):
        raise LookupError(
            "Cannot find path '%s' in nested. nested has paths: %s" %
            (field, _get_all_paths(nested)))


def transform_nest(nested, field, func):
    """Transform the node of a nested structure indicated by ``field`` using
    ``func``.

    This function can be used to update our ``namedtuple`` structure conveniently,
    comparing the following two methods:

        .. code-block:: python

            info = info._replace(rl=info.rl._replace(sac=info.rl.sac * 0.5))

    vs.

        .. code-block:: python

            info = transform_nest(info, 'rl.sac', lambda x: x * 0.5)

    The second method is usually shorter, more intuitive, and less error-prone
    when ``field`` is a long string.

    Args:
        nested (nested Tensor): the structure to be applied the transformation.
        field (str): If a string, it's the field to be transformed, multi-level
            path denoted by "A.B.C". Levels can also be integers (e.g., "0.2"),
            in which case the nest is expected to be tuples or lists at those levels.
            If ``None``, then the root object is transformed.
        func (Callable): transform func, the function will be called as
            ``func(nested)`` and should return a new nest.
    Returns:
        transformed nest
    """

    def _traverse_transform(nested, levels):
        if not levels:
            return func(nested)
        level = levels[0]
        if is_namedtuple(nested):
            new_val = _traverse_transform(
                nested=getattr(nested, level), levels=levels[1:])
            return nested._replace(**{level: new_val})
        elif isinstance(nested, dict):
            new_val = nested.copy()
            new_val[level] = _traverse_transform(
                nested=nested[level], levels=levels[1:])
            return new_val
        elif isinstance(nested, (list, tuple)):
            new_val = list(nested).copy()
            new_val[int(level)] = _traverse_transform(
                nested=nested[int(level)], levels=levels[1:])
            return type(nested)(new_val)
        else:
            raise TypeError("")

    return _traverse_transform(
        nested=nested, levels=field.split('.') if field else [])


def transform_nests(nests, field, func):
    """Transform the node of each of the nest in nests indicated by ``field``
        using ``func``.

    This function can be used to transform multiple nests, and perform
        transformations with inter-nest interactions.

        .. code-block:: python

            res1, res2 = transform_nests([nest1, nest2], 'a.b',
                                lambda x: (x[0] * x[1], x[0] + x[1]))

        where ``x[0]`` denotes the value from ``nest1`` and ``x[1]`` is
        from ``nest2``.
    Args:
        nests ([nested Tensor]): the structure to be applied the transformation.
        field (str): If a string, it's the field to be transformed, multi-level
            path denoted by "A.B.C". If ``None``, then the root object is
            transformed.
        func (Callable): transform func, the function will be called as
            ``func(nested)`` and should return a new nest.
    Returns:
        list of transformed nests, with its length the same as the input nests
    """

    assert len(nests) > 0

    def _traverse_transform(nests, levels):
        if not levels:
            return func(nests)

        type_check = [
            is_namedtuple(nest) or isinstance(nest, dict) for nest in nests
        ]
        assert all(type_check), TypeError(
            "For multiple nested inputs, each of "
            "its elements must be either a dict or namedtuple!")

        level = levels[0]
        if is_namedtuple(nests[0]):
            new_vals = _traverse_transform(
                nests=[getattr(nest, level) for nest in nests],
                levels=levels[1:])
            return [
                nest._replace(**{level: new_val})
                for nest, new_val in zip(nests, new_vals)
            ]
        elif isinstance(nests[0], dict):
            new_nests = [nest.copy() for nest in nests]

            trans_nests_level = _traverse_transform(
                nests=[nest[level] for nest in nests], levels=levels[1:])

            for nest, val in zip(new_nests, trans_nests_level):
                nest[level] = val

            return new_nests

    return _traverse_transform(nests, levels=field.split('.') if field else [])


def set_field(nested, field, new_value):
    """Set the field in nested to ``new_value``.

    field is a string separated by ".". set_filed(nested, "a.b", v) is equivalent
    to ``nested._replace(a=nested.a._replace(b=v))`` if nested is constructed
    using namedtuple.

    Args:
        nested (nest): a nested structure
        field (str): indicate the path to the field with '.' separating the field
            name at different level
        new_value (any): the new value for the field
    Returns:
        nest: a nest same as ``nested`` except the filed ``field`` replaced by
            ``new_value``
    """

    return transform_nest(nested, field, lambda _: new_value)


def transpose(nested: Nest,
              shallow_nest: Nest = None,
              new_shallow_nest: Nest = None):
    """Given a nest ``A`` and its shallow nest ``a``, assuming that each child
    of ``a`` has the same nest structure ``B``, this function
    returns a new nest whose shallow nest ``b`` is a shallow nest of ``B``,
    and each child of ``b`` has a shallow nest ``a``.

    An illustrative graph shows the transpose operation::

        A = a-B = a-b-C (transpose->) b-a-C

    where ``C`` is every (same) child of ``b`` (could be empty).

    .. note::

        ALF defines the "shallow nest" of a nest as the subtree that starts from
        the nest root and contains at least all the direct children of the nest.
        It can optionally contain more descendants of the nest.

    For example,

    .. code-block:: python

        x = [(0, 1), (2, 3), (4, 5)]
        y = transpose(x, shallow_nest=[None, None, None])
        # y will be ``([0, 2, 4], [1, 3, 5])``
        y1 = transpose(x)
        # y1 will be the same with y

        x = NTuple(a=dict(x=3, y=1), b=[dict(x=5, y=10)])
        shallow_nest = NTuple(a=None, b=[False])
        y = transpose(x, shallow_nest)
        # y will be ``dict(x=NTuple(a=3, b=[5]), y=NTuple(a=1, b=[10]))``

        x = NTuple(a=dict(x=3, y=dict(n=1, m=2)),
                   b=dict(x=5, y=dict(n=1, m=3)))
        transposed_nest1 = nest.transpose(x)
        self.assertEqual(transposed_nest1,
                         dict(x=NTuple(a=3, b=5), y=NTuple(a=dict(n=1, m=2),
                                                           b=dict(n=1, m=3))))

    Args:
        nested: a nested structure
        shallow_nest: a nested structure indicating the first "axis" for the
            transpose. If None, then ``nest_top_level(nested)`` will be used.
        new_shallow_nest: a nested structure indicating the second "axis" for
            the transpose. Note that this shallow nest is w.r.t. each child ``B``.
            If not provided, then ``nest_top_level(B)`` will be used.

    Returns:
        nested: a transposed nested structure
    """
    if not is_nested(nested):
        return nested

    if shallow_nest is None:
        shallow_nest = nest_top_level(nested)

    if not is_nested(shallow_nest):
        return nested

    # ``nested`` is ``A`` and each leaf is ``B`` in the docstring
    leaves = flatten_up_to(shallow_nest, nested)
    for leaf in leaves:
        assert_same_structure(leaves[0], leaf)

    if new_shallow_nest is None:
        # this is ``b`` in the docstring
        new_shallow_nest = nest_top_level(leaves[0])
    matrix = [flatten_up_to(new_shallow_nest, leaf) for leaf in leaves]
    transposed_matrix = list(zip(*matrix))
    new_nest = pack_sequence_as(new_shallow_nest, transposed_matrix)
    new_nest = map_structure_up_to(
        new_shallow_nest, lambda flat: pack_sequence_as(shallow_nest, flat),
        new_nest)
    return new_nest


def nest_top_level(nested: Nest):
    """Given a nest, return its top-level structure, where the values are set to
    ``None``.

    Args:
        nested: a nested structure
    """
    if not is_nested(nested):
        return nested
    if isinstance(nested, list) or is_unnamedtuple(nested):
        return type(nested)([None] * len(nested))
    fields_vals = extract_fields_from_nest(nested)
    return type(nested)(**{field: None for field, _ in fields_vals})


def sum_nest(nested: Nest):
    """Sum all elements in a nest.

    Args:
        nested: a nested structure
    """
    return sum(flatten(nested))
