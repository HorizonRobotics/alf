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


def nest_list_to_tuple(nest):
    """Convert the lists in a nest to tuples.

    Some tf-agents function (e.g. ReplayBuffer) cannot accept nest containing
    list. So we need some utitity to convert back and forth.

    Args:
        nest (a nest): a nest structure
    Returns:
        nest with the same content as the input but lists are changed to tuples
    """
    if isinstance(nest, tuple):
        new_nest = tuple(nest_list_to_tuple(item) for item in nest)
        if hasattr(nest, '_fields'):
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
    list. So we need some utitity to convert back and forth.

    Args:
        nest (a nest): a nest structure wihtout list
        example (a nest): the example structure that nest will be converted to
    Returns:
        nest with the same content as the input but some tuples are changed to
        lists
    """
    if isinstance(nest, tuple):
        new_nest = tuple(
            nest_tuple_to_list(nst, exp) for nst, exp in zip(nest, example))
        if hasattr(example, '_fields'):
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
