# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""Alf configuration utilities."""

from absl import logging
import functools
import gin
import inspect
from inspect import Parameter

__all__ = [
    'config', 'config1', 'configurable', 'get_all_config_names',
    'get_operative_configs', 'get_inoperative_configs'
]


def config(prefix_or_dict, mutable=True, **kwargs):
    """Set the values for the configs with given name as suffix.

    Example:

    Assume we have the following decorated functions and classes:

    .. code-block:: python

        @alf.configurable
        def cool_func(param1, cool_arg1='a default value', cool_arg2=3):
            ...

        @alf.configurable
        def dumb_func(param1, a=1, b=2):
            ...

        @alf.configurable
        class Worker(obj):
            def __init__(self, job1=1, job2=2):
                ...

            def func(self, a, b):
                ...

    We can config in the following ways:

    .. code-block::

        alf.config('cool_func', cool_arg2='new_value', cool_arg2='another_value')
        alf.config('Worker.func', b=3)
        alf.config('func', b=3)     # 'Worker.func' can be uniquely identified by 'func'
        alf.config({
            'dumb_func.b': 3,
            'Worker.job1': 2
        })


    Args:
        prefix_or_dict (str|dict): if a dict, each (key, value) pair in it
            specifies the value for a config with name key. If a str, it is used
            as prefix so that each (key, value) pair in kwargs specifies the
            value for config with name ``prefix + '.' + key``
        mutable (bool): whether the config can be changed later. If the user
            tries to change an existing immutable config, the change will be
            ignored and a warning will be generated. You can always change a
            mutable config. ``ValueError`` will be raised if trying to set a new
                immutable value to an existing immutable value.
        **kwargs: only used if ``prefix_or_dict`` is a str.
    """
    if isinstance(prefix_or_dict, str):
        prefix = prefix_or_dict
        configs = dict([(prefix + '.' + k, v) for k, v in kwargs.items()])
    elif isinstance(prefix_or_dict, dict):
        assert len(kwargs) == 0, ("**kwargs should not be provided when "
                                  "'prefix_or_dict' is a dict")
        configs = prefix_or_dict
    else:
        raise ValueError(
            "Unsupported type for 'prefix_or_dict': %s" % type(prefix_or_dict))
    for key, value in configs.items():
        config1(key, value, mutable)


def get_all_config_names():
    """Get the names of all configurable values."""
    return sorted([name for name, config in _get_all_leaves(_CONF_TREE)])


def get_operative_configs():
    """Get all the configs that have been used.

    A config is operative if a function call does not explicitly specify the value
    of that config and hence its default value or the value provided through
    alf.config() needs to be used.

    Returns:
        list[tuple[config_name, Any]]
    """
    configs = [(name, config.get_effective_value())
               for name, config in _get_all_leaves(_CONF_TREE)
               if config.is_used()]
    return sorted(configs, key=lambda x: x[0])


def get_inoperative_configs():
    """Get all the configs that have not been used.

    A config is inoperative if its value has been set through ``alf.config()``
    but its set value has never been used by any function calls.

    Returns:
        list[tuple[config_name, Any]]
    """
    configs = [(name, config.get_value())
               for name, config in _get_all_leaves(_CONF_TREE)
               if config.is_configured() and not config.is_used()]
    return sorted(configs, key=lambda x: x[0])


def _get_all_leaves(conf_dict):
    """
    Returns:
        list[tupe[path, _Config]]
    """
    leaves = []
    for k, v in conf_dict.items():
        if not isinstance(v, dict):
            leaves.append((k, v))
        else:
            leaves.extend(
                [(name + '.' + k, node) for name, node in _get_all_leaves(v)])
    return leaves


class _Config(object):
    """Object representing one configurable value."""

    def __init__(self):
        self._configured = False
        self._used = False
        self._has_default_value = False
        self._mutable = True

    def set_default_value(self, value):
        self._default_value = value
        self._has_default_value = True

    def get_default_value(self):
        return self._default_value

    def is_configured(self):
        return self._configured

    def set_mutable(self, mutable):
        self._mutable = mutable

    def is_mutable(self):
        return self._mutable

    def set_value(self, value):
        self._configured = True
        self._value = value

    def get_value(self):
        assert self._configured
        return self._value

    def get_effective_value(self):
        assert self._configured or self._has_default_value
        return self._value if self._configured else self._default_value

    def set_used(self):
        self._used = True

    def is_used(self):
        return self._used


# _CONF_TREE is a suffix tree. For a name such as "abc.def.ghi", the corresponding
# node can be found using _CONF_TREE['ghi']['def']['abc']
_CONF_TREE = {}


def config1(config_name, value, mutable=True):
    """Set one configurable value.

    Args:
        config_name (str): name of the config
        value (any): value of the config
        mutable (bool): whether the config can be changed later. If the user
            tries to change an existing immutable config, the change will be
            ignored and a warning will be generated. You can always change a
            mutable config. ``ValueError`` will be raised if trying to set a new
            immutable value to an existing immutable value.


    """
    tree = _CONF_TREE
    path = config_name.split('.')
    for name in reversed(path):
        if not isinstance(tree, dict) or name not in tree:
            raise ValueError("Cannot find config name %s" % config_name)
        tree = tree[name]

    if isinstance(tree, dict):
        leaves = _get_all_leaves(tree)
        if len(leaves) > 1:
            # only show at most 3 ambiguous choices
            leaves = leaves[:3]
            names = [name + '.' + config_name for name, node in leaves]
            raise ValueError("config name '%s' is ambiguous. There are %s" %
                             (config_name, names))

        assert len(leaves) == 1
        config_node = leaves[0][1]
    else:
        config_node = tree

    if config_node.is_used():
        raise ValueError(
            "Config '%s' has already been used. You should config "
            "its value before using it." % config_name)
    if config_node.is_configured():
        if config_node.is_mutable():
            logging.warning(
                "The value of config '%s' has been configured to %s. It is "
                "replaced by the new value %s" %
                (config_name, config_node.get_value(), value))
            config_node.set_value(value)
            config_node.set_mutable(mutable)
        else:
            if not mutable:
                raise ValueError(
                    "The config '%s' has been configured to an immutable value "
                    "of %s. You cannot change it to a new immutable value")
            logging.warning(
                "The config '%s' has been configured to an immutable value "
                "of %s. The new value %s will be ignored" %
                (config_name, config_node.get_value(), value))
    else:
        config_node.set_value(value)
        config_node.set_mutable(mutable)


def _make_config(signature, whitelist, blacklist):
    """Create a dictionary of _Config for given signature.

    Args:
        signature (inspec.Signature): function signature
        whitelist (list[str]): allowed configurable argument names
        blacklist (list[str]): disallowed configurable argument names
    Returns:
        dict: name => _Config
    """
    configs = {}
    for name, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD):
            continue
        if ((not blacklist and not whitelist)
                or (whitelist and name in whitelist)
                or (blacklist and name not in blacklist)):
            config = _Config()
            configs[name] = config
            if param.default is not inspect.Parameter.empty:
                config.set_default_value(param.default)

    return configs


def _add_to_conf_tree(module_path, func_name, arg_name, node):
    """Add a config object to _CONF_TREE.

    Args:
        module_path (list[str]): module path of this function
        func_name (str): name of the function
        node (_Config): config object for this value
        arg_name: (str): name of the argument
    """

    tree = _CONF_TREE
    path = module_path + func_name.split('.') + [arg_name]
    names = []
    for name in reversed(path[1:]):
        if not isinstance(tree, dict):
            raise ValueError("'%s' conflicts with existing config name '%s'" %
                             ('.'.join(path), '.'.join(names)))
        if name not in tree:
            tree[name] = {}
        tree = tree[name]
        names.insert(0, name)

    if not isinstance(tree, dict):
        raise ValueError("'%s' conflicts with existing config name '%s'" %
                         ('.'.join(path), '.'.join(names)))
    if path[0] in tree:
        if isinstance(tree[path[0]], dict):
            leaves = _get_all_leaves(tree)
            raise ValueError(
                "'%s' conflicts with existing config name '%s'" %
                ('.'.join(path), '.'.join([leaves[0][0]] + names)))
        else:
            raise ValueError("'%s' has already been defined." % '.'.join(path))

    tree[path[0]] = node


def _find_class_construction_fn(cls):
    """Find the first __init__ or __new__ method in the given class's MRO.

    Adapted from gin-config/gin/config.py
    """
    for base in type.mro(cls):
        if '__init__' in base.__dict__:
            return base.__init__
        if '__new__' in base.__dict__:
            return base.__new__


def _ensure_wrappability(fn):
    """Make sure `fn` can be wrapped cleanly by functools.wraps.

    Adapted from gin-config/gin/config.py
    """
    # Handle "builtin_function_or_method", "wrapped_descriptor", and
    # "method-wrapper" types.
    unwrappable_types = (type(sum), type(object.__init__),
                         type(object.__call__))
    if isinstance(fn, unwrappable_types):
        # pylint: disable=unnecessary-lambda
        wrappable_fn = lambda *args, **kwargs: fn(*args, **kwargs)
        wrappable_fn.__name__ = fn.__name__
        wrappable_fn.__doc__ = fn.__doc__
        wrappable_fn.__module__ = ''  # These types have no __module__, sigh.
        wrappable_fn.__wrapped__ = fn
        return wrappable_fn

    # Otherwise we're good to go...
    return fn


def _make_wrapper(fn, configs, signature, has_self):
    """Wrap the function.

    Args:
        fn (Callable): function to be wrapped
        configs (dict[_Config]): config associated with the arguments of function
            ``fn``
        signature (inspect.Signature): Signature object of ``fn``. It is provided
            as an argument so that we don't need to call ``inspect.signature(fn)``
            repeatedly, whith is expensive.
        has_self (bool): whether the first argument is expected to be self but
            signature does not contains parameter for self. This should be True
            if fn is __init__() function of a class.
    Returns:
        The wrapped function
    """

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        unspecified_positional_args = []
        unspecified_kw_args = {}
        num_positional_args = len(args)
        num_positional_args -= has_self

        for i, (name, param) in enumerate(signature.parameters.items()):
            config = configs.get(name, None)
            if config is None:
                continue
            elif i < num_positional_args:
                continue
            elif param.kind in (Parameter.VAR_POSITIONAL,
                                Parameter.VAR_KEYWORD):
                continue
            elif param.kind == Parameter.POSITIONAL_ONLY:
                if config.is_configured():
                    unspecified_positional_args.append(config.get_value)
                    config.set_used()
            elif name not in kwargs and param.kind in (
                    Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY):
                if config.is_configured():
                    unspecified_kw_args[name] = config.get_value()
                config.set_used()

        return fn(*args, *unspecified_positional_args, **kwargs,
                  **unspecified_kw_args)

    return _wrapper


def _decorate(fn_or_cls, name, whitelist, blacklist):
    """decorate a function or class.

    Args:
        fn_or_cls (Callable): a function or a class
        name (str): name for the function. If None, ``fn_or_cls.__qualname__``
            will be used.
        whitelist (list[str]): A whitelisted set of kwargs that should be configurable.
            All other kwargs will not be configurable. Only one of ``whitelist`` or
            `blacklist` should be specified.
        blacklist (list[str]): A blacklisted set of kwargs that should not be
            configurable. All other kwargs will be configurable. Only one of
            ``whitelist` or ``blacklist`` should be specified.
    Returns:
        The decorated function
    """
    signature = inspect.signature(fn_or_cls)
    configs = _make_config(signature, whitelist, blacklist)

    if name is None or '.' not in name:
        module_path = fn_or_cls.__module__.split('.')
    else:
        parts = name.split('.')
        module_path = parts[:-1]
        name = parts[-1]

    if name is None:
        name = fn_or_cls.__qualname__

    for arg_name, node in configs.items():
        _add_to_conf_tree(module_path, name, arg_name, node)

    if inspect.isclass(fn_or_cls):
        # cannot use _make_wrapper() directly on fn_or_cls. This is because
        # _make_wrapper() returns a function. But we want to return a class.
        construction_fn = _find_class_construction_fn(fn_or_cls)
        has_self = construction_fn.__name__ != '__new__'
        decorated_fn = _make_wrapper(
            _ensure_wrappability(construction_fn), configs, signature,
            has_self)
        if construction_fn.__name__ == '__new__':
            decorated_fn = staticmethod(decorated_fn)
        setattr(fn_or_cls, construction_fn.__name__, decorated_fn)
        return fn_or_cls
    else:
        return _make_wrapper(fn_or_cls, configs, signature, has_self=0)


def configurable(fn_or_name=None, whitelist=[], blacklist=[]):
    """Decorator to make a function or class configurable.

    This decorator registers the decorated function/class as configurable, which
    allows its parameters to be supplied from the global configuration (i.e., set
    through ``alf.config()``). The decorated function is associated with a name in
    the global configuration, which by default is simply the name of the function
    or class, but can be specified explicitly to avoid naming collisions or improve
    clarity.

    If some parameters should not be configurable, they can be specified in
    ``blacklist``. If only a restricted set of parameters should be configurable,
    they can be specified in ``whitelist``.

    The decorator can be used without any parameters as follows:

    .. code-block: python

        @alf.configurable
        def my_function(param1, param2='a default value'):
            ...

    In this case, the function is associated with the name
    'my_function' in the global configuration, and both param1 and param2 are
    configurable.

    The decorator can be supplied with parameters to specify the configurable name
    or supply a whitelist/blacklist:

    .. code-block: python

        @alf.configurable('my_func', whitelist=['param2'])
        def my_function(param1, param2='a default value'):
            ...

    In this case, the configurable is associated with the name 'my_func' in the
    global configuration, and only param2 is configurable.

    Classes can be decorated as well, in which case parameters of their
    constructors are made configurable:

    .. code-block:: python

        @alf.configurable
        class MyClass(object):
            def __init__(self, param1, param2='a default value'):
                ...

    In this case, the name of the configurable is 'MyClass', and both `param1`
    and `param2` are configurable.

    The full name of a configurable value is MODULE_PATH.FUNC_NAME.ARG_NAME. It
    can be referred using any suffixes as long as there is no ambiguity. For
    example, assuming there are two configurable values "abc.def.func.a" and
    "xyz.uvw.func.a", you can use "abc.def.func.a", "def.func.a", "xyz.uvw.func.a"
    or "uvw.func.a" to refer these two configurable values. You cannot use
    "func.a" because of the ambiguity. Because of this, you cannot have a config
    name which is the strict suffix of another config name. For example,
    "A.Test.arg" and "Test.arg" cannot both be defined. You can supply a different
    name for the function to avoid conflict:

    .. code-block:: python

        @alf.configurable("NewTest")
        def Test(arg):
            ...

    or

    .. code-block:: python

        @alf.configurable("B.Test")
        def Test(arg):
            ...


    Note: currently, to maintain the compatibility with gin-config, all the
    functions decorated using alf.configurable are automatically configurable
    using gin. The values specified using ``alf.config()`` will override
    values specified through gin.

    Args:
        fn_or_name (Callable|str): A name for this configurable, or a function
            to decorate (in which case the name will be taken from that function).
            If not set, defaults to the name of the function/class that is being made
            configurable. If a name is provided, it may also include module components
            to be used for disambiguation. If the module components is provided,
            the original module name of the function will not be used to compose
            the full name.
        whitelist (list[str]): A whitelisted set of kwargs that should be configurable.
            All other kwargs will not be configurable. Only one of ``whitelist`` or
            ``blacklist`` should be specified.
        blacklist (list[str]): A blacklisted set of kwargs that should not be
            configurable. All other kwargs will be configurable. Only one of
            ``whitelist`` or ``blacklist`` should be specified.
    Returns:
        decorated function if fn_or_name is Callable.
        a decorator if fn is not Callable.
    Raises:
        ValueError: If a configurable with ``name`` (or the name of `fn_or_cls`)
            already exists, or if both a whitelist and blacklist are specified.
    """

    if callable(fn_or_name):
        name = None
    else:
        name = fn_or_name

    gin_ret = gin.configurable(
        fn_or_name, whitelist=whitelist, blacklist=blacklist)

    if whitelist and blacklist:
        raise ValueError("Only one of 'whitelist' and 'blacklist' can be set.")

    if not callable(fn_or_name):

        def _decorator(fn_or_cls):
            fn_or_cls = gin_ret(fn_or_cls)
            return _decorate(fn_or_cls, name, whitelist, blacklist)

        return _decorator
    else:
        return _decorate(gin_ret, name, whitelist, blacklist)
