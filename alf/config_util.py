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
import os
import pprint
import runpy
import shutil

__all__ = [
    'config',
    'config1',
    'configurable',
    'define_config',
    'get_all_config_names',
    'get_config_value',
    'get_handled_pre_configs',
    'get_inoperative_configs',
    'get_operative_configs',
    'import_config',
    'load_config',
    'override_sole_config',
    'pre_config',
    'reset_configs',
    'validate_pre_configs',
    'repr_wrapper',
    'save_config',
]


def GET_ALF_SOLE_CONFIG():
    """Simple getter function for the env variable ALF_SOLE_CONFIG.

    This can be set to 0 or 1 to enable/disable the sole config mode.
    More details in config(...).
    """
    ALF_SOLE_CONFIG = os.getenv("ALF_SOLE_CONFIG", "0")
    assert ALF_SOLE_CONFIG in ["0", "1"]
    return bool(int(ALF_SOLE_CONFIG))


@logging.skip_log_prefix
def config(prefix_or_dict,
           mutable=True,
           raise_if_used=True,
           sole_init=False,
           override_sole_init=False,
           **kwargs):
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

            @alf.configurable
            def func(self, a, b):
                ...

    We can config in the following ways:

    .. code-block::

        alf.config('cool_func', cool_arg1='new_value', cool_arg2='another_value')
        alf.config('Worker.func', b=3)
        alf.config('func', b=3)     # 'Worker.func' can be uniquely identified by 'func'
        alf.config({
            'dumb_func.b': 3,
            'Worker.job1': 2        # now the default value of job1 for Worker() becomes 2.
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
        raise_if_used (bool): If True, ValueError will be raised if trying to
            config a value which has already been used.
        sole_init (bool): If True, the config value can no longer be set again
            after this config call. Any future calls will raise a RuntimeError.
            This is helpful in enforcing a singular point of initialization,
            thus eliminating any potential side effects from possible future
            overrides. For users wanting this to be the default behavior, the
            ALF_SOLE_CONFIG env variable can be set to 1.
        override_sole_init (bool): If True, the value of the config will be set
            regardless of any previous ``sole_init`` setting. This should be used
            only when absolutely necessary (e.g., a teacher-student training loop,
            where the student must override certain configs inherited from the
            teacher). If the config is immutable, a warning will be declared with
            no changes made.
        **kwargs: only used if ``prefix_or_dict`` is a str.
    """
    if isinstance(prefix_or_dict, str):
        assert len(kwargs) > 0, ("**kwargs should be provided when "
                                 "'prefix_or_dict' is a str")
        prefix = prefix_or_dict
        configs = dict([(prefix + '.' + k, v) for k, v in kwargs.items()])
    elif isinstance(prefix_or_dict, dict):
        assert len(kwargs) == 0, ("**kwargs should not be provided when "
                                  "'prefix_or_dict' is a dict")
        configs = prefix_or_dict
    else:
        raise ValueError(
            "Unsupported type for 'prefix_or_dict': %s" % type(prefix_or_dict))

    # If ALF_SOLE_CONFIG is set to 1, sole_init is always True.
    sole_init = sole_init or GET_ALF_SOLE_CONFIG()

    for key, value in configs.items():
        config1(key, value, mutable, raise_if_used, sole_init,
                override_sole_init)


def override_sole_config(prefix_or_dict, **kwargs):
    """Wrapper function for configuring a config with override_sole_init=True.

    This call allows a user to attempt to overwrite a config's value even
    if it is protected by ``sole_init``. It is highly recommended that this be
    used only when absolutely necessary (e.g., a teacher-student training loop,
    where the student must override certain configs inherited from the teacher).
    This call has no effect for configs who are immutable.

    Args:
        prefix_or_dict (str|dict): if a dict, each (key, value) pair in it
            specifies the value for a config with name key. If a str, it is used
            as prefix so that each (key, value) pair in kwargs specifies the
            value for config with name ``prefix + '.' + key``
        **kwargs: only used if ``prefix_or_dict`` is a str.
    """
    config(prefix_or_dict, override_sole_init=True, **kwargs)


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
        list[tuple[path, _Config]]
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
        self._sole_init = False

    def set_default_value(self, value):
        self._default_value = value
        self._has_default_value = True

    def has_default_value(self):
        return self._has_default_value

    def get_default_value(self):
        return self._default_value

    def is_configured(self):
        return self._configured

    def set_mutable(self, mutable):
        self._mutable = mutable

    def is_mutable(self):
        return self._mutable

    def set_sole_init(self, sole_init):
        self._sole_init = sole_init

    def get_sole_init(self):
        return self._sole_init

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

    def reset(self):
        self._used = False
        self._configured = False
        self._mutable = True


# _CONF_TREE is a suffix tree. For a name such as "abc.def.ghi", the corresponding
# node can be found using _CONF_TREE['ghi']['def']['abc']
_CONF_TREE = {}
_PRE_CONFIGS = []
_HANDLED_PRE_CONFIGS = []
_DEFINED_CONFIGS = []
_CONF_FILES = {}  # key: file name, value: content
_CONFIG_MODULES = {}
_IMPORT_STACK = []
_ROOT_CONF_FILE = None


def reset_configs():
    """Reset all the configs to their initial states."""

    def _reset_configs(tree):
        for child in tree.values():
            if isinstance(child, dict):
                _reset_configs(child)
            else:
                child.reset()

    _reset_configs(_CONF_TREE)
    for name in _DEFINED_CONFIGS:
        _remove_config_node(name)

    _DEFINED_CONFIGS.clear()
    _PRE_CONFIGS.clear()
    _HANDLED_PRE_CONFIGS.clear()
    _CONF_FILES.clear()
    _CONFIG_MODULES.clear()
    global _ROOT_CONF_FILE
    _ROOT_CONF_FILE = None


def _remove_config_node(config_name):
    """Remove the _Config object corresponding to config_name."""
    node = _CONF_TREE
    path = config_name.split('.')
    tree_name_pairs = []
    for name in reversed(path):
        tree = node
        if not isinstance(tree, dict) or name not in tree:
            raise ValueError("Cannot find config name %s" % config_name)
        node = tree[name]
        tree_name_pairs.append((tree, name))

    assert isinstance(
        node, _Config), "config_name is not a full path: %s" % config_name
    del tree[name]
    tree_name_pairs.pop()
    for tree, name in reversed(tree_name_pairs):
        if len(tree[name]) == 0:
            del tree[name]
        else:
            break


def _get_config_node(config_name):
    """Get the _Config object corresponding to config_name."""
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

    return config_node


@logging.skip_log_prefix
def config1(config_name,
            value,
            mutable=True,
            raise_if_used=True,
            sole_init=False,
            override_sole_init=False,
            override_all=False):
    """Set one configurable value.

    Args:
        config_name (str): name of the config
        value (any): value of the config
        mutable (bool): whether the config can be changed later. If the user
            tries to change an existing immutable config, the change will be
            ignored and a warning will be generated. You can always change a
            mutable config. ``ValueError`` will be raised if trying to set a new
            immutable value to an existing immutable value.
        raise_if_used (bool): If True, ValueError will be raised if trying to
            config a value which has already been used.
        sole_init (bool): If True, the config value can no longer be set again
            after this config call. Any future calls will raise a RuntimeError.
            This is helpful in enforcing a singular point of initialization,
            thus eliminating any potential side effects from possible future
            overrides. For users wanting this to be the default behavior, the
            ALF_SOLE_CONFIG env variable can be set to 1.
        override_sole_init (bool): If True, the value of the config will be set
            regardless of any previous ``sole_init`` setting. This should be used
            only when absolutely necessary (e.g., a teacher-student training loop,
            where the student must override certain configs inherited from the
            teacher). If the config is immutable, a warning will be declared with
            no changes made.
        override_all (bool): If True, the value of the config will be set regardless
            of any pre-existing ``mutable`` or ``sole_init`` settings. This should
            be used only when absolutely necessary (e.g., adjusting certain configs
            such as mini_batch_size for DDP workers.).
    """
    config_node = _get_config_node(config_name)

    if raise_if_used and config_node.is_used():
        raise ValueError(
            "Config '%s' has already been used. You should config "
            "its value before using it." % config_name)

    if override_all:
        if config_node.get_sole_init():
            logging.warning(
                "The value of config '%s' (%s) is protected by sole_init. "
                "It is now being overridden by the overide_all flag to a new value %s. "
                "Use at your own risk." % (config_name,
                                           config_node.get_value(), value))
        if not config_node.is_mutable():
            logging.warning(
                "The value of config '%s' (%s) is immutable. "
                "It is now being overridden by the overide_all flag to a new value %s. "
                "Use at your own risk." % (config_name,
                                           config_node.get_value(), value))
        config_node.set_value(value)
        return
    elif override_sole_init:
        if config_node.is_configured():
            if not config_node.is_mutable():
                logging.warning(
                    "The value of config '%s' (%s) is immutable. "
                    "Override flag with new value %s is ignored. " %
                    (config_name, config_node.get_value(), value))
                return
            elif config_node.get_sole_init():
                logging.warning(
                    "The value of config '%s' (%s) is protected by sole_init. "
                    "It is now being overridden by the override flag to a new value %s. "
                    "Use at your own risk." % (config_name,
                                               config_node.get_value(), value))
    elif config_node.is_configured():
        if config_node.get_sole_init():
            raise RuntimeError(
                "Config '%s' is protected by sole_init and cannot be reconfigured. "
                "If you wish to set this config value, do so the location of the "
                "previous call." % config_name)

        if config_node.is_mutable():
            logging.warning(
                "The value of config '%s' has been configured to %s. It is "
                "replaced by the new value %s" %
                (config_name, config_node.get_value(), value))
        else:
            logging.warning(
                "The config '%s' has been configured to an immutable value "
                "of %s. The new value %s will be ignored" %
                (config_name, config_node.get_value(), value))
            config_node.set_sole_init(sole_init)
            return

    config_node.set_value(value)
    config_node.set_mutable(mutable)
    if not override_sole_init:
        config_node.set_sole_init(sole_init)


@logging.skip_log_prefix
def pre_config(configs):
    """Preset the values for configs before the module defining it is imported.

    This function is useful for handling the config params from commandline,
    where there are no module imports and hence no config has been defined.

    The value is bound to the config when the module defining the config is
    imported later. ``validate_pre_configs()` should be called after the config
    file has been loaded to ensure that all the pre_configs have been correctly
    bound.

    Args:
        configs (dict): dictionary of config name to value
    """
    for name, value in configs.items():
        try:
            config1(name, value, mutable=False, sole_init=False)
            _HANDLED_PRE_CONFIGS.append((name, value))
        except ValueError:
            _PRE_CONFIGS.append((name, value))


def _handle_pre_configs(path, node):
    def _handle1(item):
        name, value = item
        parts = name.split('.')
        if len(parts) > len(path):
            return True
        for i in range(-len(parts), 0):
            if parts[i] != path[i]:
                return True
        node.set_value(value)
        node.set_mutable(False)
        _HANDLED_PRE_CONFIGS.append(item)
        return False

    global _PRE_CONFIGS
    _PRE_CONFIGS = list(filter(_handle1, _PRE_CONFIGS))


def validate_pre_configs():
    """Validate that all the configs set through ``pre_config()`` are correctly bound."""

    if _PRE_CONFIGS:
        raise ValueError((
            "A pre-config '%s' was not handled, either because its config name "
            +
            "was not found, or there was some error when calling pre_config()")
                         % _PRE_CONFIGS[0][0])

    for (config_name, _) in _HANDLED_PRE_CONFIGS:
        _get_config_node(config_name)


def get_handled_pre_configs():
    """Return a list of handled pre-config ``(name, value)``."""
    return _HANDLED_PRE_CONFIGS


def get_config_value(config_name):
    """Get the value of the config with the name ``config_name``.

    Args:
        config_name (str): name of the config or its suffix which can uniquely
            identify the config.
    Returns:
        Any: value of the config
    Raises:
        ValueError: if the value of the config has not been configured and it
            does not have a default value.
    """
    config_node = _get_config_node(config_name)
    if not config_node.is_configured() and not config_node.has_default_value():
        raise ValueError(
            "Config '%s' is not configured nor has a default value." %
            config_name)

    config_node.set_used()
    return config_node.get_effective_value()


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

    _handle_pre_configs(path, node)


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


def _make_wrapper(fn, configs, signature, has_self, config_only_args):
    """Wrap the function.

    Args:
        fn (Callable): function to be wrapped
        configs (dict[_Config]): config associated with the arguments of function
            ``fn``
        signature (inspect.Signature): Signature object of ``fn``. It is provided
            as an argument so that we don't need to call ``inspect.signature(fn)``
            repeatedly, with is expensive.
        has_self (bool): whether the first argument is expected to be self but
            signature does not contains parameter for self. This should be True
            if fn is __init__() function of a class.
        config_only_args (list[str]): list of args that should be guarded. In other
            words, their values can only be set / changed globally via ``alf.config()``.
            This protects against local untracked changes as a result of 1) using
            ``partial()`` or 2) setting an argument when calling the function, which
            can cause unintended side effects.
    Returns:
        The wrapped function
    """

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        unspecified_positional_args = []
        unspecified_kw_args = {}
        num_positional_args = len(args)
        num_positional_args -= has_self

        set_positional_args = []
        for i, (name, param) in enumerate(signature.parameters.items()):
            config = configs.get(name, None)
            if config is None:
                continue
            elif i < num_positional_args:
                set_positional_args.append(name)
            elif param.kind in (Parameter.VAR_POSITIONAL,
                                Parameter.VAR_KEYWORD):
                continue
            elif param.kind == Parameter.POSITIONAL_ONLY:
                if config.is_configured():
                    unspecified_positional_args.append(config.get_value())
                    config.set_used()
            elif name not in kwargs and param.kind in (
                    Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY):
                if config.is_configured():
                    unspecified_kw_args[name] = config.get_value()
                config.set_used()

        for config_only_arg in config_only_args:
            if config_only_arg in set_positional_args or config_only_arg in kwargs:
                raise ValueError(
                    f"The arg '{config_only_arg}' of {fn.__qualname__} is guarded but has been modified. "
                    f"Most likely partial() was used to change this value, which is not allowed."
                )

        return fn(*args, *unspecified_positional_args, **kwargs,
                  **unspecified_kw_args)

    return _wrapper


def _decorate(fn_or_cls, name, whitelist, blacklist, config_only_args):
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
        config_only_args (list[str]): list of args that should be guarded. In other
            words, their values can only be set / changed globally via ``alf.config()``.
            This protects against local untracked changes as a result of 1) using
            ``partial()`` or 2) setting an argument when calling the function, which
            can cause unintended side effects.
    Returns:
        The decorated function
    """
    signature = inspect.signature(fn_or_cls)
    configs = _make_config(signature, whitelist, blacklist)

    orig_name = name

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
            has_self, config_only_args)
        if construction_fn.__name__ == '__new__':
            decorated_fn = staticmethod(decorated_fn)
        setattr(fn_or_cls, construction_fn.__name__, decorated_fn)
    else:
        fn_or_cls = _make_wrapper(
            fn_or_cls,
            configs,
            signature,
            has_self=0,
            config_only_args=config_only_args)

    if fn_or_cls.__module__ != '<run_path>' and os.environ.get(
            'ALF_USE_GIN', "1") == "1":
        # If a file is executed using runpy.run_path(), the module name is
        # '<run_path>', which is not an acceptable name by gin.
        return gin.configurable(
            orig_name, whitelist=whitelist, blacklist=blacklist)(fn_or_cls)
    else:
        return fn_or_cls


def repr_wrapper(cls):
    """A wrapper for automatically generating readable repr for an object.

    The presentation shows the arguments used to construct of object.
    It does not include the default arguments, nor the class members.

    To use it, simply use it to decorate an class.

    Example:

    .. code-block:: python

        @repr_wrapper
        class MyClass(object):
            def __init__(self, a, b, c=100, d=200):
                pass

        a = MyClass(1, 2)
        assert repr(a) == "MyClass(1, 2)"
        a = MyClass(3, 5, d=300)
        assert repr(a) == "MyClass(1, 2, d=300)"

    """
    assert inspect.isclass(cls)
    signature = inspect.signature(cls)
    construction_fn = _find_class_construction_fn(cls)
    has_self = construction_fn.__name__ != '__new__'
    fn = _ensure_wrappability(construction_fn)
    defaults = {}
    for name, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is not inspect.Parameter.empty:
            defaults[name] = param.default

    setattr(cls, '__repr__', lambda self: self._repr_wrapper_str_)

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        ret = fn(*args, **kwargs)
        if has_self:
            self = args[0]
        else:
            self = ret

        s = []
        for val in args[has_self:]:
            s.append(pprint.pformat(val))
        for k, val in kwargs.items():
            if k not in defaults or val != defaults[k]:
                s.append(k + '=' + pprint.pformat(val))
        l = sum(map(len, s))
        multiline = l > 80 or any(map(lambda x: '\n' in x, s))
        if multiline:
            s = ['  ' + x for x in s]
            self._repr_wrapper_str_ = '%s(\n%s)' % (cls.__qualname__,
                                                    ",\n".join(s))
        else:
            self._repr_wrapper_str_ = '%s(%s)' % (cls.__qualname__,
                                                  ", ".join(s))
        return ret

    decorated_fn = _wrapper
    if construction_fn.__name__ == '__new__':
        decorated_fn = staticmethod(decorated_fn)
    setattr(cls, construction_fn.__name__, decorated_fn)
    return cls


def configurable(fn_or_name=None,
                 whitelist=[],
                 blacklist=[],
                 config_only_args=[]):
    """Decorator to make a function or class configurable.

    This decorator registers the decorated function/class as configurable, which
    allows its parameters to be supplied from the global configuration (i.e., set
    through ``alf.config()``). The decorated function is associated with a name in
    the global configuration, which by default is simply the name of the function
    or class, but can be specified explicitly to avoid naming collisions or improve
    clarity.

    If some parameters should not be configurable, they can be specified in
    ``blacklist``. If only a restricted set of parameters should be configurable,
    they can be specified in ``whitelist``. Furthermore, parameters can be
    guarded by being specified in ``config_only_args`` so that their values can only be
    changed globally via alf.config(). This prevents unintended side effects
    that may arise from having inconsistent parameter values caused by local
    changes (e.g., partial()).

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
    values specified through gin. Gin wrapper is quite convoluted and can make
    debugging more challenging. It can be disabled by setting environment
    variable ALF_USE_GIN to 0 if you are not using gin.

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
            ``whitelist`` or ``blacklist`` should be specified. An entry that is in
            ``blacklist`` cannot be in ``config_only_args``.
        config_only_args (list[str]): list of args that should be guarded. In other
            words, their values can only be set / changed globally via ``alf.config()``.
            This protects against local untracked changes as a result of 1) using
            ``partial()`` or 2) setting an argument when calling the function, which
            can cause unintended side effects. An entry that is in ``config_only_args``
            cannot be in ``blacklist``.
    Returns:
        decorated function if fn_or_name is Callable.
        a decorator if fn is not Callable.
    Raises:
        ValueError: Can be raised
            1) If a configurable with ``name`` (or the name of `fn_or_cls`) already exists
            2) If both a whitelist and blacklist are specified.
            3) If the same entry is found in both blacklist and config_only_args.
            4) If an arg listed in config_only_args is changed without using alf.config().
    """

    if callable(fn_or_name):
        name = None
    else:
        name = fn_or_name

    if whitelist and blacklist:
        raise ValueError("Only one of 'whitelist' and 'blacklist' can be set.")

    for entry in blacklist:
        if entry in config_only_args:
            raise ValueError(
                f"Entry '{entry}' found in both blacklist and config_only_args. "
                f"An entry can only be in one of these lists.")

    if not callable(fn_or_name):

        def _decorator(fn_or_cls):
            return _decorate(fn_or_cls, name, whitelist, blacklist,
                             config_only_args)

        return _decorator
    else:
        return _decorate(fn_or_name, name, whitelist, blacklist,
                         config_only_args)


def define_config(name, default_value):
    """Define a configurable value with given ``default_value``.

    Its value can be retrieved by ``get_config_value("_CONFIG._USER.{name}")``.

    Args:
        name (str): name of the configurable value
        default_value (Any): default value
    Returns:
        the configured value
    """
    node = _Config()
    node.set_default_value(default_value)
    _add_to_conf_tree(['_CONFIG'], '_USER', name, node)
    _DEFINED_CONFIGS.append('_CONFIG._USER.' + name)
    return get_config_value("_CONFIG._USER." + name)


def _get_conf_file_full_path(conf_file):
    if os.path.isabs(conf_file):
        if os.path.exists(conf_file):
            return os.path.realpath(conf_file)
    if len(_IMPORT_STACK) == 0:
        # called from load_config()
        dir = os.getcwd()
    else:
        # callded from import_config()
        dir = os.path.dirname(_IMPORT_STACK[-1])
    candidate = os.path.join(dir, conf_file)
    if os.path.exists(candidate):
        return os.path.realpath(candidate)
    conf_path = os.environ.get("ALF_CONFIG_PATH", None)
    conf_dirs = []
    if conf_path is not None:
        conf_dirs = conf_path.split(':')
    for dir in conf_dirs:
        candidate = os.path.join(dir, conf_file)
        if os.path.exists(candidate):
            return os.path.realpath(candidate)
    raise ValueError(f"Cannot find conf file {conf_file}")


def _add_conf_file(conf_file):
    if conf_file in _CONF_FILES:
        return
    with open(conf_file, "r") as f:
        _CONF_FILES[conf_file] = f.read()


def import_config(conf_file):
    """Import the config from another file.

    Different from ``load_config()``, ``import_config()`` should only be used in
    config files. And it can be used multiple times inside your config files.

    If ``conf_file`` is a relative path, ``load_config()`` will first try to find it
    in the directory of the config file calling this function. If it cannot be found
    there, directories in the environment varianble ``ALF_CONFIG_PATH`` will be
    searched in order.

    Examples:

    1. Suppose you have a config file ``~/code/my_conf.py``. You want to import
    another config file ``~/code/my_conf2.py``. You can use ``import_config("my_conf2.py")``
    to import ``my_config2.py``.

    2. Suppose you have a config file ``~/code/my_conf.py``. You want to import
    another config file ``~/code/base/my_conf2.py``. You can use ``import_config("base/my_conf2.py")``
    to import ``my_config2.py``.

    3. Suppose you have a config file ``~/code/my_conf.py``. You want to import
    another config file ``~/packages/my_conf2.py``. You need to set the environment
    variable as ``ALF_CONFIG_PATH=~/packages``. Then can use ``import_config("my_conf2.py")``
    to import ``my_config2.py``.

    Args:
        conf_file
    Returns:
        the config module object, which can be used in a similar way as python
        imported module.
    """
    if len(_IMPORT_STACK) == 0:
        raise ValueError("alf.import_config() can only be called inside a "
                         "config file.")
    conf_file = _get_conf_file_full_path(conf_file)
    return _import_config(conf_file)


class ConfigModule:
    pass


def _import_config(conf_file):
    if conf_file in _CONFIG_MODULES:
        return _CONFIG_MODULES[conf_file]
    _add_conf_file(conf_file)
    _IMPORT_STACK.append(conf_file)
    kv = runpy.run_path(conf_file)
    _IMPORT_STACK.pop()
    module = ConfigModule()
    for k, v in kv.items():
        setattr(module, k, v)
    _CONFIG_MODULES[conf_file] = module
    return module


def load_config(conf_file):
    """Load config from a file.

    Different from ``import_config()``, ``load_config()`` should only be used by
    your main code to load the config. And it should be only called once unless
     ``reset_configs()`` is called to reset the configuration to default state.

    If ``conf_file`` is a relative path, ``load_config()`` will first try to find it
    in the current working directory. If it cannot be found there, directories in
    the environment varianble ``ALF_CONFIG_PATH`` will be searched in order.

    Args:
        conf_file
    Returns:
        the config module object, which can be used in a similar way as python
        imported module.
    """
    global _ROOT_CONF_FILE
    if _ROOT_CONF_FILE is not None:
        raise ValueError(
            "One process can only call alf.load_config() once. "
            "If you want to call it multiple times, you need to call "
            "alf.reset_configs() between the calls.")
    conf_file = _get_conf_file_full_path(conf_file)
    _ROOT_CONF_FILE = conf_file
    return _import_config(conf_file)


def save_config(alf_config_file):
    """Save config files.

    This will save config set using ``pre_config()``, the file loaded using
    ``load_config()`` and the files imported using ``import_config()`` if they
    are in the config root directory or its sub-directory, where the config root
    directory is the directory of the conf file loaded by ``load_config()``.

    """
    if _ROOT_CONF_FILE is None:
        raise ValueError("alf.save_config() cannot be called before "
                         "alf.load_config()")
    config_dirname = "config_files"
    dir = os.path.join(os.path.dirname(alf_config_file), config_dirname)
    os.makedirs(dir, exist_ok=True)
    conf_file_name = os.path.basename(_ROOT_CONF_FILE)
    conf_root_dir = os.path.dirname(_ROOT_CONF_FILE)

    pre_configs = get_handled_pre_configs()
    config = ''
    config += "import alf\n"
    if pre_configs:
        config += "alf.pre_config({\n"
        for config_name, config_value in pre_configs:
            if isinstance(config_value, str):
                config += "    '%s': '%s',\n" % (config_name, config_value)
            else:
                config += "    '%s': %s,\n" % (config_name, config_value)
        config += "})\n\n"
    config += f"config = alf.import_config('{config_dirname}/{conf_file_name}')\n"
    f = open(alf_config_file, 'w')
    f.write(config)
    f.close()

    for conf_file, content in _CONF_FILES.items():
        if conf_file.startswith(conf_root_dir):
            conf_rel_path = conf_file[len(conf_root_dir) + 1:]
            conf_rel_dir = os.path.dirname(conf_rel_path)
            if conf_rel_dir:
                os.makedirs(os.path.join(dir, conf_rel_dir), exist_ok=True)
            with open(os.path.join(dir, conf_rel_path), "w") as f:
                f.write(content)
