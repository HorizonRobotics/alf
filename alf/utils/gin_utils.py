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

import gin.config
import six
import sys
import copy


def inoperative_config_str(max_line_length=80, continuation_indent=4):
    """Retrieve the "inoperative" configuration as a config string.

    Args:
        max_line_length (int): A (soft) constraint on the maximum length
            of a line in the formatted string.
        continuation_indent (int): The indentation for continued lines.
    Returns:
        A config string capturing all parameter values configured but not
            used by the current program (override by explicit call).
    """
    inoperative_config = {}
    config = gin.config._CONFIG
    operative_config = gin.config._OPERATIVE_CONFIG
    imported_module = gin.config._IMPORTED_MODULES
    for module, module_config in config.items():
        inoperative_module_config = {}
        if module not in operative_config:
            inoperative_module_config = module_config
        else:
            operative_module_config = operative_config[module]
            for key, value in module_config.items():
                if key not in operative_module_config or \
                        value != operative_module_config[key]:
                    inoperative_module_config[key] = value

        if inoperative_module_config:
            inoperative_config[module] = inoperative_module_config

    # hack below
    # `gin.operative_config_str` only depends on `_OPERATIVE_CONFIG` and `_IMPORTED_MODULES`
    gin.config._OPERATIVE_CONFIG = inoperative_config
    gin.config._IMPORTED_MODULES = {}
    inoperative_str = gin.operative_config_str(max_line_length,
                                               continuation_indent)
    gin.config._OPERATIVE_CONFIG = operative_config
    gin.config._IMPORTED_MODULES = imported_module
    return inoperative_str


def _config_gin_eval(func):
    name = func.__name__
    module = getattr(func, '__module__', None)
    selector = module + '.' + name if module else name
    fn = gin.config._ensure_wrappability(func)

    @six.wraps(fn)
    def gin_wrapper(**kwargs):
        scope_components = gin.config.current_scope()
        new_kwargs = {}
        for i in range(len(scope_components) + 1):
            partial_scope_str = '/'.join(scope_components[:i])
            new_kwargs.update(
                gin.config._CONFIG.get((partial_scope_str, selector), {}))
        scope_str = partial_scope_str
        operative_parameter_values = {}
        operative_parameter_values.update(new_kwargs)
        gin.config._OPERATIVE_CONFIG.setdefault(
            (scope_str, selector), {}).update(operative_parameter_values)
        new_kwargs = copy.deepcopy(new_kwargs)
        # hack here ,injection was actually called at frame_trace[-7]
        # (gin-config>=0.1.3,<=0.3.3)
        frame = sys._getframe(7)
        context = lambda c: (c, frame.f_globals, frame.f_locals)
        new_kwargs = {k: context(v) for k, v, in new_kwargs.items()}
        new_kwargs.update(kwargs)
        return fn(**new_kwargs)

    gin.config._REGISTRY[selector] = gin.config.Configurable(
        gin_wrapper,
        name=name,
        module=module,
        whitelist=None,
        blacklist=None,
        selector=selector)
    return gin_wrapper


@_config_gin_eval
def gin_eval(source):
    """Evaluate the given source in the context of globals and locals.

    A helper function that makes passing expression or unregistered functions
    and classes as parameter value possible by gin config

    Usage:
    arg_scope/gin_eval.source='...'
    func_scope/func.arg=@arg_scope/gin_eval()

    Examples
    --------
    Passing expression as parameter value
    >>> import numpy as np
    >>> @gin.configurable
        def calc_arc_len(radius, radian):
            return radius * radian
    >>> r = 2
    >>> calc_arc_len()

    # Inside "config.gin"
    radius/gin_eval.source="r"
    radian/gin_eval.source="0.3*np.pi"
    test/calc_arc.radius=@radius/gin_eval()
    test/calc_arc.radian=@radian/gin_eval()

    --------
    Passing other unregistered functions or classes as parameter value
    >>> @gin.configurable
        def activate(value, activation_fn=torch.relu)
            pass

    # Inside "config.gin"
    torch_exp/gin_eval.source='torch.exp'
    activate.activation_fn=@torch_exp/gin_eval()
    --------

    Args:
        source (tuple): source and its context to be evaluated
    """
    source_str, f_globals, f_locals = source
    return eval(source_str, f_globals, f_locals)
