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

import torch.nn as nn
from typing import Callable

import alf
from alf.data_structures import AlgStep, LossInfo, TimeStep
from alf.nest import flatten
from alf.nest.utils import get_nested_field
from alf.networks import Network
from alf.utils.math_ops import add_ignore_empty

from .algorithm import Algorithm
from .config import TrainerConfig
from .on_policy_algorithm import OnPolicyAlgorithm
from .rl_algorithm import RLAlgorithm


class AlgorithmContainer(Algorithm):
    def __init__(self, algs, train_state_spec, rollout_state_spec,
                 predict_state_spec, debug_summaries, name):
        """
        Args:
            algs (dict[Algorithm]): a dictionary of algorithms.
            train_state_spec (nested TensorSpec): for the network state of
                ``train_step()``.
            rollout_state_spec (nested TensorSpec): for the network state of
                ``predict_step()``. If None, it's assumed to be the same as
                ``train_state_spec``.
            predict_state_spec (nested TensorSpec): for the network state of
                ``predict_step()``. If None, it's assume to be same as
                ``rollout_state_spec``.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): name of this algorithm.
        """
        is_on_policy = None
        on_policy_algs = [
            alg for alg in algs.values() if alg.is_on_policy() == True
        ]
        off_policy_algs = [
            alg for alg in algs.values() if alg.is_on_policy() == False
        ]
        if on_policy_algs and off_policy_algs:
            raise ValueError("%s is on-policy, but %s is off-policy." %
                             (on_policy_algs[0].name, off_policy_algs[0].name))
        if on_policy_algs or off_policy_algs:
            is_on_policy = bool(on_policy_algs)
            for alg in algs.values():
                alg.set_on_policy(is_on_policy)

        super().__init__(
            train_state_spec=train_state_spec,
            rollout_state_spec=rollout_state_spec,
            predict_state_spec=predict_state_spec,
            is_on_policy=is_on_policy,
            debug_summaries=debug_summaries,
            name=name)

        self._algs = algs

    def set_on_policy(self, is_on_policy):
        super().set_on_policy(is_on_policy)
        for alg in self._algs.values():
            alg.set_on_policy(is_on_policy)

    def set_path(self, path):
        super().set_path(path)
        prefix = path
        if path:
            prefix = prefix + '.'
        for name, alg in self._algs.items():
            alg.set_path(path + name)

    def calc_loss(self, info):
        extra = {}
        scalar_loss = ()
        loss = ()
        priority = ()
        for name, alg in self._algs.items():
            loss_info = alg.calc_loss(info[name])
            extra[name] = loss_info.extra
            loss = add_ignore_empty(loss_info.loss, loss)
            scalar_loss = add_ignore_empty(loss_info.scalar_loss, scalar_loss)
            priority = add_ignore_empty(loss_info.priority, priority)

        return LossInfo(
            loss=loss, scalar_loss=scalar_loss, extra=extra, priority=priority)

    def preprocess_experience(self, root_inputs, rollout_info, batch_info):
        new_infos = {}
        for name, alg in self._algs.items():
            root_inputs, info = alg.preprocess_experience(
                root_inputs, rollout_info[name], batch_info)
            new_infos[name] = info

        return root_inputs, new_infos


def SequentialAlg(*modules,
                  output='',
                  name="SequentialAlg",
                  debug_summaries=False,
                  **named_modules):

    return _SequentialAlg(
        elements=modules,
        element_dict=named_modules,
        output=output,
        debug_summaries=debug_summaries,
        name=name)


def _build_nested_fields(paths):
    """
    Examples:

        nest = _build_nested_fields(['a.b', 'a', 'c.d'])
        assert nest == {'a': 'a', 'c': {'d': 'c.d'}}}
    """
    nested_fields = {}
    for path in paths:
        fields = path.split('.')
        sub_nest = nested_fields
        for i, field in enumerate(fields):
            if isinstance(sub_nest, str):
                break
            elif i == len(fields) - 1:
                sub_nest[field] = path
            else:
                if field in sub_nest:
                    sub_nest = sub_nest[field]
                else:
                    sub_nest[field] = {}
                    sub_nest = sub_nest[field]
    return nested_fields


class _SequentialAlg(AlgorithmContainer):
    def __init__(self,
                 elements=(),
                 element_dict={},
                 output='',
                 debug_summaries=False,
                 name='SequentialAlg'):
        train_state_spec = []
        rollout_state_spec = []
        predict_state_spec = []
        modules = []
        inputs = []
        outputs = []
        named_elements = list(zip([''] * len(elements), elements)) + list(
            element_dict.items())
        is_nested_str = lambda s: all(
            map(lambda x: type(x) == str, flatten(s)))

        algs = {}

        for i, (out, element) in enumerate(named_elements):
            input = ''
            if isinstance(element, tuple) and len(element) == 2:
                input, module = element
            else:
                module = element
            if not (isinstance(module, (Callable, Algorithm))
                    and is_nested_str(input)):
                raise ValueError(
                    "Argument %s is not in the form of Callable|Algorithm "
                    "or (nested str, Callable|Algorithm): %s" % (out or str(i),
                                                                 element))
            if isinstance(module, Algorithm):
                train_state_spec.append(module.train_state_spec)
                rollout_state_spec.append(module.rollout_state_spec)
                predict_state_spec.append(module.predict_state_spec)
                assert module.name not in algs, (
                    "Duplicated algorithm name %s" % module.name)
                algs[module.name] = module
            elif isinstance(module, Network):
                train_state_spec.append(module.state_spec)
                rollout_state_spec.append(module.state_spec)
                predict_state_spec.append(module.state_spec)
            else:
                train_state_spec.append(())
                rollout_state_spec.append(())
                predict_state_spec.append(())
            if not out:
                out = str(i)
            inputs.append(input)
            outputs.append(out)
            modules.append(module)

        assert is_nested_str(output), (
            "output should be a nested str: %s" % output)

        super().__init__(
            algs,
            train_state_spec=train_state_spec,
            rollout_state_spec=rollout_state_spec,
            predict_state_spec=predict_state_spec,
            debug_summaries=debug_summaries,
            name=name)

        self._networks = modules
        self._nets = nn.ModuleList(
            filter(lambda m: isinstance(m, nn.Module), modules))
        self._output = output
        self._inputs = inputs
        self._outputs = outputs

    def rollout_step(self, inputs, state):
        info_dict = {}
        state_dict = {}
        var_dict = {'input': inputs, 'info': info_dict, 'state': state_dict}
        info = {}
        new_state = [()] * len(self._networks)
        x = inputs
        for i, net in enumerate(self._networks):
            output = self._outputs[i]
            if self._inputs[i]:
                x = get_nested_field(var_dict, self._inputs[i])
            if isinstance(net, Algorithm):
                alg_step = net.rollout_step(x, state[i])
                x = alg_step.output
                new_state[i] = alg_step.state
                info[net.name] = alg_step.info
                info_dict[output] = alg_step.info
            elif isinstance(net, Network):
                x, new_state[i] = net(x, state[i])
                state_dict[output] = new_state[i]
            else:
                x = net(x)
            var_dict[output] = x
        if self._output:
            x = get_nested_field(var_dict, self._output)
        return AlgStep(output=x, state=new_state, info=info)

    def train_step(self, inputs, state, rollout_info):
        info_dict = {}
        state_dict = {}
        var_dict = {'input': inputs, 'info': info_dict, 'state': state_dict}
        info = {}
        new_state = [()] * len(self._networks)
        x = inputs
        for i, net in enumerate(self._networks):
            output = self._outputs[i]
            if self._inputs[i]:
                x = get_nested_field(var_dict, self._inputs[i])
            if isinstance(net, Algorithm):
                alg_step = net.train_step(x, state[i], rollout_info[net.name])
                x = alg_step.output
                new_state[i] = alg_step.state
                info[net.name] = alg_step.info
                info_dict[output] = alg_step.info
            elif isinstance(net, Network):
                x, new_state[i] = net(x, state[i])
                state_dict[output] = new_state[i]
            else:
                x = net(x)
            var_dict[output] = x
        if self._output:
            x = get_nested_field(var_dict, self._output)
        return AlgStep(output=x, state=new_state, info=info)


class EchoAlg(Algorithm):
    def __init__(self, alg, echo_spec, debug_summaries=False, name='EchoAlg'):
        """
        Args:
            alg (Algorithm): the module for performing the actual computation
            echo_spec ():
        """
        assert isinstance(alg, Algorithm), (
            "block must be an instance of "
            "alf.algorithms.algorithm.Algorithm. Got %s" % type(alg))

        super().__init__(
            train_state_spec=(alg.train_state_spec, echo_spec),
            rollout_state_spec=(alg.rollout_state_spec, echo_spec),
            predict_state_spec=(alg.predict_state_spec, echo_spec),
            is_on_policy=alg.is_on_policy(),
            debug_summaries=debug_summaries,
            name=name)

        self._alg = alg

    def set_on_policy(self, is_on_policy):
        super().set_on_policy(is_on_policy)
        self._alg.set_on_policy(is_on_policy)

    def predict_step(self, inputs, state):
        block_state, echo_state = state
        block_input = dict(input=inputs, echo=echo_state)
        alg_step = self._alg.predict_step(block_input, block_state)
        real_output = alg_step.output['output']
        echo_output = alg_step.output['echo']
        return AlgStep(
            output=real_output,
            state=(alg_step.state, echo_output),
            info=alg_step.info)

    def rollout_step(self, inputs, state):
        block_state, echo_state = state
        block_input = dict(input=inputs, echo=echo_state)
        alg_step = self._alg.rollout_step(block_input, block_state)
        real_output = alg_step.output['output']
        echo_output = alg_step.output['echo']
        return AlgStep(
            output=real_output,
            state=(alg_step.state, echo_output),
            info=alg_step.info)

    def train_step(self, inputs, state, rollout_info):
        block_state, echo_state = state
        block_input = dict(input=inputs, echo=echo_state)
        alg_step = self._alg.train_step(block_input, block_state, rollout_info)
        real_output = alg_step.output['output']
        echo_output = alg_step.output['echo']
        return AlgStep(
            output=real_output,
            state=(alg_step.state, echo_output),
            info=alg_step.info)

    def calc_loss(self, info):
        return self._alg.calc_loss(info)

    def preprocess_experience(self, root_inputs, rollout_info, batch_info):
        return self._alg.preprocess_experience(root_inputs, rollout_info,
                                               batch_info)


@alf.configurable
class RLAlgWrapper(OnPolicyAlgorithm):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 algorithm,
                 env=None,
                 reward_spec=alf.TensorSpec(()),
                 reward_weights=None,
                 config: TrainerConfig = None,
                 optimizer=None,
                 debug_summaries=False,
                 name="RLAlgWrapper"):
        self._is_on_policy = algorithm.is_on_policy()
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            train_state_spec=algorithm.train_state_spec,
            reward_spec=reward_spec,
            predict_state_spec=algorithm.predict_state_spec,
            rollout_state_spec=algorithm.rollout_state_spec,
            reward_weights=reward_weights,
            env=env,
            config=config,
            optimizer=optimizer,
            debug_summaries=debug_summaries,
            name=name)

        self._algorithm = algorithm

    def is_on_policy(self):
        return self._is_on_policy

    def set_on_policy(self, is_on_policy):
        super().set_on_policy(is_on_policy)
        self._algorithm.set_on_policy()

    def rollout_step(self, inputs: TimeStep, state):
        return self._algorithm.rollout_step(inputs, state)

    def train_step(self, inputs: TimeStep, state, rollout_info):
        return self._algorithm.train_step(inputs, state, rollout_info)

    def calc_loss(self, info):
        return self._algorithm.calc_loss(info)

    def preprocess_experience(self, root_inputs, rollout_info, batch_info):
        return self._algorithm.preprocess_experience(root_inputs, rollout_info,
                                                     batch_info)


class LossAlg(Algorithm):
    def __init__(self, loss_weight=1.0, name="LossAlg"):
        super().__init__(name=name)
        self._loss_weight = loss_weight

    def rollout_step(self, inputs, state):
        if self.is_on_policy():
            return AlgStep(info=inputs)
        else:
            return AlgStep()

    def train_step(self, inputs, state, rollout_info):
        return AlgStep(info=inputs)

    def calc_loss(self, info):
        return LossInfo(loss=self._loss_weight * info, extra=info)
