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
from alf.data_structures import AlgStep, Experience, LossInfo, TimeStep
from alf.nest import (flatten, flatten_up_to, get_field, map_structure,
                      map_structure_up_to, pack_sequence_as)
from alf.nest.utils import get_nested_field
from alf.networks import Network
from alf.utils.math_ops import add_ignore_empty
from alf.utils.spec_utils import is_same_spec

from .algorithm import Algorithm
from .config import TrainerConfig
from .on_policy_algorithm import OnPolicyAlgorithm
from .rl_algorithm import RLAlgorithm


def SequentialAlg(*modules,
                  output='',
                  name="SequentialAlg",
                  debug_summaries=False,
                  **named_modules):

    return _Sequential(
        modules,
        named_modules,
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


class AlgRLWrapper(Algorithm):
    def __init__(self, rl):
        super().__init__(
            train_state_spec=rl.train_state_spec,
            rollout_state_spec=rl.rollout_state_spec,
            predict_state_spec=rl.predict_state_spec,
            name=rl.name)
        self._rl = rl

    def rollout_step(self, inputs, state):
        return self._rl.rollout_step(inputs, state)

    def calc_loss(self, inputs, train_info):
        exp = Experience(
            step_type=inputs.step_type,
            reward=inputs.reward,
            discount=inputs.discount,
            observation=inputs.observation,
            prev_action=inputs.prev_action,
            action=train_info.action)
        return self._rl.calc_loss(exp, train_info)


class _Sequential(Algorithm):
    def __init__(self, elements, element_dict, output, debug_summaries, name):
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

        # name of all inputs of all algorithms
        alg_inputs = []
        alg_names = set()

        for i, (out, element) in enumerate(named_elements):
            input = ''
            if isinstance(element, tuple) and len(element) == 2:
                input, module = element
            else:
                module = element
            if isinstance(module, RLAlgorithm):
                assert module.is_on_policy(), (
                    "Only on-policy RLAlgorithm is supported: %s", module)
                module = AlgRLWrapper(module)
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
                assert module.name not in alg_names, (
                    "Duplicated algorithm name %s" % module.name)
                alg_names.add(module.name)
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
            if isinstance(module, Algorithm):
                if input == '':
                    if i == 0:
                        alg_inputs.append('input')
                    else:
                        alg_inputs.append(outputs[i - 1])
                else:
                    alg_inputs.append(input)

        assert is_nested_str(output), (
            "output should be a nested str: %s" % output)

        super().__init__(
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
        self._alg_inputs = _build_nested_fields(alf.nest.flatten(alg_inputs))
        del self._alg_inputs['input']

    def rollout_step(self, inputs, state):
        info_dict = {}
        state_dict = {}
        var_dict = {'input': inputs, 'info': info_dict, 'state': state_dict}
        infos = [()] * len(self._networks)
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
                infos[i] = alg_step.info
                info_dict[output] = alg_step.info
            elif isinstance(net, Network):
                x, new_state[i] = net(x, state[i])
                state_dict[output] = new_state[i]
            else:
                x = net(x)
            var_dict[output] = x
        alg_inputs = get_nested_field(var_dict, self._alg_inputs)
        if self._output:
            x = get_nested_field(var_dict, self._output)
        return AlgStep(output=x, state=new_state, info=(infos, alg_inputs))

    def calc_loss(self, input, train_info):
        infos, alg_inputs = train_info
        alg_inputs['input'] = input
        extra = {}
        scalar_loss = ()
        loss = ()
        priority = ()
        x = input
        for i, net in enumerate(self._networks):
            if not isinstance(net, Algorithm):
                continue
            if self._inputs[i]:
                x = get_nested_field(alg_inputs, self._inputs[i])
            else:
                x = alg_inputs[self._outputs[i - 1]]
            loss_info = net.calc_loss(x, infos[i])
            extra[net.name] = loss_info.extra
            loss = add_ignore_empty(loss_info.loss, loss)
            scalar_loss = add_ignore_empty(loss_info.scalar_loss, scalar_loss)
            priority = add_ignore_empty(loss_info.priority, priority)

        return LossInfo(
            loss=loss, scalar_loss=scalar_loss, extra=extra, priority=priority)


class EchoAlg(Algorithm):
    def __init__(self, echo_spec, alg, debug_summaries=False, name='EchoAlg'):
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
            debug_summaries=debug_summaries,
            name=name)

        self._alg = alg

    def rollout_step(self, inputs, state):
        block_state, echo_state = state
        block_input = dict(input=inputs, echo=echo_state)
        alg_step = self._alg.rollout_step(block_input, block_state)
        real_output = alg_step.output['output']
        echo_output = alg_step.output['echo']
        return AlgStep(
            output=real_output,
            state=(block_state, echo_output),
            info=(alg_step.info, echo_state))

    def calc_loss(self, inputs, train_info):
        block_info, echo_state = train_info
        block_input = dict(input=inputs, echo=echo_state)
        return self._alg.calc_loss(block_input, block_info)


@alf.configurable
class RLAlgWrapper(OnPolicyAlgorithm):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 algorithm,
                 env,
                 reward_spec=alf.TensorSpec(()),
                 reward_weights=None,
                 config: TrainerConfig = None,
                 optimizer=None,
                 debug_summaries=False,
                 name="RLAlgWrapper"):
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

    def rollout_step(self, time_step: TimeStep, state):
        return self._algorithm.rollout_step(time_step, state)

    def calc_loss(self, experience, train_info):
        return self._algorithm.calc_loss(experience, train_info)


class LossAlg(Algorithm):
    def __init__(self, name="LossAlg"):
        super().__init__(name=name)

    def rollout_step(self, inputs, state):
        return AlgStep(output=inputs, state=state, info=LossInfo(loss=inputs))
