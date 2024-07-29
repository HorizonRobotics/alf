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
from .rl_algorithm import RLAlgorithm


class AlgorithmContainer(Algorithm):
    """Algorithm that contains several sub-algorithms.

    It provides sensible implementation of several interface functions of
    Algorithm.
    """

    def __init__(self, algs, train_state_spec, rollout_state_spec,
                 predict_state_spec, is_on_policy, debug_summaries, name):
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
            is_on_policy (None|bool): whether the algorithm is on-policy or not.
                If None, the on-policiness will be decided based on the on-policiness
                of each sub-algorithm.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): name of this algorithm.
        """
        assert isinstance(algs, dict)
        if is_on_policy is not None:
            for aname, alg in algs.items():
                if alg.on_policy is not None:
                    assert alg.on_policy == is_on_policy, (
                        "is_on_policy=%s "
                        "is different from algs[%s].on_policy=%s" %
                        (is_on_policy, aname, alg.on_policy))
        else:
            on_policy_algs = [
                alg for alg in algs.values() if alg.on_policy == True
            ]
            off_policy_algs = [
                alg for alg in algs.values() if alg.on_policy == False
            ]
            if on_policy_algs and off_policy_algs:
                raise ValueError(
                    "%s is on-policy, but %s is off-policy." %
                    (on_policy_algs[0].name, off_policy_algs[0].name))
            if on_policy_algs or off_policy_algs:
                is_on_policy = bool(on_policy_algs)
        if is_on_policy is not None:
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
        """Call set_on_policy of each sub-algorithm."""
        super().set_on_policy(is_on_policy)
        for alg in self._algs.values():
            alg.set_on_policy(is_on_policy)

    def set_path(self, path):
        """Set the path for each sub-algorithm."""
        super().set_path(path)
        prefix = path
        if path:
            prefix = prefix + '.'
        for name, alg in self._algs.items():
            alg.set_path(path + name)

    def calc_loss(self, info):
        """Call calc_loss of each sub-algorithm and accumulate the loss."""
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
        """Call the preprocess_experience of each sub-algorithm."""
        new_infos = {}
        for name, alg in self._algs.items():
            root_inputs, info = alg.preprocess_experience(
                root_inputs, rollout_info[name], batch_info)
            new_infos[name] = info

        return root_inputs, new_infos

    def after_update(self, root_inputs, info):
        """Call after_update of each sub-algorithm."""
        for name, alg in self._algs.items():
            alg.after_update(root_inputs, info[name])

    def after_train_iter(self, root_inputs, rollout_info):
        """Call after_train_iter of each sub-algorithm."""
        for name, alg in self._algs.items():
            alg.after_train_iter(root_inputs, rollout_info[name])


def SequentialAlg(*modules,
                  output='',
                  is_on_policy=None,
                  name="SequentialAlg",
                  **named_modules):
    """Compose Algorithms Networks sequentially as a new Algorithm.

    All the modules provided through ``modules`` and ``named_modules`` are calculated
    sequentially in the same order as they appear in the call to ``SequentialAlg``.
    By default, each module takes the output of the previous module as its input
    (or the input to the SequentialAlg if it is the first module), and the output of
    the last module is the output of the ``SequentialAlg``. Note that the output
    of a module means differently depending on the type of the module:

    * Algorithm: ``AlgStep.output`` field from ``predict_step``, ``rollout_step``
        or ``train_step``
    * Network: the first element of the tuple returned from ``forward()``
    * torch.nn.Module or Callable: the return value of the Callable.

    In addition to using the output of the previous module as input,
    ``SequentialAlg`` also allow using other output, state or info from previous
    module as the input to a module. To do this,
    one can pass a tuple of (nested_str, module) instead of module as an argument
    to ``SequentialAlg``. With this, the inputs to the module will be obtained using
    ``get_nested_field(named_results, nested_str)``, where ``named_results``
    is a dictionary containing the inputs to ``SequentialAlg`` and all the results calulcated
    by previous modules. More specifically, ``named_results['input']`` is the
    inputs to this algorithm. ``named_results['a']`` is the output of the module
    named 'a'. ``named_results['info']['a']`` is the info output of the algorithm
    named 'a'.  And ``named_results['state']['a']`` is state output of the
    algorithm/network named 'a'.

    The result of ``(predict/rollout/train)_step()`` of the ``SequentialAlg`` is
    an ``AlgStep`` with the following fields:

    - output: the output of the last module or the output specified by ``output``.
    - state: a list of states from all the modules.
    - info: a dictionary of info from all the sub-algorithms. The keys
        are the names of the sub-algorithms (i.e. the name obtained using
        ``algorithm.name``)

    Example 1:

    The following constructs an algorithm which predicts the future of its input:

    .. code-block:: python

        predictor = EncodingNetwork(...)

        alg = SequentialAlg(
            predicted=predictor,
            delayed=networks.Delay(),
            error=(('delayed', 'input'), lambda xy: (xy[0] - xy[1]) ** 2),
            loss=Loss(),
            output='predicted',
        )

    It is equivalent to the following:

    .. code-block:: python

        class PredictAlgorithm(Algorithm):
            def __init__(self, predictor):
                super().__init__(train_state_spec=(
                    predictor.state_spec,
                    predictor.input_tensor_spec))
                self._predictor = predictor
                self._loss = Loss()

            def rollout_step(self, inputs, state):
                return self._step(inputs, state)

            def train_step(self, inputs, state, rollout_info):
                return self._step(inputs, state)

            def _step(self, inputs, state):
                predictor_state, delayed = state
                predicted, predictor_state = self._predictor(inputs, predictor_state)
                error = (delayed - inputs) ** 2
                loss_step = self._loss.rollout_step(error)
                return AlgStep(
                    output=predicted,
                    state=(predictor_state, predicted),
                    info=loss_step.info)

            def calc_loss(info):
                return self._loss.calc_loss(info)

        alg = PredictAlgorithm(predictor)

    Example 2:

    The following example constructs an actor-critic algorithm:

    .. code-block:: python

        value_net = ValueNetwork(...)
        actor_net = ActorDistributionNetwork(...)

        alg = SequentialAlg(
            is_on_policy=True,
            value=('input.observation', value_net),
            action_dist=('input.observation', actor_net),
            action=dist_utils.sample_action_distribution,
            loss=(ActorCriticInfo(
                reward='input.reward',
                step_type='input.step_type',
                discount='input.discount',
                action_distribution='action_dist',
                action='action',
                value='value'), ActorCriticLoss()),
            output='action')

    It is equivalent to the following:

    .. code-block:: python

        class ACAlgorithm(Algorithm):
            def __init__(self, value_net, actor_net):
                super().__init__(
                    train_state_spec=(value_net.state_spec, actor_net.state_spec),
                    is_on_policy=True)
                self._value_net = value_net
                self._actor_net = actor_net
                self._loss = ActorCriticLoss()

            def rollout_step(self, inputs, state):
                value, value_state = self._value_net(inputs.observation, state[0])
                action_dist, actor_state = self._actor_net(inputs.observation, state[1])
                action = dist_utils.sample_action_distribution(action_dist)
                loss_step = self._loss.rollout_step(ActorCriticInfo(
                    reward=inputs.reward,
                    step_type=inputs.step_type,
                    discount=inputs.discount,
                    action_distribution=action_dist,
                    action=action,
                    value=value))
                )
                return AlgStep(
                    output=action,
                    state=(value_state, actor_state),
                    info=loss_step.info)

            def calc_loss(self, info):
                self._loss.calc_loss(info)

        alg = ACAlgorithm(value_net, actor_net)

    Args:
        modules (Callable | Algorithm | (nested str, Callable) | (nested str, Algorithm)):
            The ``Callable`` can be a ``torch.nn.Module``, ``alf.nn.Network``
            or plain ``Callable``. Optionally, their inputs can be specified
            by the first element of the tuple. If input is not provided, it is
            assumed to be the result of the previous module (or input to this
            ``Sequential`` for the first module). If input is provided, it
            should be a nested str. It will be used to retrieve results from
            the dictionary of the current ``named_results``. For modules
            specified by ``modules``, because no ``named_modules`` has been
            invoked, ``named_outputs`` is ``{'input': input}``.
        named_modules (Callable | Algorithm  | (nested str, Callable) | (nested str, Algorithm)):
            The ``Callable`` can be a ``torch.nn.Module``, ``alf.nn.Network``
            or plain ``Callable``. Optionally, their inputs can be specified
            by the first element of the tuple. If input is not provided, it is
            assumed to be the result of the previous module (or input to this
            ``Sequential`` for the first module). If input is provided, it
            should be a nested str. It will be used to retrieve results from
            the dictionary of the current ``named_results``. ``named_results``
            is updated once the result of a named module is calculated.
        output (nested str): if not provided, the result from the last module
            will be used as output. Otherwise, it will be used to retrieve
            results from ``named_results`` after the results of all modules
            have been calculated.
        is_on_policy (bool): whether this supports on-policy or off-policy training.
            If is None, it should support both on-policy and off-policy training.
        name (str): name of this algorithm

    """

    return _SequentialAlg(
        elements=modules,
        element_dict=named_modules,
        output=output,
        is_on_policy=is_on_policy,
        name=name)


class _SequentialAlg(AlgorithmContainer):
    def __init__(self,
                 elements=(),
                 element_dict=None,
                 output='',
                 is_on_policy=None,
                 name='SequentialAlg'):
        train_state_spec = []
        rollout_state_spec = []
        predict_state_spec = []
        modules = []
        inputs = []
        outputs = []
        named_elements = list(zip([''] * len(elements), elements))
        if element_dict:
            named_elements.extend(element_dict.items())
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
            is_on_policy=is_on_policy,
            debug_summaries=False,
            name=name)

        self._networks = modules
        self._nets = nn.ModuleList(
            filter(lambda m: isinstance(m, nn.Module), modules))
        self._output = output
        self._inputs = inputs
        self._outputs = outputs

    def _step(self, step_func, inputs, state):
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
                alg_step = getattr(net, step_func)(x, state[i])
                x = alg_step.output
                new_state[i] = alg_step.state
                info[net.name] = alg_step.info
                info_dict[output] = alg_step.info
                state_dict[output] = new_state[i]
            elif isinstance(net, Network):
                x, new_state[i] = net(x, state[i])
                state_dict[output] = new_state[i]
            else:
                x = net(x)
            var_dict[output] = x
        if self._output:
            x = get_nested_field(var_dict, self._output)
        return AlgStep(output=x, state=new_state, info=info)

    def predict_step(self, inputs, state):
        return self._step('predict_step', inputs, state)

    def rollout_step(self, inputs, state):
        return self._step('rollout_step', inputs, state)

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
                state_dict[output] = new_state[i]
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
    """Echo Algorithm.

    Echo algorithm uses part of the output of ``alg`` of current step as part of
    the input of ``alg`` for the next step. It assumes that the input of ``alg``
    is a dict with two keys: 'input' and 'echo', and the output of ``alg`` is
    a dict with two keys: 'output' and 'echo'. The 'echo' output of current step
    will be the 'echo' input of the next step. 'input' of ``alg``'s input is from
    the input of ``EchoAlg`` and 'output' of ``alg``'s output is the output of
    ``EchoAlg``.
    """

    def __init__(self, alg, echo_spec, name='EchoAlg'):
        """
        Args:
            alg (Algorithm): the module for performing the actual computation
            echo_spec (nested TensorSpec): describe the data format of echo.
            name (str):
        """
        assert isinstance(alg, Algorithm), (
            "block must be an instance of "
            "alf.algorithms.algorithm.Algorithm. Got %s" % type(alg))

        super().__init__(
            train_state_spec=(alg.train_state_spec, echo_spec),
            rollout_state_spec=(alg.rollout_state_spec, echo_spec),
            predict_state_spec=(alg.predict_state_spec, echo_spec),
            is_on_policy=alg.on_policy,
            name=name)

        self._alg = alg

    def set_on_policy(self, is_on_policy):
        super().set_on_policy(is_on_policy)
        self._alg.set_on_policy(is_on_policy)

    def set_path(self, path):
        super().set_path(path)
        self._alg.set_path(path)

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

    def after_update(self, root_inputs, info):
        self._alg.after_update(root_inputs, info)

    def after_train_iter(self, root_inputs, rollout_info):
        self._alg.after_train_iter(root_inputs, rollout_info)


@alf.configurable
class RLAlgWrapper(RLAlgorithm):
    """Wrap an ``Algorithm`` instance as an ``RLAlgorithm`` instance
       so that it can be used for RLTrainer.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 algorithm,
                 env=None,
                 reward_spec=alf.TensorSpec(()),
                 config: TrainerConfig = None,
                 optimizer=None,
                 debug_summaries=False,
                 name="RLAlgWrapper"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            algorithm (Algorithm): algorithm to be wrapped. It should take
                ``TimeStep`` as input and its output will be used as action.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            env (Environment): The environment to interact with. ``env`` is a
                batched environment, which means that it runs multiple
                simulations simultaneously. Running multiple environments in
                parallel is crucial to on-policy algorithms as it increases the
                diversity of data and decreases temporal correlation. ``env`` only
                needs to be provided to the root ``Algorithm``.
            config (TrainerConfig): config for training. ``config`` only needs to
                be provided to the algorithm which performs a training iteration
                by itself.
            optimizer (torch.optim.Optimizer): The default optimizer for training.
            debug_summaries (bool): If True, debug summaries will be created.
            name (str): Name of this algorithm.

        """
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            train_state_spec=algorithm.train_state_spec,
            reward_spec=reward_spec,
            predict_state_spec=algorithm.predict_state_spec,
            rollout_state_spec=algorithm.rollout_state_spec,
            is_on_policy=algorithm.on_policy,
            env=env,
            config=config,
            optimizer=optimizer,
            debug_summaries=debug_summaries,
            name=name)

        self._algorithm = algorithm

    def set_on_policy(self, is_on_policy):
        super().set_on_policy(is_on_policy)
        self._algorithm.set_on_policy(is_on_policy)

    def set_path(self, path):
        super().set_path(path)
        self._algorithm.set_path(path)

    def predict_step(self, inputs: TimeStep, state):
        return self._algorithm.predict_step(inputs, state)

    def rollout_step(self, inputs: TimeStep, state):
        return self._algorithm.rollout_step(inputs, state)

    def train_step(self, inputs: TimeStep, state, rollout_info):
        return self._algorithm.train_step(inputs, state, rollout_info)

    def calc_loss(self, info):
        return self._algorithm.calc_loss(info)

    def preprocess_experience(self, root_inputs, rollout_info, batch_info):
        return self._algorithm.preprocess_experience(root_inputs, rollout_info,
                                                     batch_info)

    def after_update(self, root_inputs, info):
        self._algorithm.after_update(root_inputs, info)

    def after_train_iter(self, root_inputs, rollout_info):
        self._algorithm.after_train_iter(root_inputs, rollout_info)
