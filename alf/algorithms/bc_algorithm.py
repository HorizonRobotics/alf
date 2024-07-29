# Copyright (c) 2022 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""Behavior Cloning (BC) Algorithm."""

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.data_structures import TimeStep, LossInfo, namedtuple
from alf.data_structures import AlgStep
from alf.networks import ActorNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import dist_utils

BcState = namedtuple("BcState", ["actor"], default_value=())

BcInfo = namedtuple("BcInfo", ["actor"], default_value=())

BcLossInfo = namedtuple("LossInfo", ["actor"], default_value=())


@alf.configurable
class BcAlgorithm(OffPolicyAlgorithm):
    r"""Behavior cloning algorithm.
    Behavior cloning is an offline approach to learn a policy
    :math:`\pi_{\theta}(a|s)`, which is a function that maps an input
    observation :math:`s` to an action :math:`a`. The paramerates (:math:`\theta`)
    of this policy is learned by using the expert action as supervision for
    training, e.g., by maximizing the probability of the expert actions on the
    training data :math:`D`:
    :math:`\max_{\theta} E_{(s,a)~D}\log \pi_{\theta}(a|s)`

    Reference:
    ::
        Pomerleau ALVINN: An Autonomous Land Vehicle in a Neural Network, NeurIPS 1988.
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorNetwork,
                 actor_optimizer=None,
                 env=None,
                 config: TrainerConfig = None,
                 checkpoint=None,
                 debug_summaries=False,
                 epsilon_greedy=None,
                 name="BcAlgorithm"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions; can
                be a mixture of discrete and continuous actions. The number of
                continuous actions can be arbitrary while only one discrete
                action is allowed currently. If it's a mixture, then it must be
                a tuple/list ``(discrete_action_spec, continuous_action_spec)``.
            reward_spec (Callable): a rank-1 or rank-0 tensor spec representing
                the reward(s). For interface compatibility purpose. Not actually
                used in BcAlgorithm.
            actor_network_cls (Callable): is used to construct the actor network.
                The constructed actor network is a deterministic network and
                will be used to generate continuous actions.
            actor_optimizer (torch.optim.optimizer): The optimizer for actor.
            env (Environment): The environment to interact with. ``env`` is a
                batched environment, which means that it runs multiple simulations
                simultateously. ``env` only needs to be provided to the root
                algorithm.
            config (TrainerConfig): config for training. It only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            checkpoint (None|str): a string in the format of "prefix@path",
                where the "prefix" is the multi-step path to the contents in the
                checkpoint to be loaded. "path" is the full path to the checkpoint
                file saved by ALF. Refer to ``Algorithm`` for more details.
            debug_summaries (bool): True if debug summaries should be created.
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation. If None, its value is taken
                from ``config.epsilon_greedy`` and then
                ``alf.get_config_value(TrainerConfig.epsilon_greedy)``.
            name (str): The name of this algorithm.
        """

        if epsilon_greedy is None:
            epsilon_greedy = alf.utils.common.get_epsilon_greedy(config)
        self._epsilon_greedy = epsilon_greedy

        actor_network = actor_network_cls(
            input_tensor_spec=observation_spec, action_spec=action_spec)

        action_state_spec = actor_network.state_spec
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=BcState(actor=action_state_spec),
            predict_state_spec=BcState(actor=action_state_spec),
            reward_weights=None,
            env=env,
            config=config,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name)

        self._actor_network = actor_network

        if actor_optimizer is not None and actor_network is not None:
            self.add_optimizer(actor_optimizer, [actor_network])
        self._actor_optimizer = actor_optimizer

    def _predict_action(self, observation, state):
        action_dist, actor_network_state = self._actor_network(
            observation, state=state)

        return action_dist, actor_network_state

    def predict_step(self, inputs: TimeStep, state: BcState):
        action_dist, new_state = self._predict_action(
            inputs.observation, state=state.actor)
        action = dist_utils.epsilon_greedy_sample(action_dist,
                                                  self._epsilon_greedy)

        return AlgStep(output=action, state=BcState(actor=new_state))

    def _actor_train_step_imitation(self, inputs: TimeStep, rollout_info,
                                    action_dist):

        exp_action = rollout_info.action
        im_loss = -action_dist.log_prob(exp_action)

        actor_info = LossInfo(loss=im_loss, extra=BcLossInfo(actor=im_loss))

        return actor_info

    def train_step_offline(self,
                           inputs: TimeStep,
                           state,
                           rollout_info,
                           pre_train=False):

        action_dist, new_state = self._predict_action(
            inputs.observation, state=state.actor)

        actor_loss = self._actor_train_step_imitation(inputs, rollout_info,
                                                      action_dist)

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                alf.summary.scalar("imitation_loss", actor_loss.loss.mean())

        info = BcInfo(actor=actor_loss)
        return AlgStep(
            rollout_info.action, state=BcState(actor=new_state), info=info)

    def calc_loss_offline(self, info, pre_train=False):

        actor_loss = info.actor
        return LossInfo(
            loss=actor_loss.loss, extra=BcLossInfo(actor=actor_loss.extra))
