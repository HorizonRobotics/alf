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

from typing import Optional
import copy
import torch

from alf.algorithms.data_transformer import IdentityDataTransformer
from alf.data_structures import namedtuple
from alf.algorithms.config import TrainerConfig
from alf.algorithms.ppg import PPGRolloutInfo, PPGTrainInfo, PPGAuxPhaseLoss, ppg_network_forward
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.data_structures import TimeStep, AlgStep, LossInfo
from alf.experience_replayers.replay_buffer import ReplayBuffer
from alf.utils import dist_utils
from alf.tensor_specs import TensorSpec

# Data structure to store the options for PPG's auxiliary phase
# training iterations.
PPGAuxOptions = namedtuple(
    'PPGAuxOptions',
    [
        # Set to False to disable aux phase for good so that the PPG algorithm
        # degenerate to PPO
        'enabled',

        # Perform one auxiliary phase update after this number of normal
        # training iterations
        'interval',

        # Counterpart to TrainerConfig.mini_batch_length for aux phase. When set
        # to None, it will automatically overridden by unroll_length.
        'mini_batch_length',

        # Counterpart to TrainerConfig.mini_batch_size for aux phase
        'mini_batch_size',

        # Counterpart to TrainerConfig.num_updates_per_train_iter for aux phase
        'num_updates_per_train_iter',
    ],
    default_values={
        'enabled': True,
        'interval': 32,
        'mini_batch_length': None,
        'mini_batch_size': 8,
        'num_updates_per_train_iter': 6,
    })


class PPGAuxAlgorithm(OffPolicyAlgorithm):
    """An algorithm that performs the auxiliary phase update of PPG.

    The algorithm is used as a sub algorithm of PPGAlgorithm. Auxiliary phase
    updates does not require new rollouts. Instead it will collect all of the
    experiences since the last auxiliary phase updates in ITS OWN replay buffer.

    """

    # TODO(breakds): Also conduct an experiment to see whether we can just
    # do sampling from the auxiliary replay buffer to achieve similar
    # training performance.

    def __init__(self,
                 observation_spec,
                 action_spec,
                 reward_spec=TensorSpec(()),
                 config: Optional[TrainerConfig] = None,
                 optimizer: torch.optim.Optimizer = None,
                 dual_actor_value_network=None,
                 aux_options: PPGAuxOptions = PPGAuxOptions(),
                 debug_summaries=False,
                 name: str = 'PPGAuxAlgorithm'):
        """Construct a PPGAuxAlgorithm instances.

        Args:

            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            config (TrainerConfig): config for training. config only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            optimizer: optimizer used for auxiliary phase update.
            dual_actor_value_network: the underlying network for PPG algorithm.
                PPGAuxAlgorithm does not own the network. Instead, this should
                be shared reference from the parent PPGAlgorithm.
            aux_options: Options that controls the auxiliary phase training.
            name (str): Name of this algorithm.

        """
        # Use a config with overridden fields as auxiliary phase updates has a
        # bunch of its own options.
        updated_config = copy.copy(config)
        updated_config.unroll_length = config.unroll_length * aux_options.interval
        updated_config.whole_replay_buffer_training = True
        updated_config.clear_replay_buffer = True
        updated_config.mini_batch_length = (aux_options.mini_batch_length
                                            or config.unroll_length)
        updated_config.mini_batch_size = aux_options.mini_batch_size
        updated_config.num_updates_per_train_iter = aux_options.num_updates_per_train_iter

        # Since we are going to store already-transformed experience in the
        # replay buffer, the aux algorithm shall not inherit the data
        # transformer from the parent algorithm (PPGAlgorithm).
        updated_config.data_transformer = IdentityDataTransformer(
            observation_spec=observation_spec)

        super().__init__(
            config=updated_config,
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            predict_state_spec=dual_actor_value_network.state_spec,
            train_state_spec=dual_actor_value_network.state_spec,
            optimizer=optimizer,
            debug_summaries=debug_summaries,
            name=name)

        self._network = dual_actor_value_network
        self._loss = PPGAuxPhaseLoss()
        self._interval = aux_options.interval

    @property
    def interval(self):
        return self._interval

    def observe_for_aux_replay(self, exp):
        """Save the experience in the replay buffer for auxiliary phase update.

        Args:

            exp (nested Tensor): Experience to be saved. The shape is [B, ...]
                where B is the batch size of the batch environment.

        """
        if not self._use_rollout_state:
            exp = exp._replace(state=())
        # Set the experience spec explicitly if it is not set, based on this
        # (sample) experience
        if not self._experience_spec:
            self._experience_spec = dist_utils.extract_spec(exp, from_dim=1)
        exp = dist_utils.distributions_to_params(exp)

        # Construct the aux specific replay buffer if not present.
        if self._replay_buffer is None:
            exp_spec = dist_utils.to_distribution_param_spec(
                self._experience_spec)
            # Note that this unroll_length is the updated one for auxiliary
            # phase update. It is equal to auxiliary phase interval * policy
            # phase unroll_length (see __init__()).
            max_length = self._config.unroll_length
            max_length += 1 + self._num_earliest_frames_ignored
            self._replay_buffer = ReplayBuffer(
                data_spec=exp_spec,
                num_environments=exp.env_id.shape[0],
                max_length=max_length,
                prioritized_sampling=False,
                num_earliest_frames_ignored=self._num_earliest_frames_ignored,
                name='ppg_aux_replay_buffer')

        self._replay_buffer.add_batch(exp, exp.env_id)

    def train_step(self, inputs: TimeStep, state,
                   plain_rollout_info: PPGRolloutInfo) -> AlgStep:
        alg_step = ppg_network_forward(
            self._network, inputs, state, require_aux=True)

        train_info = PPGTrainInfo(
            action=plain_rollout_info.action,
            rollout_value=plain_rollout_info.value,
            rollout_action_distribution=plain_rollout_info.
            action_distribution).absorbed(alg_step.info)

        return alg_step._replace(info=train_info)

    def calc_loss(self, info: PPGTrainInfo) -> LossInfo:
        return self._loss(info)
