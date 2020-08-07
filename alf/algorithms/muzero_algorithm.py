# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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

import gin
import torch

import alf
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.data_structures import AlgStep, Experience, LossInfo, namedtuple, TimeStep
from alf.experience_replayers.replay_buffer import BatchInfo, ReplayBuffer
from alf.algorithms.mcts_algorithm import MCTSAlgorithm
from alf.algorithms.mcts_models import MCTSModel, ModelOutput, ModelTarget
from alf.nest.utils import convert_device
from alf.utils import dist_utils

MuzeroInfo = namedtuple(
    'MuzeroInfo',
    [
        # actual actions taken in the next unroll_steps
        # [B, unroll_steps, ...]
        'action',

        # value computed by MCTSAlgorithm
        'value',

        # MCTSModelTarget
        # [B, unroll_steps + 1, ...]
        'target'
    ])


def scale_gradient(tensor, scale):
    """Scales the gradient for the backward pass."""
    return tensor * scale + tensor.detach() * (1 - scale)


@gin.configurable
class MuzeroAlgorithm(OffPolicyAlgorithm):
    """MuZero algorithm.

    MuZero is described in the paper:
    `Schrittwieser et. al. Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model <https://arxiv.org/abs/1911.08265>`_.

    The pseudocode can be downloaded from `<https://arxiv.org/src/1911.08265v2/anc/pseudocode.py>`_
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 model_ctor,
                 mcts_algorithm_ctor,
                 num_unroll_steps,
                 td_steps,
                 recurrent_gradient_scaling_factor=0.5,
                 debug_summaries=False,
                 name="MuZero"):
        """
        Args:
            observation_spec (TensorSpec): representing the observations.
            action_spec (BoundedTensorSpec): representing the actions.
            model_ctor (Callable): will be called as
                ``model_ctor(observation_spec=?, action_spec=?, debug_summaries=?)``
                to construct the model. The model should follow the interface
                ``alf.algorithms.mcts_models.MCTSModel``.
            mcts_algorithm_ctor (Callable): will be called as
                ``mcts_algorithm_ctor(observation_spec=?, action_spec=?, debug_summaries=?)``
                to construct an ``MCTSAlgorithm`` instance.
            num_unroll_steps (int): steps for unrolling the model during training.
            td_steps (int): bootstrap so many steps into the future for calculating
                the discounted return. -1 means to bootstrap to the end of the game.
                Can only used for environments whose rewards are zero except for
                the last step as the current implmentation only use the reward
                at the last step to calculate the return.
            recurrent_gradient_scaling_factor (float): the gradient go through
                the ``model.recurrent_inference`` is scaled by this factor. This
                is suggested in Appendix G.
            debug_summaries (bool):
            name (str):
        """
        assert not alf.nest.is_nested(observation_spec), (
            "Nested observation is not supported")
        assert not alf.nest.is_nested(action_spec), (
            "Nested action is not supported")
        model = model_ctor(
            observation_spec, action_spec, debug_summaries=debug_summaries)
        mcts = mcts_algorithm_ctor(
            observation_spec=observation_spec,
            action_spec=action_spec,
            debug_summaries=debug_summaries)
        mcts.set_model(model)
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            train_state_spec=(),
            predict_state_spec=mcts.predict_state_spec,
            rollout_state_spec=mcts.predict_state_spec,
            debug_summaries=debug_summaries,
            name=name)

        self._mcts = mcts
        self._model = model
        self._num_unroll_steps = num_unroll_steps
        self._td_steps = td_steps
        self._discount = mcts.discount
        self._recurrent_gradient_scaling_factor = recurrent_gradient_scaling_factor

    def predict_step(self, time_step: TimeStep, state, epsilon_greedy):
        return self._mcts.predict_step(time_step, state)

    def rollout_step(self, time_step: TimeStep, state):
        return self._mcts.predict_step(time_step, state)

    def train_step(self, exp: Experience, state):
        model_output = self._model.initial_inference(exp.observation)
        model_output_spec = dist_utils.extract_spec(model_output)
        model_outputs = [dist_utils.distributions_to_params(model_output)]
        info = exp.rollout_info

        for i in range(self._num_unroll_steps):
            model_output = self._model.recurrent_inference(
                model_output.state, info.action[:, i, ...])
            model_output = model_output._replace(
                state=scale_gradient(model_output.state, self.
                                     _recurrent_gradient_scaling_factor))
            model_outputs.append(
                dist_utils.distributions_to_params(model_output))

        model_outputs = alf.nest.utils.stack_nests(model_outputs, dim=1)
        model_outputs = dist_utils.params_to_distributions(
            model_outputs, model_output_spec)
        return AlgStep(info=self._model.calc_loss(model_outputs, info.target))

    @torch.no_grad()
    def preprocess_experience(self, experience: Experience):
        """Fill experience.rollout_info with MuzeroInfo

        Note that the shape of experience is [B, T, ...]
        """
        assert experience.batch_info != ()
        batch_info: BatchInfo = experience.batch_info
        replay_buffer: ReplayBuffer = experience.replay_buffer
        info_path: str = experience.rollout_info_field
        mini_batch_length = experience.step_type.shape[1]
        assert mini_batch_length == 1, (
            "Only support TrainerConfig.mini_batch_length=1, got %s" %
            mini_batch_length)

        value_field = info_path + '.value'
        candidate_actions_field = info_path + '.candidate_actions'
        candidate_action_visit_probabilities_field = (
            info_path + '.candidate_action_visit_probabilities')

        with alf.device(replay_buffer.device):
            positions = convert_device(batch_info.positions)  # [B]
            env_ids = convert_device(batch_info.env_ids)  # [B]

            # [B]
            steps_to_episode_end = replay_buffer.steps_to_episode_end(
                positions, env_ids)
            # [B]
            episode_end_positions = positions + steps_to_episode_end

            # [B, unroll_steps+1]
            positions = positions.unsqueeze(-1) + torch.arange(
                self._num_unroll_steps + 1)
            # [B, 1]
            env_ids = batch_info.env_ids.unsqueeze(-1)
            # [B, 1]
            episode_end_positions = episode_end_positions.unsqueeze(-1)

            beyond_episode_end = positions > episode_end_positions
            positions = torch.min(positions, episode_end_positions)

            if self._td_steps >= 0:
                values = self._calc_bootstrap_return(replay_buffer, env_ids,
                                                     positions, value_field)
            else:
                values = self._calc_monte_carlo_return(replay_buffer, env_ids,
                                                       positions, value_field)

            rewards = replay_buffer.get_field('reward', env_ids, positions)
            rewards[beyond_episode_end] = 0.
            candidate_actions = replay_buffer.get_field(
                candidate_actions_field, env_ids, positions)
            candidate_action_visit_probabilities = replay_buffer.get_field(
                candidate_action_visit_probabilities_field, env_ids, positions)
            game_overs = positions == episode_end_positions
            discount = replay_buffer.get_field('discount', env_ids, positions)
            # In the case of discount == 0, the game over may not always be correct
            # since the episode is truncated because of TimeLimit or incomplete
            # last episode in the replay buffer. There is no way to know for sure
            # the future game overs.
            game_overs = torch.min(game_overs, discount == 0.)
            values[game_overs] = 0.
            action = replay_buffer.get_field('action', env_ids, positions)

            rollout_info = MuzeroInfo(
                action=action,
                value=experience.rollout_info.value,
                target=ModelTarget(
                    reward=rewards,
                    action=candidate_actions,
                    action_probability=candidate_action_visit_probabilities,
                    value=values,
                    game_over=game_overs))

        # make the shape to [B, T, ...], where T=1
        rollout_info = alf.nest.map_structure(lambda x: x.unsqueeze(1),
                                              rollout_info)
        return experience._replace(rollout_info=convert_device(rollout_info))

    def _calc_bootstrap_return(self, replay_buffer, env_ids, positions,
                               value_field):
        # [B, unroll_steps+1]
        steps_to_episode_end = replay_buffer.steps_to_episode_end(
            positions, env_ids)
        # [B, unroll_steps+1]
        episode_end_positions = positions + steps_to_episode_end
        # [B, unroll_steps+1]

        bootstrap_n = steps_to_episode_end.clamp(max=self._td_steps)
        bootstrap_positions = positions + bootstrap_n

        values = replay_buffer.get_field(value_field, env_ids,
                                         bootstrap_positions)
        discount = replay_buffer.get_field('discount', env_ids,
                                           bootstrap_positions)
        values = values * discount * (self._discount**bootstrap_n)

        # [B, unroll_steps+1, td_steps]
        positions = 1 + positions.unsqueeze(-1) + torch.arange(self._td_steps)
        # [B, 1, 1]
        env_ids = env_ids.unsqueeze(-1)
        # [B, unroll_steps+1, 1]
        episode_end_positions = episode_end_positions.unsqueeze(-1)
        # [B, unroll_steps+1, td_steps]
        rewards = replay_buffer.get_field(
            'reward', env_ids, torch.min(positions, episode_end_positions))
        rewards[positions > episode_end_positions] = 0.
        discounts = self._discount**torch.arange(
            self._td_steps, dtype=torch.float32)
        return values + (rewards * discounts).sum(dim=-1)

    def _calc_monte_carlo_return(self, replay_buffer, env_ids, positions,
                                 value_field):
        # We only use the reward at the episode end.
        # [B, unroll_steps]
        steps_to_episode_end = replay_buffer.steps_to_episode_end(
            positions, env_ids)
        # [B, unroll_steps]
        episode_end_positions = positions + steps_to_episode_end

        reward = replay_buffer.get_field('reward', env_ids,
                                         episode_end_positions)
        # For the current implementation of replay buffer, the last episode is
        # likely to be incomplete, which means that the episode end is not the
        # real episode end and the corresponding discount is 1. So we bootstrap
        # with value in these cases.
        # TODO: only use complete episodes from replay buffer.
        discount = replay_buffer.get_field('discount', env_ids,
                                           episode_end_positions)
        value = replay_buffer.get_field(value_field, env_ids,
                                        episode_end_positions)
        reward = reward + self._discount * discount * value
        return reward * (self._discount**(steps_to_episode_end - 1))

    def calc_loss(self, experience, train_info: LossInfo):
        assert experience.batch_info != ()
        if (experience.batch_info != ()
                and experience.batch_info.importance_weights != ()):
            priority = (experience.rollout_info.value -
                        experience.rollout_info.target.value[..., 0])
            priority = priority.abs().sum(dim=1)
        else:
            priority = ()

        return train_info._replace(priority=priority)
