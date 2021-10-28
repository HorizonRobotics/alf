# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""MuZero algorithm."""

from functools import partial
from typing import Optional
import torch

import alf
from alf.algorithms.data_transformer import create_data_transformer
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.config import TrainerConfig
from alf.data_structures import AlgStep, Experience, LossInfo, namedtuple, TimeStep
from alf.experience_replayers.replay_buffer import BatchInfo, ReplayBuffer
from alf.algorithms.mcts_algorithm import MCTSAlgorithm, MCTSInfo
from alf.algorithms.mcts_models import MCTSModel, ModelOutput, ModelTarget
from alf.nest.utils import convert_device
from alf.utils import common, dist_utils
from alf.utils.tensor_utils import scale_gradient
from alf.tensor_specs import TensorSpec

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
        'target',

        # Loss from training
        'loss',
    ],
    default_value=())


@alf.configurable
class MuzeroAlgorithm(OffPolicyAlgorithm):
    """MuZero algorithm.

    MuZero is described in the paper:
    `Schrittwieser et al. Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model <https://arxiv.org/abs/1911.08265>`_.

    The pseudocode can be downloaded from `<https://arxiv.org/src/1911.08265v2/anc/pseudocode.py>`_
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 model_ctor,
                 mcts_algorithm_ctor,
                 num_unroll_steps,
                 td_steps,
                 reward_spec=TensorSpec(()),
                 recurrent_gradient_scaling_factor=0.5,
                 reward_normalizer=None,
                 reward_clip_value=-1.,
                 calculate_priority=False,
                 train_reward_function=True,
                 train_game_over_function=True,
                 reanalyze_mcts_algorithm_ctor=None,
                 reanalyze_ratio=0.,
                 reanalyze_td_steps=5,
                 reanalyze_batch_size=None,
                 data_transformer_ctor=None,
                 target_update_tau=1.,
                 target_update_period=1000,
                 config: Optional[TrainerConfig] = None,
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
                ``mcts_algorithm_ctor(observation_spec=?, action_spec=?, debug_summaries=?, name=?)``
                to construct an ``MCTSAlgorithm`` instance.
            num_unroll_steps (int): steps for unrolling the model during training.
            td_steps (int): bootstrap so many steps into the future for calculating
                the discounted return. -1 means to bootstrap to the end of the game.
                Can only used for environments whose rewards are zero except for
                the last step as the current implmentation only use the reward
                at the last step to calculate the return.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            recurrent_gradient_scaling_factor (float): the gradient go through
                the ``model.recurrent_inference`` is scaled by this factor. This
                is suggested in Appendix G.
            reward_normalizer (Normalizer|None): if provided, will be used to
                normalize reward.
            train_reward_function (bool): whether train reward function. If
                False, reward should only be given at the last step of an episode.
            calculate_priority (bool): whether to calculate priority. This is
                only useful if priority replay is enabled.
            train_game_over_function (bool): whether train game over function.
            reanalyze_mcts_algorithm_ctor (Callable): will be called as
                ``mcts_algorithm_ctor(observation_spec=?, action_spec=?, debug_summaries=?, name=?)``
                to construct an ``MCTSAlgorithm`` instance. If not provided,
                ``mcts_algorithm_ctor`` will be used.
            reanalyze_ratio (float): float number in [0., 1.]. Reanalyze so much
                portion of data retrieved from replay buffer. Reanalyzing means
                using recent model to calculate the value and policy target.
            reanalyze_td_steps (int): the n for the n-step return for reanalyzing.
            reanalyze_batch_size (int|None): the memory usage may be too much for
                reanalyzing all the data for one training iteration. If so, provide
                a number for this so that it will analyzing the data in several
                batches.
            data_transformer_ctor (Callable|list[Callable]): should be same as
                ``TrainerConfig.data_transformer_ctor``.
            target_update_tau (float): Factor for soft update of the target
                networks used for reanalyzing.
            target_update_period (int): Period for soft update of the target
                networks used for reanalyzing.
            config: The trainer config that will eventually be assigned to
                ``self._config``. It is NOT NECESSARY to pass it to construct
                ``MuzeroAlgorithm`` and the agrument is kept here to be
                compatible to use with Agent.
            debug_summaries (bool):
            name (str):

        """
        model = model_ctor(
            observation_spec, action_spec, debug_summaries=debug_summaries)
        mcts = mcts_algorithm_ctor(
            observation_spec=observation_spec,
            action_spec=action_spec,
            debug_summaries=debug_summaries,
            name="mcts")
        mcts.set_model(model)
        self._calculate_priority = calculate_priority
        self._device = alf.get_default_device()
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=mcts.predict_state_spec,
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
        self._reward_normalizer = reward_normalizer
        self._reward_clip_value = reward_clip_value
        self._train_reward_function = train_reward_function
        self._train_game_over_function = train_game_over_function
        self._reanalyze_ratio = reanalyze_ratio
        self._reanalyze_td_steps = reanalyze_td_steps
        self._reanalyze_batch_size = reanalyze_batch_size
        self._data_transformer = None
        self._data_transformer_ctor = data_transformer_ctor
        self._mcts.set_model(model)
        self._update_target = None
        self._reanalyze_mcts = None
        if reanalyze_ratio > 0:
            reanalyze_mcts_algorithm_ctor = reanalyze_mcts_algorithm_ctor or mcts_algorithm_ctor
            self._reanalyze_mcts = reanalyze_mcts_algorithm_ctor(
                observation_spec=observation_spec,
                action_spec=action_spec,
                debug_summaries=debug_summaries,
                name="mcts_reanalyze")
            self._target_model = model_ctor(
                observation_spec, action_spec, debug_summaries=debug_summaries)
            self._update_target = common.get_target_updater(
                models=[self._model],
                target_models=[self._target_model],
                tau=target_update_tau,
                period=target_update_period)
            self._reanalyze_mcts.set_model(self._target_model)

    def _trainable_attributes_to_ignore(self):
        return ['_target_model', '_reanalyze_mcts']

    def predict_step(self, time_step: TimeStep, state):
        if self._reward_normalizer is not None:
            time_step = time_step._replace(
                reward=self._reward_normalizer.normalize(
                    time_step.reward, self._reward_clip_value))
        return self._mcts.predict_step(time_step, state)

    def rollout_step(self, time_step: TimeStep, state):
        if self._reward_normalizer is not None:
            self._reward_normalizer.update(time_step.reward)
            time_step = time_step._replace(
                reward=self._reward_normalizer.normalize(
                    time_step.reward, self._reward_clip_value))
        return self._mcts.predict_step(time_step, state)

    def train_step(self, exp: TimeStep, state, rollout_info):
        def _hook(grad, name):
            alf.summary.scalar("MCTS_state_grad_norm/" + name, grad.norm())

        model_output = self._model.initial_inference(exp.observation)
        if alf.summary.should_record_summaries():
            model_output.state.register_hook(partial(_hook, name="s0"))
        model_output_spec = dist_utils.extract_spec(model_output)
        model_outputs = [dist_utils.distributions_to_params(model_output)]
        info = rollout_info

        for i in range(self._num_unroll_steps):
            model_output = self._model.recurrent_inference(
                model_output.state, info.action[:, i, ...])
            if alf.summary.should_record_summaries():
                model_output.state.register_hook(
                    partial(_hook, name="s" + str(i + 1)))
            model_output = model_output._replace(
                state=scale_gradient(model_output.state, self.
                                     _recurrent_gradient_scaling_factor))
            model_outputs.append(
                dist_utils.distributions_to_params(model_output))

        model_outputs = alf.nest.utils.stack_nests(model_outputs, dim=1)
        model_outputs = dist_utils.params_to_distributions(
            model_outputs, model_output_spec)
        return AlgStep(
            info=info._replace(
                loss=self._model.calc_loss(model_outputs, info.target)))

    @torch.no_grad()
    def preprocess_experience(self, root_inputs: TimeStep,
                              rollout_info: MCTSInfo, batch_info):
        """Fill rollout_info with MuzeroInfo

        Note that the shape of experience is [B, T, ...]
        """
        assert batch_info != ()
        replay_buffer: ReplayBuffer = batch_info.replay_buffer
        info_path: str = "rollout_info"
        info_path += "." + self.path if self.path else ""
        mini_batch_length = root_inputs.step_type.shape[1]
        rollout_value = rollout_info.value
        assert mini_batch_length == 1, (
            "Only support TrainerConfig.mini_batch_length=1, got %s" %
            mini_batch_length)

        value_field = info_path + '.value'
        candidate_actions_field = info_path + '.candidate_actions'
        candidate_action_policy_field = (
            info_path + '.candidate_action_policy')

        with alf.device(replay_buffer.device):
            positions = convert_device(batch_info.positions)  # [B]
            env_ids = convert_device(batch_info.env_ids)  # [B]

            if self._reanalyze_ratio > 0:
                # Here we assume state and info have similar name scheme.
                mcts_state_field = 'state' + info_path[len('rollout_info'):]
                r = torch.rand(
                    root_inputs.step_type.shape[0]) < self._reanalyze_ratio
                r_candidate_actions, r_candidate_action_policy, r_values = self._reanalyze(
                    replay_buffer, env_ids[r], positions[r], mcts_state_field)

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

            candidate_actions = replay_buffer.get_field(
                candidate_actions_field, env_ids, positions)
            candidate_action_policy = replay_buffer.get_field(
                candidate_action_policy_field, env_ids, positions)

            if self._reanalyze_ratio > 0:
                if candidate_actions != ():
                    candidate_actions[r] = r_candidate_actions
                candidate_action_policy[r] = r_candidate_action_policy
                values[r] = r_values

            game_overs = ()
            if self._train_game_over_function or self._train_reward_function:
                game_overs = positions == episode_end_positions
                discount = replay_buffer.get_field('discount', env_ids,
                                                   positions)
                # In the case of discount != 0, the game over may not always be correct
                # since the episode is truncated because of TimeLimit or incomplete
                # last episode in the replay buffer. There is no way to know for sure
                # the future game overs.
                game_overs = game_overs & (discount == 0.)

            rewards = ()
            if self._train_reward_function:
                rewards = self._get_reward(replay_buffer, env_ids, positions)
                rewards[beyond_episode_end] = 0.
                values[game_overs] = 0.

            if not self._train_game_over_function:
                game_overs = ()

            action = replay_buffer.get_field('prev_action', env_ids,
                                             positions[:, 1:])

            rollout_info = MuzeroInfo(
                action=action,
                value=(),
                target=ModelTarget(
                    reward=rewards,
                    action=candidate_actions,
                    action_policy=candidate_action_policy,
                    value=values,
                    game_over=game_overs))

        # make the shape to [B, T, ...], where T=1
        rollout_info = alf.nest.map_structure(lambda x: x.unsqueeze(1),
                                              rollout_info)
        rollout_info = convert_device(rollout_info)
        rollout_info = rollout_info._replace(value=rollout_value)

        if self._reward_normalizer:
            root_inputs = root_inputs._replace(
                reward=rollout_info.target.reward[:, :, 0])
        return root_inputs, rollout_info

    def _calc_bootstrap_return(self, replay_buffer, env_ids, positions,
                               value_field):
        game_overs = replay_buffer.get_field('discount', env_ids,
                                             positions) == 0.

        # [B, unroll_steps+1]
        steps_to_episode_end = replay_buffer.steps_to_episode_end(
            positions, env_ids)
        # [B, unroll_steps+1]
        bootstrap_n = steps_to_episode_end.clamp(max=self._td_steps)
        bootstrap_positions = positions + bootstrap_n

        values = replay_buffer.get_field(value_field, env_ids,
                                         bootstrap_positions)
        discount = replay_buffer.get_field('discount', env_ids,
                                           bootstrap_positions)
        values = values * discount
        values = values * (self._discount**bootstrap_n.to(torch.float32))
        sum_reward = self._sum_discounted_reward(
            replay_buffer, env_ids, positions, bootstrap_positions,
            self._td_steps)

        if not self._train_reward_function:
            # For this condition, we need to set the value at and after the last
            # step to be the last reward.
            rewards = self._get_reward(replay_buffer, env_ids,
                                       bootstrap_positions)
            values = torch.where(game_overs, rewards, values)
        return values + sum_reward

    def _sum_discounted_reward(self, replay_buffer, env_ids, positions,
                               bootstrap_positions, td_steps):
        # [B, unroll_steps+1, td_steps]
        positions = 1 + positions.unsqueeze(-1) + torch.arange(td_steps)
        # [B, 1, 1]
        env_ids = env_ids.unsqueeze(-1)
        # [B, unroll_steps+1, 1]
        bootstrap_positions = bootstrap_positions.unsqueeze(-1)
        # [B, unroll_steps+1, td_steps]
        rewards = self._get_reward(replay_buffer, env_ids,
                                   torch.min(positions, bootstrap_positions))
        rewards[positions > bootstrap_positions] = 0.
        discounts = self._discount**torch.arange(td_steps, dtype=torch.float32)
        return (rewards * discounts).sum(dim=-1)

    def _calc_monte_carlo_return(self, replay_buffer, env_ids, positions,
                                 value_field):
        # We only use the reward at the episode end.
        # [B, unroll_steps]
        steps_to_episode_end = replay_buffer.steps_to_episode_end(
            positions, env_ids)
        # [B, unroll_steps]
        episode_end_positions = positions + steps_to_episode_end

        reward = self._get_reward(replay_buffer, env_ids,
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
        return reward * (self._discount**
                         (steps_to_episode_end - 1).clamp(min=0).to(
                             torch.float32))

    def _get_reward(self, replay_buffer, env_ids, positions):
        reward = replay_buffer.get_field('reward', env_ids, positions)
        if self._reward_normalizer is not None:
            reward = self._reward_normalizer.normalize(
                convert_device(reward, self._device),
                self._reward_clip_value).cpu()
        return reward

    def _reanalyze(self, replay_buffer: ReplayBuffer, env_ids, positions,
                   mcts_state_field):
        batch_size = env_ids.shape[0]
        mini_batch_size = batch_size
        if self._reanalyze_batch_size is not None:
            mini_batch_size = self._reanalyze_batch_size

        result = []
        for i in range(0, batch_size, mini_batch_size):
            # Divide into several batches so that memory is enough.
            result.append(
                self._reanalyze1(replay_buffer, env_ids[i:i + mini_batch_size],
                                 positions[i:i + mini_batch_size],
                                 mcts_state_field))

        if len(result) == 1:
            result = result[0]
        else:
            result = alf.nest.map_structure(
                lambda *tensors: torch.cat(tensors), *result)
        return convert_device(result)

    def _prepare_reanalyze_data(self, replay_buffer: ReplayBuffer, env_ids,
                                positions, n1, n2):
        """
        Get the n1 + n2 steps of experience indicated by ``positions`` and return
        as the first n1 as ``exp1`` and the next n2 steps as ``exp2``.
        """
        batch_size = env_ids.shape[0]
        n = n1 + n2
        flat_env_ids = env_ids.expand_as(positions).reshape(-1)
        flat_positions = positions.reshape(-1)
        exp = replay_buffer.get_field(None, flat_env_ids, flat_positions)

        if self._data_transformer_ctor is not None:
            if self._data_transformer is None:
                observation_spec = dist_utils.extract_spec(exp.observation)
                self._data_transformer = create_data_transformer(
                    self._data_transformer_ctor, observation_spec)

            # DataTransformer assumes the shape of exp is [B, T, ...]
            # It also needs exp.batch_info and exp.replay_buffer.
            exp = alf.nest.map_structure(lambda x: x.unsqueeze(1), exp)
            exp = exp._replace(
                batch_info=BatchInfo(flat_env_ids, flat_positions),
                replay_buffer=replay_buffer)
            exp = self._data_transformer.transform_experience(exp)
            exp = exp._replace(batch_info=(), replay_buffer=())
            exp = alf.nest.map_structure(lambda x: x.squeeze(1), exp)

        def _split1(x):
            shape = x.shape[1:]
            x = x.reshape(batch_size, n, *shape)
            return x[:, :n1, ...].reshape(batch_size * n1, *shape)

        def _split2(x):
            shape = x.shape[1:]
            x = x.reshape(batch_size, n, *shape)
            return x[:, n1:, ...].reshape(batch_size * n2, *shape)

        with alf.device(self._device):
            exp = convert_device(exp)
            exp1 = alf.nest.map_structure(_split1, exp)
            exp2 = alf.nest.map_structure(_split2, exp)

        return exp1, exp2

    def _reanalyze1(self, replay_buffer: ReplayBuffer, env_ids, positions,
                    mcts_state_field):
        """Reanalyze one batch.

        This means:
        1. Re-plan the policy using MCTS for n1 = 1 + num_unroll_steps to get fresh policy
        and value target.
        2. Caluclate the value for following n2 = reanalyze_td_steps so that we have value
        for a total of 1 + num_unroll_steps + reanalyze_td_steps.
        3. Use these values and rewards from replay buffer to caculate n2-step
        bootstraped value target for the first n1 steps.

        In order to do 1 and 2, we need to get the observations for n1 + n2 steps
        and processs them using data_transformer.
        """
        batch_size = env_ids.shape[0]
        n1 = self._num_unroll_steps + 1
        n2 = self._reanalyze_td_steps
        env_ids, positions = self._next_n_positions(
            replay_buffer, env_ids, positions, self._num_unroll_steps + n2)
        # [B, n1]
        positions1 = positions[:, :n1]
        game_overs = replay_buffer.get_field('discount', env_ids,
                                             positions1) == 0.

        steps_to_episode_end = replay_buffer.steps_to_episode_end(
            positions1, env_ids)
        bootstrap_n = steps_to_episode_end.clamp(max=n2)

        exp1, exp2 = self._prepare_reanalyze_data(replay_buffer, env_ids,
                                                  positions, n1, n2)

        bootstrap_position = positions1 + bootstrap_n
        discount = replay_buffer.get_field('discount', env_ids,
                                           bootstrap_position)
        sum_reward = self._sum_discounted_reward(
            replay_buffer, env_ids, positions1, bootstrap_position, n2)

        if not self._train_reward_function:
            rewards = self._get_reward(replay_buffer, env_ids,
                                       bootstrap_position)

        with alf.device(self._device):
            bootstrap_n = convert_device(bootstrap_n)
            discount = convert_device(discount)
            sum_reward = convert_device(sum_reward)
            game_overs = convert_device(game_overs)

            # 1. Reanalyze the first n1 steps to get both the updated value and policy
            mcts_step = self._reanalyze_mcts.predict_step(
                exp1, alf.nest.get_field(exp1, mcts_state_field))
            candidate_actions = ()
            if mcts_step.info.candidate_actions != ():
                candidate_actions = mcts_step.info.candidate_actions
                candidate_actions = candidate_actions.reshape(
                    batch_size, n1, *candidate_actions.shape[1:])
            candidate_action_policy = mcts_step.info.candidate_action_policy
            candidate_action_policy = candidate_action_policy.reshape(
                batch_size, n1, *candidate_action_policy.shape[1:])
            values1 = mcts_step.info.value.reshape(batch_size, n1)

            # 2. Calulate the value of the next n2 steps so that n2-step return
            # can be computed.
            model_output = self._target_model.initial_inference(
                exp2.observation)
            values2 = model_output.value.reshape(batch_size, n2)

            # 3. Calculate n2-step return
            values = torch.cat([values1, values2], dim=1)
            # [B, n1]
            bootstrap_pos = torch.arange(n1).unsqueeze(0) + bootstrap_n
            values = values[torch.arange(batch_size).
                            unsqueeze(-1), bootstrap_pos]
            values = values * discount * (self._discount**bootstrap_n.to(
                torch.float32))
            values = values + sum_reward
            if not self._train_reward_function:
                # For this condition, we need to set the value at and after the
                # last step to be the last reward.
                values = torch.where(game_overs, convert_device(rewards),
                                     values)
            return candidate_actions, candidate_action_policy, values

    def _next_n_positions(self, replay_buffer, env_ids, positions, n):
        """expand position to include its next n positions, limited by the
        episode end position.
        Args:
            env_ids: [B]
            positions: [B]
        Returns:
            env_ids: [B, 1]
            positions: [B, n+1]
        """
        # [B]
        steps_to_episode_end = replay_buffer.steps_to_episode_end(
            positions, env_ids)
        # [B]
        episode_end_positions = positions + steps_to_episode_end
        # [B, n+1]
        positions = positions.unsqueeze(-1) + torch.arange(n + 1)
        # [B, 1]
        env_ids = env_ids.unsqueeze(-1)
        # [B, 1]
        episode_end_positions = episode_end_positions.unsqueeze(-1)

        positions = torch.min(positions, episode_end_positions)
        return env_ids, positions

    def calc_loss(self, info: LossInfo):
        if self._calculate_priority:
            priority = (info.value - info.target.value[..., 0])
            priority = priority.abs().sum(dim=0)
        else:
            priority = ()

        return LossInfo(
            scalar_loss=info.loss.loss.mean(),
            extra=alf.nest.map_structure(torch.mean, info.loss.extra),
            priority=priority)

    def after_update(self, root_inputs, info):
        if self._update_target is not None:
            self._update_target()
