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
import typing

import alf
from alf.algorithms.data_transformer import (
    create_data_transformer, IdentityDataTransformer, RewardTransformer,
    SequentialDataTransformer)
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
from alf.trainers.policy_trainer import Trainer

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
                 reward_transformer=None,
                 calculate_priority=None,
                 train_reward_function=True,
                 train_game_over_function=True,
                 train_repr_prediction=False,
                 reanalyze_mcts_algorithm_ctor=None,
                 reanalyze_ratio=0.,
                 reanalyze_td_steps=5,
                 reanalyze_td_steps_func=None,
                 reanalyze_batch_size=None,
                 full_reanalyze=False,
                 data_transformer_ctor=None,
                 target_update_tau=1.,
                 target_update_period=1000,
                 config: Optional[TrainerConfig] = None,
                 enable_amp: bool = True,
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
            reward_transformer (Callable|None): if provided, will be used to
                transform reward.
            train_reward_function (bool): whether train reward function. If
                False, reward should only be given at the last step of an episode.
            calculate_priority (bool): whether to calculate priority. If not provided,
                will be same as ``TrainerConfig.priority_replay``. This is only
                useful if priority replay is enabled.
            train_game_over_function (bool): whether train game over function.
            train_repr_prediction (bool): whether to train to predict future
                latent representation.
            reanalyze_mcts_algorithm_ctor (Callable): will be called as
                ``reanalyze_mcts_algorithm_ctor(observation_spec=?, action_spec=?, debug_summaries=?, name=?)``
                to construct an ``MCTSAlgorithm`` instance. If not provided,
                ``mcts_algorithm_ctor`` will be used.
            reanalyze_ratio (float): float number in [0., 1.]. Reanalyze so much
                portion of data retrieved from replay buffer. Reanalyzing means
                using recent model to calculate the value and policy target.
            reanalyze_td_steps (int): the n for the n-step return for reanalyzing.
            reanalyze_td_steps_func (Callable): If provided, will be called as
                reanalyze_td_steps_func(sample_age, reanalyze_td_steps, current_max_age)
                to calculate the td_steps in reanalyze. sample_age is a Tensor
                whose elements are between 0 and 1 indicating the age of each sample.
                The age of the latest sample is 0. The age of the sample collected
                at the beginning of the training is current_max_age.
            reanalyze_batch_size (int|None): the memory usage may be too much for
                reanalyzing all the data for one training iteration. If so, provide
                a number for this so that it will analyzing the data in several
                batches.
            full_reanalyze (bool): if False, during reanalyze only the first
                ``num_unroll_steps+1`` steps are calculated using MCTS, and the next
                 ``reanalyze_td_steps`` are calculated from the model directly.
                 If True, all are calculated using MCTS.
            data_transformer_ctor (None|Callable|list[Callable]): if provided,
                will used to construct data transformer. Otherwise, the one
                provided in config will be used.
            target_update_tau (float): Factor for soft update of the target
                networks used for reanalyzing.
            target_update_period (int): Period for soft update of the target
                networks used for reanalyzing.
            config: The trainer config that will eventually be assigned to
                ``self._config``.
            enable_amp: whether to use automatic mixed precision for inference.
                This usually makes the algorithm run faster. However, the result
                may be different (mostly likely due to random fluctuation).
            debug_summaries (bool):
            name (str):

        """
        model = model_ctor(
            observation_spec,
            action_spec,
            num_unroll_steps=num_unroll_steps,
            debug_summaries=debug_summaries)
        mcts = mcts_algorithm_ctor(
            observation_spec=observation_spec,
            action_spec=action_spec,
            debug_summaries=debug_summaries,
            name="mcts")
        mcts.set_model(model)
        if calculate_priority is None:
            if config is not None:
                calculate_priority = config.priority_replay
            else:
                calculate_priority = False
        self._calculate_priority = calculate_priority
        self._device = alf.get_default_device()
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=mcts.predict_state_spec,
            predict_state_spec=mcts.predict_state_spec,
            rollout_state_spec=mcts.predict_state_spec,
            config=config,
            debug_summaries=debug_summaries,
            name=name)
        self._enable_amp = enable_amp
        self._mcts = mcts
        self._model = model
        self._num_unroll_steps = num_unroll_steps
        self._td_steps = td_steps
        self._discount = mcts.discount
        self._recurrent_gradient_scaling_factor = recurrent_gradient_scaling_factor
        self._reward_transformer = reward_transformer
        self._train_reward_function = train_reward_function
        self._train_game_over_function = train_game_over_function
        self._train_repr_prediction = train_repr_prediction
        self._reanalyze_ratio = reanalyze_ratio
        self._reanalyze_td_steps_func = reanalyze_td_steps_func
        self._reanalyze_td_steps = reanalyze_td_steps
        self._reanalyze_batch_size = reanalyze_batch_size
        self._full_reanalyze = full_reanalyze
        if data_transformer_ctor is not None:
            self._data_transformer = create_data_transformer(
                data_transformer_ctor, observation_spec)
        self._check_data_transformer()
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
                observation_spec,
                action_spec,
                num_unroll_steps=num_unroll_steps,
                debug_summaries=debug_summaries)
            self._update_target = common.get_target_updater(
                models=[self._model],
                target_models=[self._target_model],
                tau=target_update_tau,
                period=target_update_period)
            self._reanalyze_mcts.set_model(self._target_model)

    def _trainable_attributes_to_ignore(self):
        return ['_target_model', '_reanalyze_mcts']

    def _check_data_transformer(self):
        """Make sure data transformer does not contain reward transformer."""
        if isinstance(self._data_transformer, SequentialDataTransformer):
            transformers = self._data_transformer.members()
        else:
            transformers = [self._data_transformer]
        for transformer in transformers:
            assert not isinstance(transformer, RewardTransformer), (
                "DataTranformer for reward (%s) is not supported."
                "Please specify them using reward_transformer instead" %
                transformer)

    def predict_step(self, time_step: TimeStep, state):
        if self._reward_transformer is not None:
            time_step = time_step._replace(
                reward=self._reward_transformer(time_step.reward))
        with torch.cuda.amp.autocast(self._enable_amp):
            return self._mcts.predict_step(time_step, state)

    def rollout_step(self, time_step: TimeStep, state):
        if self._reward_transformer is not None:
            time_step = time_step._replace(
                reward=self._reward_transformer(time_step.reward))
        return self._mcts.predict_step(time_step, state)

    def train_step(self, exp: TimeStep, state, rollout_info: MuzeroInfo):
        def _hook(grad, name):
            alf.summary.scalar("MCTS_state_grad_norm/" + name, grad.norm())

        model_output = self._model.initial_inference(exp.observation)
        if alf.summary.should_record_summaries():
            model_output.state.state.register_hook(partial(_hook, name="s0"))
        model_output_spec = dist_utils.extract_spec(model_output)
        model_outputs = [dist_utils.distributions_to_params(model_output)]
        info = rollout_info

        for i in range(self._num_unroll_steps):
            model_output = self._model.recurrent_inference(
                model_output.state, info.action[:, i, ...])
            if alf.summary.should_record_summaries():
                model_output.state.state.register_hook(
                    partial(_hook, name="s" + str(i + 1)))
            model_output = model_output._replace(
                state=model_output.state._replace(
                    state=alf.nest.map_structure(
                        lambda x: scale_gradient(
                            x, self._recurrent_gradient_scaling_factor),
                        model_output.state.state)))
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
        """Fill rollout_info with MuzeroInfo.

        Especially, the training targets for representation learning is computed
        here with reanalyze and/or bootstrapping.

        Note that the shape of experience is [B, T, ...], where B is the batch
        size T is the mini batch length.

        """
        assert batch_info != ()
        replay_buffer: ReplayBuffer = batch_info.replay_buffer
        info_path: str = "rollout_info"
        info_path += "." + self.path if self.path else ""
        rollout_value = rollout_info.value
        value_field = info_path + '.value'
        candidate_actions_field = info_path + '.candidate_actions'
        candidate_action_policy_field = (
            info_path + '.candidate_action_policy')

        # Create aliases for mini_batch_size (B), mini_batch_length(T) and
        # predictive unroll steps (R) to make the implementation below more
        # succinct.
        B, T = root_inputs.step_type.shape
        R = self._num_unroll_steps

        with alf.device(replay_buffer.device):
            start_env_ids = convert_device(batch_info.env_ids)

            # [B, 1]
            folded_env_ids = start_env_ids.unsqueeze(-1)
            # [B, 1, 1]
            env_ids = folded_env_ids.unsqueeze(-1)

            # [B]
            start_positions = convert_device(batch_info.positions)

            # [B, T + R], capped at the end of the replay buffer.
            folded_positions = torch.min(
                start_positions.unsqueeze(-1) + torch.arange(T + R),
                replay_buffer.get_current_position()[start_env_ids, None] - 1)

            # [B, T, R + 1]
            positions = folded_positions.unfold(1, R + 1, 1)

            # [B, T]
            steps_to_episode_end = replay_buffer.steps_to_episode_end(
                positions[:, :, 0], env_ids[:, :, 0])

            # [B, T]
            episode_end_positions = positions[:, :, 0] + steps_to_episode_end
            # [B, T, 1]
            episode_end_positions = episode_end_positions.unsqueeze(-1)

            # [B, T, R + 1]
            beyond_episode_end = positions > episode_end_positions
            positions = torch.min(positions, episode_end_positions)

            if self._reanalyze_ratio > 0:
                # Here we assume state and info have similar name scheme.
                mcts_state_field = 'state' + info_path[len('rollout_info'):]

                # Applying the "unfold" trick, where we do reanalyze from the
                # starting position of B size-T trajectory for an unroll steps
                # of T + R - 1, and unfold it to [B, T, R + 1]
                if self._reanalyze_ratio < 1:
                    r = torch.randperm(B) < B * self._reanalyze_ratio
                    # [B', T + R, ...], B' = B * reanalyze_ratio
                    r_candidate_actions, r_candidate_action_policy, r_values = self._reanalyze(
                        replay_buffer, start_env_ids[r], start_positions[r],
                        mcts_state_field, T + R - 1)
                else:
                    # [B, T + R, ...]
                    candidate_actions, candidate_action_policy, values = self._reanalyze(
                        replay_buffer, start_env_ids, start_positions,
                        mcts_state_field, T + R - 1)

            if self._reanalyze_ratio < 1:
                if self._td_steps >= 0:
                    # [B, T + R]
                    values = self._calc_bootstrap_return(
                        replay_buffer, folded_env_ids, folded_positions,
                        value_field)
                else:
                    # [B, T + R]
                    values = self._calc_monte_carlo_return(
                        replay_buffer, folded_env_ids, folded_positions,
                        value_field)

                # [B, T + R, ...]
                candidate_actions = replay_buffer.get_field(
                    candidate_actions_field, folded_env_ids, folded_positions)
                # [B, T + R, ...]
                candidate_action_policy = replay_buffer.get_field(
                    candidate_action_policy_field, folded_env_ids,
                    folded_positions)

                if self._reanalyze_ratio > 0:
                    if candidate_actions != ():
                        candidate_actions[r] = r_candidate_actions
                    candidate_action_policy[r] = r_candidate_action_policy
                    values[r] = r_values

            # The operation unfold1 (unfold at dimension 1) transform a tensor
            # of shape [B, T + R, ...] to [B, T, R + 1, ...] by unfolding each
            # sequence of length T + R into T shorter sequences with indices at
            # [0:(R+1)], [1:(R+2)], .. until [(T-1):(T+R)].

            # A capped unfolding caps the index for each of such shorter
            # sequences at the episode boundary if it crosses the episode end.

            capped_unfold1_index = (
                torch.arange(B)[:, None, None],  # [B, 1, 1]
                torch.arange(T)[:, None] + torch.min(
                    steps_to_episode_end.unsqueeze(-1),
                    torch.arange(R + 1))  # [T, R + 1]
            )  # [B, T, R + 1]

            def _unfold1_adapting_episode_ends(x):
                return x[capped_unfold1_index]

            # In the logic above, they are computed in folded form to save
            # unnecessary retrieval and computation. They are unfolded here so
            # that the shape goes from [B, T + R, ...] to [B, T, R + 1, ...].
            candidate_actions, candidate_action_policy, values = alf.nest.map_structure(
                _unfold1_adapting_episode_ends,
                (candidate_actions, candidate_action_policy, values))

            game_overs = ()
            if self._train_game_over_function or self._train_reward_function:
                # [B, T, R + 1]
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
                                             positions[:, :, 1:])

        # TODO(breakds): Instead of caching the observation, we can compute the
        # representation targets directly and store the representation targets,
        # to save some space.
        observation = ()
        if self._train_repr_prediction:
            if type(self._data_transformer) == IdentityDataTransformer:
                observation = replay_buffer.get_field(
                    'observation', folded_env_ids, folded_positions)
                observation = _unfold1_adapting_episode_ends(observation)
            else:
                # In contrast to the preceding case, where we can first extract
                # observation and then unfold it, certain data transformers,
                # such as FrameStacker, are sensitive to the order in which the
                # unfold is applied. As a result, we choose to first unfold.
                observation, step_type = replay_buffer.get_field(
                    ('observation', 'step_type'), folded_env_ids,
                    folded_positions)

                # Unfold (and reshape)
                observation = _unfold1_adapting_episode_ends(observation)
                observation = observation.reshape(-1, *observation.shape[2:])
                step_type = _unfold1_adapting_episode_ends(step_type)
                step_type = step_type.reshape(-1, *step_type.shape[2:])

                exp = alf.data_structures.make_experience(
                    root_inputs, AlgStep(), state=())
                exp = exp._replace(
                    time_step=root_inputs._replace(
                        step_type=step_type, observation=observation),
                    batch_info=batch_info,
                    replay_buffer=replay_buffer)
                exp = self._data_transformer.transform_experience(exp)
                observation = exp.observation.reshape(
                    B, T, *exp.observation.shape[1:])

        # TODO(breakds): Should also include a mask in ModelTarget as an
        # indicator of overflow beyond the end of the replay buffer.
        rollout_info = MuzeroInfo(
            action=action,
            value=rollout_value,
            target=ModelTarget(
                reward=rewards,
                action=candidate_actions,
                action_policy=candidate_action_policy,
                value=values,
                game_over=game_overs,
                observation=observation))

        if self._reward_transformer:
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
        sum_reward, discount = self._sum_discounted_reward(
            replay_buffer, env_ids, positions, bootstrap_positions,
            self._td_steps)
        values = values * discount
        values = values * (self._discount**bootstrap_n.to(torch.float32))

        if not self._train_reward_function:
            # For this condition, we need to set the value at and after the last
            # step to be the last reward.
            rewards = self._get_reward(replay_buffer, env_ids,
                                       bootstrap_positions)
            values = torch.where(game_overs, rewards, values)
        return values + sum_reward

    def _sum_discounted_reward(self, replay_buffer, env_ids, positions,
                               bootstrap_positions, td_steps):
        """
        Returns:
            tuple
            - sum of discounted TimeStep.reward from positions + 1 to positions + bootstrap_positions
            - product of TimeStep.discount from positions to positions + bootstrap_positions
        """
        # [B, unroll_steps+1, td_steps+1]
        positions = positions.unsqueeze(-1) + torch.arange(td_steps + 1)
        # [B, 1, 1]
        env_ids = env_ids.unsqueeze(-1)
        # [B, unroll_steps+1, 1]
        bootstrap_positions = bootstrap_positions.unsqueeze(-1)
        # [B, unroll_steps+1, td_steps]
        rewards = self._get_reward(replay_buffer, env_ids,
                                   torch.min(positions, bootstrap_positions))
        rewards[positions > bootstrap_positions] = 0.
        discounts = replay_buffer.get_field(
            'discount', env_ids, torch.min(positions, bootstrap_positions))
        discounts = discounts.cumprod(dim=-1)
        d = discounts[..., :-1] * self._discount**torch.arange(
            td_steps, dtype=torch.float32)
        return (rewards[..., 1:] * d).sum(dim=-1), discounts[..., -1]

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
        if self._reward_transformer is not None:
            reward = self._reward_transformer(
                convert_device(reward, self._device)).cpu()
        return reward

    def _reanalyze(self,
                   replay_buffer: ReplayBuffer,
                   env_ids,
                   positions,
                   mcts_state_field,
                   horizon: Optional[int] = None):
        batch_size = env_ids.shape[0]
        mini_batch_size = batch_size
        if self._reanalyze_batch_size is not None:
            mini_batch_size = self._reanalyze_batch_size

        self._reanalyze_mcts.eval()
        result = []
        for i in range(0, batch_size, mini_batch_size):
            # Divide into several batches so that memory is enough.
            result.append(
                self._reanalyze1(replay_buffer, env_ids[i:i + mini_batch_size],
                                 positions[i:i + mini_batch_size],
                                 mcts_state_field, horizon))
        self._reanalyze_mcts.train()

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
        env_ids = env_ids.expand_as(positions)
        with alf.device(self._device):
            # [B, n1 + n2, ...]
            exp = replay_buffer.get_field(None, env_ids, positions)
            if type(self._data_transformer) != IdentityDataTransformer:
                # The shape of BatchInfo should be [B]
                exp = exp._replace(
                    batch_info=BatchInfo(env_ids[:, 0], positions[:, 0]),
                    replay_buffer=replay_buffer)
                exp = self._data_transformer.transform_experience(exp)
                exp = exp._replace(batch_info=(), replay_buffer=())

            def _split1(x):
                shape = x.shape[2:]
                if n2 > 0:
                    x = x[:, :n1, ...]
                return x.reshape(batch_size * n1, *shape)

            def _split2(x):
                shape = x.shape[2:]
                return x[:, n1:, ...].reshape(batch_size * n2, *shape)

            exp1 = alf.nest.map_structure(_split1, exp)
            exp2 = ()
            if n2 > 0:
                exp2 = alf.nest.map_structure(_split2, exp)

        return exp1, exp2

    def _reanalyze1(self,
                    replay_buffer: ReplayBuffer,
                    env_ids,
                    positions,
                    mcts_state_field,
                    horizon: Optional[int] = None):
        """Reanalyze one batch.

        This means:
        1. Re-plan the policy using MCTS for n1 = 1 + horizon to get fresh policy
        and value target.
        2. Caluclate the value for following n2 = reanalyze_td_steps so that we have value
        for a total of 1 + horizon + reanalyze_td_steps.
        3. Use these values and rewards from replay buffer to caculate n2-step
        bootstraped value target for the first n1 steps.

        In order to do 1 and 2, we need to get the observations for n1 + n2 steps
        and processs them using data_transformer.
        """
        batch_size = env_ids.shape[0]
        horizon = horizon or self._num_unroll_steps
        n1 = horizon + 1
        n2 = self._reanalyze_td_steps
        # Note that the retrievd next n positions are not capped by the ends of
        # the episodes.
        env_ids, positions = self._next_n_positions(replay_buffer, env_ids,
                                                    positions, horizon + n2)
        # [B, n1]
        positions1 = positions[:, :n1]
        game_overs = replay_buffer.get_field('discount', env_ids,
                                             positions1) == 0.

        steps_to_episode_end = replay_buffer.steps_to_episode_end(
            positions1, env_ids)
        if self._reanalyze_td_steps_func is None:
            bootstrap_n = steps_to_episode_end.clamp(max=n2)
        else:
            progress = Trainer.progress()
            current_pos = replay_buffer.get_current_position().max()
            age = progress * (1 - positions1 / current_pos)
            bootstrap_n = self._reanalyze_td_steps_func(age, n2, progress)
            bootstrap_n = torch.minimum(bootstrap_n, steps_to_episode_end)

        if self._full_reanalyze:
            # TODO: don't need to reanalyze all n1 + n2 steps because bootstrap_n
            # can be smaller than n2
            exp1, exp2 = self._prepare_reanalyze_data(replay_buffer, env_ids,
                                                      positions, n1 + n2, 0)
        else:
            exp1, exp2 = self._prepare_reanalyze_data(replay_buffer, env_ids,
                                                      positions, n1, n2)

        bootstrap_position = positions1 + bootstrap_n
        sum_reward, discount = self._sum_discounted_reward(
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
            with torch.cuda.amp.autocast(self._enable_amp):
                mcts_step = self._reanalyze_mcts.predict_step(
                    exp1, alf.nest.get_field(exp1, mcts_state_field))

            def _reshape(x):
                x = x.reshape(batch_size, -1, *x.shape[1:])
                return x[:, :n1] if self._full_reanalyze else x

            candidate_actions = mcts_step.info.candidate_actions
            if candidate_actions != ():
                candidate_actions = _reshape(candidate_actions)
            candidate_action_policy = mcts_step.info.candidate_action_policy
            candidate_action_policy = _reshape(candidate_action_policy)
            values = mcts_step.info.value.reshape(batch_size, -1)

            # 2. Calulate the value of the next n2 steps so that n2-step return
            # can be computed.
            if not self._full_reanalyze:
                with torch.cuda.amp.autocast(self._enable_amp):
                    model_output = self._target_model.initial_inference(
                        exp2.observation)
                values2 = model_output.value.reshape(batch_size, n2)
                values = torch.cat([values, values2], dim=1)

            # 3. Calculate n2-step return
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
        """expand position to include its next n positions, capped at the end of the
        replay buffer.

        Args:
            env_ids: [B]
            positions: [B]
        Returns:
            env_ids: [B, 1]
            positions: [B, n+1]

        """
        # [B, 1]
        env_ids = env_ids.unsqueeze(-1)
        # [B, n + 1]
        positions = positions.unsqueeze(-1) + torch.arange(n + 1)
        # [B, 1]
        current_pos = replay_buffer.get_current_position()[env_ids]
        # [B, n + 1]
        positions = torch.min(positions, current_pos - 1)
        return env_ids, positions

    def calc_loss(self, info: LossInfo):
        if self._calculate_priority:
            priority = info.loss.extra['value'].sqrt().sum(dim=0)
        else:
            priority = ()

        return LossInfo(
            scalar_loss=info.loss.loss.mean(),
            extra=alf.nest.map_structure(torch.mean, info.loss.extra),
            priority=priority)

    def after_update(self, root_inputs, info):
        if self._update_target is not None:
            self._update_target()


@alf.configurable
class LinearTdStepFunc(object):
    """Linearly decrease td steps from ``max_td_steps`` to ``min_td_steps``
    based on the age of a sample.

    If the age of a sample is more than ``max_bootstrap_age``, its td steps will
    be ``min_td_steps``. This is the "dynamic horizon" trick described in paper
    `Mastering Atari Games with Limited Data <https://arxiv.org/abs/2111.00210v1>`_
    """

    def __init__(self, max_bootstrap_age, min_td_steps=1):
        self._max_bootstrap_age = max_bootstrap_age
        self._min_td_steps = min_td_steps

    def __call__(self, age, max_td_steps, current_max_age):
        td_steps = self._min_td_steps + (max_td_steps - self._min_td_steps) * (
            1 - age / self._max_bootstrap_age).relu()
        return td_steps.ceil().to(torch.int64)
