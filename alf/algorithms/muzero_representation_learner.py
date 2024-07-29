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
from typing import Callable, Optional, Union, NamedTuple
import copy
import inspect

import torch

import alf
from alf.algorithms.data_transformer import (
    create_data_transformer, IdentityDataTransformer, RewardTransformer,
    SequentialDataTransformer)
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.config import TrainerConfig
from alf.data_structures import AlgStep, LossInfo, namedtuple, TimeStep, make_experience
from alf.experience_replayers.replay_buffer import BatchInfo, ReplayBuffer
from alf.algorithms.mcts_algorithm import MCTSInfo
from alf.algorithms.mcts_models import ModelTarget
from alf.nest.utils import convert_device
from alf.utils import common, dist_utils
from alf.utils.tensor_utils import scale_gradient
from alf.utils.schedulers import as_scheduler
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
class MuzeroRepresentationImpl(OffPolicyAlgorithm):
    """MuZero-style Representation Learner.

    MuZero is described in the paper:
    `Schrittwieser et al. Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model <https://arxiv.org/abs/1911.08265>`_.

    The pseudocode can be downloaded from `<https://arxiv.org/src/1911.08265v2/anc/pseudocode.py>`_

    This representation learner trains the underlying MCTSModel to

    1) Most importantly, produce a latent representation from an observation
    2) Predict the next latent representation given the current latent + an action
    3) Predict various targets (e.g. reward, value)

    Among the above, 1) can be used as the representation in combination with
    another RL algorithm; 2) and 3) can be used in policy improvements that
    requires a predictive model (e.g. Monte Carlo Tree Search).

    The model is trained with supervision on target prediction in 2) and 3).
    Some of the targets may be computed with the reanalyze component. Please
    refer to the original MuZero paper and the following paper for details.

    `Online and Offline Reinforcement Learning by Planning with a Learned Model <https://arxiv.org/abs/2104.06294>`_.

    """

    def __init__(
            self,
            observation_spec,
            action_spec,
            model_ctor,
            num_unroll_steps: int,
            td_steps: int,
            discount: float,
            reward_spec=TensorSpec(()),
            recurrent_gradient_scaling_factor: float = 0.5,
            reward_transformer=None,
            calculate_priority=None,
            train_reward_function=True,
            train_game_over_function=True,
            train_repr_prediction=False,
            train_policy=True,
            reanalyze_algorithm_ctor=None,
            reanalyze_ratio=0.,
            reanalyze_td_steps=5,
            reanalyze_td_steps_func=None,
            reanalyze_batch_size=None,
            full_reanalyze=False,
            priority_func: Union[
                Callable,
                str] = "lambda loss_info: loss_info.extra['value'].sqrt().sum(dim=0)",
            data_transformer_ctor=None,
            data_augmenter: Optional[Callable] = None,
            target_update_tau=1.,
            target_update_period=1000,
            config: Optional[TrainerConfig] = None,
            enable_amp: bool = True,
            random_action_after_episode_end=False,
            optimizer: Optional[torch.optim.Optimizer] = None,
            checkpoint=None,
            debug_summaries=False,
            name="MuzeroRepresentationImpl"):
        """
        Args:
            observation_spec (TensorSpec): representing the observations.
            action_spec (BoundedTensorSpec): representing the actions.
            model_ctor (Callable): will be called as
                ``model_ctor(observation_spec=?, action_spec=?, debug_summaries=?)``
                to construct the model. The model should follow the interface
                ``alf.algorithms.mcts_models.MCTSModel``.
            num_unroll_steps: steps for unrolling the model during training.
            td_steps: bootstrap so many steps into the future for calculating
                the discounted return. -1 means to bootstrap to the end of the game.
                Can only used for environments whose rewards are zero except for
                the last step as the current implementation only use the reward
                at the last step to calculate the return.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            recurrent_gradient_scaling_factor (float): the gradient go through
                the ``model.recurrent_inference`` is scaled by this factor. This
                is suggested in Appendix G.
            reward_transformer (Callable|None): if provided, will be used to
                transform reward.
            calculate_priority (bool): whether to calculate priority. If not provided,
                will be same as ``TrainerConfig.priority_replay``. This is only
                useful if priority replay is enabled.
            train_reward_function (bool): whether train reward function. If
                False, reward should only be given at the last step of an episode.
            train_game_over_function (bool): whether train game over function.
            train_repr_prediction (bool): whether to train to predict future
                latent representation.
            train_policy (bool): whether to train a policy. Note that training
                policy is REQUIRED when the model is used in MCTS algorithm.
            reanalyze_algorithm_ctor (Callable): will be called as
                ``reanalyze_algorithm_ctor(observation_spec=?,
                action_spec=?, discount=?, debug_summaries=?, name=?)`` to
                construct an ``Algorithm`` instance for reanalyze. It can also
                optionally accept an additional argument 'model'. If so, an
                model constructed using ``model_ctor`` will be passed to the
                constructor.
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
            priority_func: the function for calculating priority. If it is a str,
                ``eval(priority_func)`` will be called first to convert it a ``Callable``.
                It is called as ``priority_func(loss_info)``, where loss_info is
                the temporally stacked ``LossInfo`` structure returned from
                ``MCTSModel.calc_loss()``.
            data_transformer_ctor (None|Callable|list[Callable]): if provided,
                will used to construct data transformer. Otherwise, the one
                provided in config will be used.
            data_augmenter: If provided, will be called to perform data augmentation
                as ``data_augmenter(observation)`` for training observations,
                where the shape of observation is [B, T, ...] if ``train_repr_prediction``
                is False, and [B, T*(R+1), ...] if ``train_repr_prediction`` is True.
                B is mini-batch size, T is mini-batch length and R is ``num_unroll_steps``.
            target_update_tau (float): Factor for soft update of the target
                networks used for reanalyzing.
            target_update_period (int): Period for soft update of the target
                networks used for reanalyzing.
            config: The trainer config that will eventually be assigned to
                ``self._config``.
            enable_amp: whether to use automatic mixed precision for inference.
                This usually makes the algorithm run faster. However, the result
                may be different (mostly likely due to random fluctuation).
            random_action_after_episode_end: If False, the actions used to predict
                future states after the end of an episode will be the same as the
                last action. If True, they will be uniformly sampled.
            optimizer: the optimizer for independently training the representation.
            checkpoint (None|str): a string in the format of "prefix@path",
                where the "prefix" is the multi-step path to the contents in the
                checkpoint to be loaded. "path" is the full path to the checkpoint
                file saved by ALF. Refer to ``Algorithm`` for more details.
            debug_summaries (bool):
            name (str):

        """
        model = model_ctor(
            observation_spec,
            action_spec,
            num_unroll_steps=num_unroll_steps,
            debug_summaries=debug_summaries)
        if calculate_priority is None:
            if config is not None:
                calculate_priority = config.priority_replay
            else:
                calculate_priority = False
        self._calculate_priority = calculate_priority
        self._priority_func = eval(priority_func) if type(
            priority_func) == str else priority_func
        self._device = alf.get_default_device()

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=(),
            config=config,
            optimizer=optimizer,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name)
        self._enable_amp = enable_amp
        self._model = model
        self._num_unroll_steps = num_unroll_steps
        self._td_steps = td_steps
        self._discount = discount
        self._recurrent_gradient_scaling_factor = recurrent_gradient_scaling_factor
        self._reward_transformer = reward_transformer
        self._train_reward_function = train_reward_function
        self._train_game_over_function = train_game_over_function
        self._train_repr_prediction = train_repr_prediction
        self._train_policy = train_policy
        self._reanalyze_ratio = reanalyze_ratio
        self._reanalyze_td_steps_func = reanalyze_td_steps_func
        self._reanalyze_td_steps = reanalyze_td_steps
        self._reanalyze_batch_size = reanalyze_batch_size
        self._full_reanalyze = full_reanalyze
        self._data_augmenter = data_augmenter
        self._random_action_after_episode_end = random_action_after_episode_end
        if data_transformer_ctor is not None:
            self._data_transformer = create_data_transformer(
                data_transformer_ctor, observation_spec)
        self._check_data_transformer()
        self._update_target = None
        self._reanalyze_algorithm = None
        if reanalyze_ratio > 0:
            assert reanalyze_algorithm_ctor is not None, (
                'Must specify reanalyze_algorithm_ctor when reanalyze_ratio > 0'
            )
            self._target_model = model_ctor(
                observation_spec,
                action_spec,
                num_unroll_steps=num_unroll_steps,
                debug_summaries=debug_summaries)
            self._update_target = common.TargetUpdater(
                models=[self._model],
                target_models=[self._target_model],
                tau=target_update_tau,
                period=target_update_period)
            if 'model' in inspect.signature(
                    reanalyze_algorithm_ctor).parameters:
                model_kwargs = dict(model=self._target_model)
            else:
                model_kwargs = dict()
            self._reanalyze_algorithm = reanalyze_algorithm_ctor(
                observation_spec=self._model.repr_spec,
                action_spec=action_spec,
                **model_kwargs,
                debug_summaries=debug_summaries,
                name="reanalyze_algorithm")

    @property
    def model(self):
        return self._model

    def _trainable_attributes_to_ignore(self):
        return ['_target_model', '_reanalyze_algorithm']

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
        with torch.cuda.amp.autocast(self._enable_amp):
            return AlgStep(
                output=self._model.initial_representation(
                    time_step.observation),
                state=(),
                info=())

    def rollout_step(self, time_step: TimeStep, state):
        return AlgStep(
            output=self._model.initial_representation(time_step.observation),
            state=(),
            info=())

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
        info_target = info.target
        if self._train_repr_prediction:
            # [B*(R+1), ...]
            obs = alf.nest.map_structure(lambda x: x.reshape(-1, *x.shape[2:]),
                                         info.target.observation)
            with torch.no_grad():
                with torch.cuda.amp.autocast(self._enable_amp):
                    target_repr = self._model._representation_net(obs)[0]
            # [B, R+1, ...]
            target_repr = target_repr.reshape(-1, self._num_unroll_steps + 1,
                                              *target_repr.shape[1:])
            info_target = info.target._replace(observation=target_repr)
        return AlgStep(
            info=info._replace(
                loss=self._model.calc_loss(model_outputs, info_target)))

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
            folded_positions = start_positions.unsqueeze(-1) + torch.arange(T +
                                                                            R)
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

            # [B, T + R], capped at the end of the replay buffer.
            folded_positions = torch.min(
                folded_positions,
                replay_buffer.get_current_position()[start_env_ids, None] - 1)

            # [B, T, R + 1], now capped at episode ends
            positions = torch.min(positions, episode_end_positions)

            if self._reanalyze_ratio > 0:
                # Here we assume state and info have similar name scheme.
                policy_state_field = 'state' + info_path[len('rollout_info'):]

                # Applying the "unfold" trick, where we do reanalyze from the
                # starting position of B size-T trajectory for an unroll steps
                # of T + R - 1, and unfold it to [B, T, R + 1]
                if self._reanalyze_ratio < 1:
                    r = torch.randperm(B) < B * self._reanalyze_ratio
                    # [B', T + R, ...], B' = B * reanalyze_ratio
                    r_candidate_actions, r_candidate_action_policy, r_values = self._reanalyze(
                        replay_buffer, start_env_ids[r], start_positions[r],
                        policy_state_field, T + R - 1)
                else:
                    # [B, T + R, ...]
                    candidate_actions, candidate_action_policy, values = self._reanalyze(
                        replay_buffer, start_env_ids, start_positions,
                        policy_state_field, T + R - 1)
            # [B, T]
            last_discount = replay_buffer.get_field(
                'discount', env_ids[:, :, 0], positions[:, :, -1])
            # [B, T]
            is_partial_trajectory = last_discount != 0

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

            def _set_rand_action(a, spec):
                a[rand_mask] = spec.sample((rand_mask_size, ))

            if self._random_action_after_episode_end:
                rand_mask = beyond_episode_end[:, :, 1:]
                rand_mask_size = rand_mask.sum()
                if rand_mask_size > 0:
                    alf.nest.map_structure(_set_rand_action, action,
                                           self._action_spec)
        observation = ()
        if self._train_repr_prediction:
            if type(self._data_transformer) == IdentityDataTransformer:
                observation = replay_buffer.get_field(
                    'observation', folded_env_ids, folded_positions)
                # [B, T, R + 1, ...]
                observation = alf.nest.map_structure(
                    _unfold1_adapting_episode_ends, observation)
                # [B * T, R + 1, ...]
                observation = alf.nest.map_structure(
                    lambda x: x.reshape(-1, *x.shape[2:]), observation)
            else:
                # In contrast to the preceding case, where we can first extract
                # observation and then unfold it, certain data transformers,
                # such as FrameStacker, are sensitive to the order in which the
                # unfold is applied. As a result, we choose to first unfold.
                observation, step_type = replay_buffer.get_field(
                    ('observation', 'step_type'), folded_env_ids,
                    folded_positions)

                # Unfold (and reshape)
                # [B, T, R+1, ...]
                observation = alf.nest.map_structure(
                    _unfold1_adapting_episode_ends, observation)
                # [B*T, R+1, ...]
                observation = alf.nest.map_structure(
                    lambda x: x.reshape(-1, *x.shape[2:]), observation)
                step_type = _unfold1_adapting_episode_ends(step_type)
                step_type = step_type.reshape(-1, *step_type.shape[2:])

                # Will also need to update the batch info to be of shape [B *
                # T,] marking the starting positions.
                transformed_batch_info = BatchInfo(
                    replay_buffer=replay_buffer,
                    env_ids=batch_info.env_ids.repeat_interleave(T),
                    # Note that positions are already capped by episode ends.
                    positions=positions[:, :, 0].reshape(-1))

                exp = alf.data_structures.make_experience(
                    root_inputs, AlgStep(), state=())
                exp = exp._replace(
                    time_step=root_inputs._replace(
                        step_type=step_type, observation=observation),
                    batch_info=transformed_batch_info,
                    replay_buffer=replay_buffer)
                # [B*T, R+1, ...]
                observation = self._data_transformer.transform_experience(
                    exp).observation
            if self._data_augmenter is not None:
                observation = alf.nest.map_structure(
                    lambda x: x.reshape(B, T * (R + 1), *x.shape[2:]),
                    observation)
                observation = self._data_augmenter(observation)
                observation = alf.nest.map_structure(
                    lambda x: x.reshape(B, T, R + 1, *x.shape[2:]),
                    observation)
                # [B, T, ...]
                input_obs = alf.nest.map_structure(lambda x: x[:, :, 0, ...],
                                                   observation)
                root_inputs = root_inputs._replace(observation=input_obs)
            else:
                observation = alf.nest.map_structure(
                    lambda x: x.reshape(B, T, R + 1, *x.shape[2:]),
                    observation)

        if self._data_augmenter is not None and not self._train_repr_prediction:
            input_obs = self._data_augmenter(input_obs)
            root_inputs = root_inputs._replace(observation=input_obs)

        rollout_info = MuzeroInfo(
            action=action,
            target=ModelTarget(
                is_partial_trajectory=is_partial_trajectory,
                beyond_episode_end=beyond_episode_end,
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
                   policy_state_field,
                   horizon: Optional[int] = None):
        batch_size = env_ids.shape[0]
        mini_batch_size = batch_size
        if self._reanalyze_batch_size is not None:
            mini_batch_size = self._reanalyze_batch_size

        self._reanalyze_algorithm.eval()
        result = []
        for i in range(0, batch_size, mini_batch_size):
            # Divide into several batches so that memory is enough.
            result.append(
                self._reanalyze1(replay_buffer, env_ids[i:i + mini_batch_size],
                                 positions[i:i + mini_batch_size],
                                 policy_state_field, horizon))
        self._reanalyze_algorithm.train()

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
                    policy_state_field,
                    horizon: Optional[int] = None):
        """Reanalyze one batch.

        This means:
        1. Re-plan the policy using MCTS for n1 = 1 + horizon to get fresh policy
        and value target.
        2. Calculate the value for following n2 = reanalyze_td_steps so that we have value
        for a total of 1 + horizon + reanalyze_td_steps.
        3. Use these values and rewards from replay buffer to calculate n2-step
        bootstrapped value target for the first n1 steps.

        In order to do 1 and 2, we need to get the observations for n1 + n2 steps
        and process them using data_transformer.
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
                latent = self._target_model.initial_representation(
                    exp1.observation)
                exp1 = exp1._replace(
                    time_step=exp1.time_step._replace(observation=latent))
                policy_step = self._reanalyze_algorithm.rollout_step(
                    exp1, alf.nest.get_field(exp1, policy_state_field))

            def _reshape(x):
                x = x.reshape(batch_size, -1, *x.shape[1:])
                return x[:, :n1] if self._full_reanalyze else x

            candidate_action_policy = ()
            candidate_actions = ()
            if self._train_policy:
                candidate_actions = policy_step.info.candidate_actions
                if candidate_actions != ():
                    candidate_actions = _reshape(candidate_actions)
                candidate_action_policy = policy_step.info.candidate_action_policy
                candidate_action_policy = _reshape(candidate_action_policy)
            values = policy_step.info.value.reshape(batch_size, -1)

            # 2. Calculate the value of the next n2 steps so that n2-step return
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
            # Make sure that priority is float32 so that replay buffer will not
            # complain when updating the priority.
            priority = self._priority_func(info.loss).to(torch.float32)
        else:
            priority = ()

        return LossInfo(
            loss=info.loss.loss, extra=info.loss.extra, priority=priority)

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


class MuzeroRepresentationTrainingOptions(NamedTuple):
    """The options for training the Muzero Representation.

    When used together with an RL algorithm, the representation training does
    not necessarily share the training options with the RL algorithm. Therefore,
    we use this class to hold the training options private to the Muzero
    representation learner.

    """
    interval: int = 1  # Update the model every this number of iterations.
    mini_batch_length: int = 1
    mini_batch_size: int = 256
    num_updates_per_train_iter: int = 10
    replay_buffer_length: int = 100000
    initial_collect_steps: int = 2000
    priority_replay: bool = True
    priority_replay_alpha: float = 1.2
    priority_replay_beta: float = 0.0


@alf.configurable
class MuzeroRepresentationLearner(OffPolicyAlgorithm):
    """Learn representation following the MuZero style.

    This is a thin wrapper over the MuzeroRepresentationImpl, so as to make it
    possible to work in combination with an RL algorithm (within ``Agent``).

    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 config: TrainerConfig,
                 training_options: Optional[
                     MuzeroRepresentationTrainingOptions] = None,
                 reward_spec=TensorSpec(()),
                 impl_cls: Callable[
                     ..., MuzeroRepresentationImpl] = MuzeroRepresentationImpl,
                 debug_summaries: bool = False,
                 name: str = "MuZeroRepresentationLearner"):
        """Construct a MuzeroRepresentationLearner.

        Args:
            observation_spec (TensorSpec): representing the observations.
            action_spec (BoundedTensorSpec): representing the actions.
            config: The trainer config, usually passed down from ``Agent``.
            training_options: The representation learner trains its underlying model
                independent of the RL algorithm, and therefore will need a separate
                set of parameters for the training options. See
                ``MuzeroRepresentationTrainingOptions`` above for details. If not
                set, training will not happen.
            reward_spec: a rank-1 or rank-0 tensor spec representing the
                reward(s). Will passed down to the underlying wrapped
                ``MuzeroRepresentationImpl``.
            impl_cls: a callable to construct the underlying
                ``MuzeroRepresentationImpl``. It will be called as ``impl_cls(
                observation_spec=?, action_spec=?, reward_spec=?, config=?,
                debug_summaries=?)``.
            debug_summaries:
            name:

        """
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=(),
            config=None,
            debug_summaries=debug_summaries,
            name=name)

        self._training_options = training_options

        # Override the training behavior related parameters in the config when
        # ``training_options`` is explicitly provided, and pass it as the
        # configuration for the underlying implementation ``self._impl``.
        updated = copy.copy(config)
        if training_options is not None:
            updated.whole_replay_buffer_training = False
            updated.clear_replay_buffer = False
            updated.mini_batch_length = training_options.mini_batch_length
            updated.mini_batch_size = training_options.mini_batch_size
            updated.num_updates_per_train_iter = training_options.num_updates_per_train_iter
            updated.replay_buffer_length = training_options.replay_buffer_length
            updated.initial_collect_steps = training_options.initial_collect_steps
            updated.priority_replay = training_options.priority_replay
            updated.priority_replay_alpha = as_scheduler(
                training_options.priority_replay_alpha)
            updated.priority_replay_beta = as_scheduler(
                training_options.priority_replay_beta)

        self._impl = impl_cls(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            config=updated,
            debug_summaries=debug_summaries)
        self._impl.force_params_visible_to_parent = True
        assert self._impl._reanalyze_ratio in [
            0.0, 1.0
        ], ('Currently MuzeroRepresentationLearner only support reanalyze ratio 0.0 or 1.0'
            )

        if self._impl._reanalyze_ratio > 0:
            assert config.use_rollout_state, (
                'use_rollout_state needs to be True when reanalyze is used.')

    @property
    def output_spec(self):
        """Access the spec of the produced representation.

        This will be used as the observation spec for the subsequent RL
        algorithm.

        """
        return self._impl._model.repr_spec

    def predict_step(self, time_step: TimeStep, state):
        return self._impl.rollout_step(time_step, state)

    def rollout_step(self, time_step: TimeStep, state):
        repr_step = self._impl.rollout_step(time_step, state)
        if self._training_options is not None:
            # Save in the representation learner's own replay buffer. Note that
            # ``observe_for_replay`` when called for the first time will have the
            # side effect of creating the replay buffer.
            if self._impl._replay_buffer is None:
                self._impl.set_replay_buffer(
                    time_step.env_id.shape[0],
                    self._training_options.replay_buffer_length,
                    self._training_options.priority_replay)
            exp = make_experience(time_step.untransformed, repr_step, state)
            self._impl.observe_for_replay(exp)
        return repr_step

    def train_step(self, exp: TimeStep, state, rollout_info):
        return self._impl.rollout_step(exp, state)

    def preprocess_experience(self, root_inputs: TimeStep, rollout_info,
                              batch_info):
        return root_inputs, ()

    def calc_loss(self, info):
        # The calc_loss() here does nothing so that ``Agent`` will only handle
        # the loss from other sub algorithm such as RL algorithm.
        #
        # The actual loss for training the representation itself is within
        # ``self._impl``.
        return LossInfo(loss=(), extra={})

    def after_update(self, root_inputs, info):
        pass

    def after_train_iter(self, experience, info):
        if self._training_options is None:
            return

        # Independently run the training logic for the MuZero representation
        # learner's implementation.
        if alf.summary.get_global_counter(
        ) % self._training_options.interval == 0:
            self._impl.train_from_replay_buffer(update_global_counter=False)
