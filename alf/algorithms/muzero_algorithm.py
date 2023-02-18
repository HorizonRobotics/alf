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

from typing import Optional, Callable
import torch

import alf
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.config import TrainerConfig
from alf.data_structures import AlgStep, LossInfo, TimeStep
from alf.algorithms.mcts_algorithm import MCTSAlgorithm, MCTSInfo
from alf.algorithms.muzero_representation_learner import MuzeroRepresentationImpl, MuzeroInfo
from alf.tensor_specs import TensorSpec
from alf.trainers.policy_trainer import Trainer


@alf.configurable
class MuzeroAlgorithm(OffPolicyAlgorithm):
    """MuZero algorithm.
    MuZero is described in the paper:
    `Schrittwieser et al. Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model <https://arxiv.org/abs/1911.08265>`_.

    This is a wrapper that combines two sub algorithm components:

    1. A Muzero-style representation learner.

       The representation learner employs a MCTSModel to train a translation from a raw
       observation to its latent representation. The model is also used to predict the
       reward, values, policy, etc which will be used in the MCTS algorithm.

    2. A MCTS-based policy algorithm. It will perform tree search using the model provided
       by the representation learner to give the final policy on each predict and rollout
       step.

    NOTE: Currently, the MCTS-based policy algorithm is assumed to NOT have any learnable
    parameters. This means that training will only update the parameters of the underlying
    model in the representation learner, and training related hooks for example
    ``train_step()`` and ``preprocess_experience()`` will delegate directly to their
    counterparts in the representation learner. This behavior can be changed if needed in
    the future.

    """

    def __init__(
            self,
            observation_spec,
            action_spec,
            discount: float,
            reward_spec=TensorSpec(()),
            representation_learner_ctor: Callable[
                ..., MuzeroRepresentationImpl] = MuzeroRepresentationImpl,
            mcts_algorithm_ctor: Callable[..., MCTSAlgorithm] = MCTSAlgorithm,
            reward_transformer=None,
            config: Optional[TrainerConfig] = None,
            enable_amp: bool = True,
            checkpoint=None,
            debug_summaries=False,
            name="MuZero"):
        """
        Args:
            observation_spec (TensorSpec): representing the observations.
            action_spec (BoundedTensorSpec): representing the actions.
            representation_learner_ctor: It will be called to construct a
                MuZero-style representation learner. It is expected to be called
                as ``representation_learner_ctor(observation_spec=?,
                action_spec=?, reward_spec=?, discount=?, reward_transformer=?,
                enable_amp=?, config=?, debug_summaries=?, name=?)``.
            mcts_algorithm_ctor: will be called as
                ``mcts_algorithm_ctor(observation_spec=?, action_spec=?,
                discount=?, debug_summaries=?, name=?)`` to construct an
                ``MCTSAlgorithm`` instance. The constructed MCTS algorithm is
                assumed to have no learnable parameters. It also relies on the
                model from the representation learner ro run MCTS.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            reward_transformer (Callable|None): if provided, will be used to
                transform reward.
            config: The trainer config that will eventually be assigned to
                ``self._config``.
            enable_amp: whether to use automatic mixed precision for inference. This
                usually makes the algorithm run faster. However, the result may be
                different (mostly likely due to random fluctuation). Note that
                rollout_step is exempted from using AMP.
            checkpoint (None|str): a string in the format of "prefix@path",
                where the "prefix" is the multi-step path to the contents in the
                checkpoint to be loaded. "path" is the full path to the checkpoint
                file saved by ALF. Refer to ``Algorithm`` for more details.
            debug_summaries (bool):
            name (str):

        """

        representation_learner = representation_learner_ctor(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            discount=discount,
            reward_transformer=reward_transformer,
            enable_amp=enable_amp,
            config=config,
            debug_summaries=debug_summaries,
            name="muzero_repr")

        mcts = mcts_algorithm_ctor(
            observation_spec=representation_learner.model.repr_spec,
            action_spec=action_spec,
            discount=discount,
            model=representation_learner.model,
            debug_summaries=debug_summaries,
            name="muzero_policy")

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=mcts.train_state_spec,
            predict_state_spec=mcts.predict_state_spec,
            rollout_state_spec=mcts.rollout_state_spec,
            config=config,
            debug_summaries=debug_summaries,
            checkpoint=checkpoint,
            name=name)

        self._config = config
        self._repr_learner = representation_learner
        self._mcts = mcts
        self._reward_transformer = reward_transformer
        self._enable_amp = enable_amp

    def set_path(self, path):
        super().set_path(path)
        # Set the path of the representation learner to be the same as this algorithm, so
        # that the representation can effectively use the MuzeroAlgorithm's information to
        # compute the targets.
        self._repr_learner.set_path(self.path)

    def predict_step(self, time_step: TimeStep, state) -> AlgStep:
        if self._reward_transformer is not None:
            time_step = time_step._replace(
                reward=self._reward_transformer(time_step.reward))
        with torch.cuda.amp.autocast(self._enable_amp):
            latent = self._repr_learner.predict_step(time_step, state).output
            return self._mcts.predict_step(
                time_step._replace(observation=latent), state)

    def rollout_step(self, time_step: TimeStep, state) -> AlgStep:
        if self._reward_transformer is not None:
            time_step = time_step._replace(
                reward=self._reward_transformer(time_step.reward))
        latent = self._repr_learner.rollout_step(time_step, state).output
        return self._mcts.rollout_step(
            time_step._replace(observation=latent), state)

    def train_step(self, exp: TimeStep, state, rollout_info: MuzeroInfo):
        return self._repr_learner.train_step(exp, state, rollout_info)

    def preprocess_experience(self, root_inputs: TimeStep,
                              rollout_info: MCTSInfo, batch_info):
        return self._repr_learner.preprocess_experience(
            root_inputs, rollout_info, batch_info)

    def calc_loss(self, info: LossInfo):
        return self._repr_learner.calc_loss(info)

    def after_update(self, root_inputs, info):
        return self._repr_learner.after_update(root_inputs, info)
