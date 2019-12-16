# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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
from collections import namedtuple
from typing import Callable

import gin.tf
import tensorflow as tf

from tf_agents.networks.network import Network
import tf_agents.specs.tensor_spec as tensor_spec

from alf.algorithms.algorithm import Algorithm, AlgorithmStep, LossInfo
from alf.utils.data_buffer import DataBuffer
from tf_agents.trajectories.time_step import StepType
from alf.utils.encoding_network import EncodingNetwork
from alf.utils.common import transpose2
from alf.data_structures import ActionTimeStep
from alf.utils import math_ops

MISCInfo = namedtuple("MISCInfo", ["reward"])


@gin.configurable
class MISCAlgorithm(Algorithm):
    """Mutual Information-based State Control (MISC)
    Author: Rui Zhao
    Work done during a research internship at Horizon Robotics.
    The paper is currently under review in a conference.

    This algorithm generates the intrinsic reward based on the mutual information
    estimation between the goal states and the controllable states.

    See Zhao et al "Mutual Information-based State-Control for Intrinsically Motivated Reinforcement Learning",
    https://arxiv.org/abs/2002.01963
    """

    def __init__(self,
                 batch_size,
                 observation_spec,
                 action_spec,
                 soi_spec,
                 soc_spec,
                 split_observation_fn: Callable,
                 network: Network = None,
                 mi_r_scale=5000.0,
                 hidden_size=128,
                 buffer_size=100,
                 n_objects=1,
                 name="MISCAlgorithm"):
        """Create an MISCAlgorithm.

        Args:
            batch_size (int): batch size
            observation_spec (tf.TensorSpec): observation size
            action_spec (tf.TensorSpec): action size
            soi_spec (tf.TensorSpec): state of interest size
            soc_spec (tf.TensorSpec): state of context size
            split_observation_fn (Callable): split observation function.
                The input is observation and action concatenated.
                The outputs are the context states and states of interest
            network (Network): network for estimating mutual information (MI)
            mi_r_scale (float): scale factor of MI estimation
            hidden_size (int): number of hidden units in neural nets
            buffer_size (int): buffer size for the data buffer storing the trajectories
                for training the Mutual Information Neural Estimator
            n_objects: number of objects for estimating the mutual information reward
            name (str): the algorithm name, "MISCAlgorithm"
        """

        super(MISCAlgorithm, self).__init__(
            train_state_spec=[observation_spec, action_spec], name=name)

        assert isinstance(observation_spec, tf.TensorSpec), \
            "does not support nested observation_spec"
        assert isinstance(action_spec, tf.TensorSpec), \
            "does not support nested action_spec"

        if network is None:
            network = EncodingNetwork(
                input_tensor_spec=[soc_spec, soi_spec],
                fc_layer_params=(hidden_size, ),
                activation_fn='relu',
                last_layer_size=1,
                last_activation_fn='tanh')

        self._network = network

        self._traj_spec = tf.TensorSpec(
            shape=[batch_size] + [
                observation_spec.shape.as_list()[0] +
                action_spec.shape.as_list()[0]
            ],
            dtype=observation_spec.dtype)
        self._buffer_size = buffer_size
        self._buffer = DataBuffer(self._traj_spec, capacity=self._buffer_size)
        self._mi_r_scale = mi_r_scale
        self._n_objects = n_objects
        self._split_observation_fn = split_observation_fn

    def _mine(self, x_in, y_in):
        """Mutual Infomation Neural Estimator.

        Implement mutual information neural estimator from
        Belghazi et al "Mutual Information Neural Estimation"
        http://proceedings.mlr.press/v80/belghazi18a/belghazi18a.pdf
        'DV':  sup_T E_P(T) - log E_Q(exp(T))
        where P is the joint distribution of X and Y, and Q is the product
         marginal distribution of P. DV is a lower bound for
         KLD(P||Q)=MI(X, Y).

        """
        y_in_tran = transpose2(y_in, 1, 0)
        y_shuffle_tran = math_ops.shuffle(y_in_tran)
        y_shuffle = transpose2(y_shuffle_tran, 1, 0)

        # propagate the forward pass
        T_xy, _ = self._network([x_in, y_in])
        T_x_y, _ = self._network([x_in, y_shuffle])

        # compute the negative loss (maximize loss == minimize -loss)
        mean_exp_T_x_y = tf.reduce_mean(tf.math.exp(T_x_y), axis=1)
        loss = tf.reduce_mean(T_xy, axis=1) - tf.math.log(mean_exp_T_x_y)
        loss = tf.squeeze(loss, axis=-1)  # Mutual Information

        return loss

    def train_step(self,
                   time_step: ActionTimeStep,
                   state,
                   calc_intrinsic_reward=True):
        """
        Args:
            time_step (ActionTimeStep): input time_step data
            state (tuple): state for MISC (previous observation,
                previous previous action)
            calc_intrinsic_reward (bool): if False, only return the losses
        Returns:
            TrainStep:
                outputs: empty tuple ()
                state: tuple of observation and previous action
                info: (MISCInfo):
        """
        feature = time_step.observation
        prev_action = time_step.prev_action
        feature = tf.concat([feature_state, prev_action], axis=-1)
        prev_feature = tf.concat(state, axis=-1)

        feature_reshaped = tf.expand_dims(feature, axis=1)
        prev_feature_reshaped = tf.expand_dims(prev_feature, axis=1)
        feature_pair = tf.concat([prev_feature_reshaped, feature_reshaped], 1)
        feature_reshaped_tran = transpose2(feature_reshaped, 1, 0)

        def add_batch():
            self._buffer.add_batch(feature_reshaped_tran)

        if calc_intrinsic_reward:
            add_batch()

        if self._n_objects < 2:
            obs_tau_excludes_goal, obs_tau_achieved_goal = \
                self._split_observation_fn(feature_pair)
            loss = self._mine(obs_tau_excludes_goal, obs_tau_achieved_goal)
        elif self._n_objects == 2:
            obs_tau_excludes_goal, obs_tau_achieved_goal_1, obs_tau_achieved_goal_2 \
            = self._split_observation_fn(
                feature_pair)
            loss_1 = self._mine(obs_tau_excludes_goal, obs_tau_achieved_goal_1)
            loss_2 = self._mine(obs_tau_excludes_goal, obs_tau_achieved_goal_2)
            loss = loss_1 + loss_2

        intrinsic_reward = ()
        if calc_intrinsic_reward:
            # scale/normalize the MISC intrinsic reward
            if self._n_objects < 2:
                intrinsic_reward = tf.clip_by_value(self._mi_r_scale * loss, 0,
                                                    1)
            elif self._n_objects == 2:
                intrinsic_reward = tf.clip_by_value(
                    self._mi_r_scale * loss_1, 0,
                    1) + 1 * tf.clip_by_value(self._mi_r_scale * loss_2, 0, 1)

        return AlgorithmStep(
            outputs=(), state=[feature_state, prev_action], \
            info=MISCInfo(reward=intrinsic_reward))

    def calc_loss(self, info: MISCInfo):
        feature_tau_sampled = self._buffer.get_batch(
            batch_size=self._buffer_size)
        feature_tau_sampled_tran = transpose2(feature_tau_sampled, 1, 0)
        if self._n_objects < 2:
            obs_tau_excludes_goal, obs_tau_achieved_goal = self._split_observation_fn(
                feature_tau_sampled_tran)
            loss = self._mine(obs_tau_excludes_goal, obs_tau_achieved_goal)
        elif self._n_objects == 2:
            obs_tau_excludes_goal, obs_tau_achieved_goal_1, obs_tau_achieved_goal_2 = \
            self._split_observation_fn(
                feature_tau_sampled_tran)
            loss_1 = self._mine(obs_tau_excludes_goal, obs_tau_achieved_goal_1)
            loss_2 = self._mine(obs_tau_excludes_goal, obs_tau_achieved_goal_2)
            loss = loss_1 + loss_2

        neg_loss = -loss
        neg_loss_scalar = tf.reduce_mean(neg_loss)
        return LossInfo(scalar_loss=neg_loss_scalar)
