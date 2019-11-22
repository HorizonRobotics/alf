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

import gin.tf
import tensorflow as tf

from tf_agents.networks.network import Network
import tf_agents.specs.tensor_spec as tensor_spec

from alf.algorithms.algorithm import Algorithm, AlgorithmStep, LossInfo
from alf.utils.data_buffer import DataBuffer
from tf_agents.trajectories.time_step import StepType
from alf.utils.conditional_ops import conditional_update, run_if

MISCInfo = namedtuple("MISCInfo", ["reward", "loss"])


@gin.configurable
class MISCAlgorithm(Algorithm):
    """Mutual Information-based State Control (MISC)
    Author: Rui Zhao 
    Work done during a research internship at Horizon Robotics. 
    The paper is currently under review in a conference.

    This algorithm generates the intrinsic reward based on the mutual information estimation between the states of interests and the context states.

    See Zhao et al "Self-Supervised State-Control through Intrinsic Mutual Information Rewards"
    """

    def __init__(self,
                 feature_spec,
                 traj_spec,
                 soi_spec,
                 soc_spec,
                 mi_r_scale=5000,
                 hidden_size=256,
                 misc_layerx: Network = None,
                 misc_layery: Network = None,
                 misc_layero: Network = None,
                 buffer_size=100,
                 n_objects=1,
                 empowerment=False,
                 env_name='SocialBot-PlayGround-v0',
                 name="MISCAlgorithm"):
        """Create an MISCAlgorithm.
        
        Args:
            feature_spec: size of the state and the action
            traj_spec: (num_parallel_environments, feature_spec)
            soi_spec: (None, None, state of interest size)
            soc_spec: (None, None, state of context size)
            mi_r_scale: scale factor of mi estimation
            hidden_size: number of hidden units in neural nets
            misc_layerx: input layer with context states as input, output dimension is of hidden_size
            misc_layery: input layer with states of interest as input, output dimension is of hidden_size
            misc_layero: output layer with with input shape (None, None, hidden_size), output dimension is of 1
            buffer_size: buffer size for the data buffer storing the trajectories for training the Mutual Information Neural Estimator
            n_objects: number of objects for estimating the mutual information
            empowerment: if empowerment is true, then use action instead of the context states
            name: the algorithm name, "MISCAlgorithm"
        """

        super(MISCAlgorithm, self).__init__(
            train_state_spec=feature_spec, name=name)

        feature_dim = tf.nest.flatten(feature_spec)[0].shape[-1]

        if misc_layerx is None:
            misc_layerx = tf.keras.layers.Dense(
                hidden_size, input_shape=soc_spec.shape, activation=None)
            misc_layery = tf.keras.layers.Dense(
                hidden_size, input_shape=soi_spec.shape, activation=None)
            misc_layero = tf.keras.layers.Dense(
                1, input_shape=(None, None, hidden_size), activation=None)

        self._misc_layerx = misc_layerx
        self._misc_layery = misc_layery
        self._misc_layero = misc_layero

        self._traj_spec = traj_spec
        self._buffer_size = buffer_size
        self._buffer = DataBuffer(self._traj_spec, capacity=self._buffer_size)
        self._mi_r_scale = mi_r_scale
        self._n_objects = n_objects
        self._empowerment = empowerment
        self._env_name = env_name

    def split_observation_tf(self, o):

        dimo = o.get_shape().as_list()[-1]

        if self._env_name in ['SocialBot-PlayGround-v0']:
            if dimo == 21:
                task_specific_ob, agent_pose, agent_vel, internal_states, action = tf.split(
                    o, [3, 6, 6, 4, 2], axis=-1)

                agent_pose_1, agent_pose_2 = tf.split(
                    agent_pose, [3, 3], axis=-1)
                joint_pose_1, joint_pose_2, joint_vel_1, joint_vel_2 = tf.split(
                    internal_states, [1, 1, 1, 1], axis=-1)
                joint_1 = tf.concat([joint_pose_1, joint_vel_1], axis=-1)
                joint_2 = tf.concat([joint_pose_2, joint_vel_2], axis=-1)

                if self._n_objects == 1:
                    obs_achieved_goal = task_specific_ob
                    obs_excludes_goal = agent_pose_1
                    if self._empowerment:
                        obs_excludes_goal = action
                elif self._n_objects == 0:
                    obs_achieved_goal = joint_1
                    obs_excludes_goal = joint_2

                return (obs_excludes_goal, obs_achieved_goal)

            elif (dimo == 24) and (self._n_objects == 2):
                task_specific_ob_1, task_specific_ob_2, agent_pose, agent_vel, internal_states, action = tf.split(
                    o, [3, 3, 6, 6, 4, 2], axis=-1)

                agent_pose_1, agent_pose_2 = tf.split(
                    agent_pose, [3, 3], axis=-1)

                obs_achieved_goal_1 = task_specific_ob_1
                obs_achieved_goal_2 = task_specific_ob_2
                obs_excludes_goal = agent_pose_1

                return (obs_excludes_goal, obs_achieved_goal_1,
                        obs_achieved_goal_2)
            else:
                print(
                    'Please specify the state of interests and the context state in the misc_alorighm.py first.'
                )
                exit()
        else:
            print(
                'Please specify the state of interests and the context state in the misc_alorighm.py first.'
            )
            exit()

    def mine(self, x_in, y_in):
        """Mutual Infomation Neural Estimator.

        Implement mutual information neural estimator from
        Belghazi et al "Mutual Information Neural Estimation"
        http://proceedings.mlr.press/v80/belghazi18a/belghazi18a.pdf
        'DV':  sup_T E_P(T) - log E_Q(exp(T))
        where P is the joint distribution of X and Y, and Q is the product marginal distribution of P. DV is a lower bound for KLD(P||Q)=MI(X, Y).

        """
        y_in_tran = tf.transpose(y_in, perm=[1, 0, 2])
        y_shuffle_tran = tf.gather(
            y_in_tran, tf.random.shuffle(tf.range(tf.shape(y_in_tran)[0])))
        y_shuffle = tf.transpose(y_shuffle_tran, perm=[1, 0, 2])
        x_conc = tf.concat([x_in, x_in], axis=-2)
        y_conc = tf.concat([y_in, y_shuffle], axis=-2)

        # propagate the forward pass
        layerx = self._misc_layerx(inputs=x_conc)
        layery = self._misc_layery(inputs=y_conc)
        layer2 = tf.nn.relu(layerx + layery)
        output = self._misc_layero(inputs=layer2)
        output = tf.nn.tanh(output)

        # split in T_xy and T_x_y predictions
        N_samples = tf.shape(x_in)[-2]
        T_xy = output[:, :N_samples, :]
        T_x_y = output[:, N_samples:, :]

        # compute the negative loss (maximise loss == minimise -loss)
        mean_exp_T_x_y = tf.reduce_mean(tf.math.exp(T_x_y), axis=-2)
        loss = tf.reduce_mean(T_xy, axis=-2) - tf.math.log(mean_exp_T_x_y)
        loss = tf.squeeze(loss)  # Mutual Information

        return loss

    def train_step(self, inputs, state, calc_intrinsic_reward=True):
        """
        Args:
            inputs (tuple): observation
            state (Tensor):  state for MISC (previous feature)
            calc_intrinsic_reward (bool): if False, only return the losses
        Returns:
            TrainStep:
                outputs: empty tuple ()
                state:  empty tuple ()
                info: (MISCInfo):
        """
        feature_state, prev_action = inputs
        feature = tf.concat([feature_state, prev_action], axis=-1)
        prev_feature = state

        feature_reshaped = tf.expand_dims(feature, axis=1)
        prev_feature_reshaped = tf.expand_dims(prev_feature, axis=1)
        feature_pair = tf.concat([prev_feature_reshaped, feature_reshaped], 1)
        feature_reshaped_tran = tf.transpose(feature_reshaped, perm=[1, 0, 2])

        def add_batch():
            self._buffer.add_batch(feature_reshaped_tran)

        # run add_batch function, if there are new trajectories being rollouted
        run_if(
            tf.reduce_all(
                tf.equal(
                    tf.shape(feature),
                    tf.constant(self._traj_spec.shape.as_list()))), add_batch)

        if self._n_objects < 2:
            obs_tau_excludes_goal, obs_tau_achieved_goal = self.split_observation_tf(
                feature_pair)
            loss = self.mine(obs_tau_excludes_goal, obs_tau_achieved_goal)
        elif self._n_objects == 2:
            obs_tau_excludes_goal, obs_tau_achieved_goal_1, obs_tau_achieved_goal_2 = self.split_observation_tf(
                feature_pair)
            loss_1 = self.mine(obs_tau_excludes_goal, obs_tau_achieved_goal_1)
            loss_2 = self.mine(obs_tau_excludes_goal, obs_tau_achieved_goal_2)
            loss = loss_1 + loss_2

        intrinsic_reward = ()
        if calc_intrinsic_reward:
            # scale/normalize the MISC intrinsic reward
            if self._n_objects < 2:
                intrinsic_reward = tf.clip_by_value(self._mi_r_scale * loss,
                                                    *(0, 1))
            elif self._n_objects == 2:
                intrinsic_reward = tf.clip_by_value(
                    self._mi_r_scale * loss_1, *(0, 1)) + 1 * tf.clip_by_value(
                        self._mi_r_scale * loss_2, *(0, 1))

        return AlgorithmStep(
            outputs=(), state=feature, info=MISCInfo(reward=intrinsic_reward))

    def calc_loss(self, info: MISCInfo):
        feature_tau_sampled = self._buffer.get_batch(
            batch_size=self._buffer_size)
        feature_tau_sampled_tran = tf.transpose(
            feature_tau_sampled, perm=[1, 0, 2])
        if self._n_objects < 2:
            obs_tau_excludes_goal, obs_tau_achieved_goal = self.split_observation_tf(
                feature_tau_sampled_tran)
            loss = self.mine(obs_tau_excludes_goal, obs_tau_achieved_goal)
        elif self._n_objects == 2:
            obs_tau_excludes_goal, obs_tau_achieved_goal_1, obs_tau_achieved_goal_2 = self.split_observation_tf(
                feature_tau_sampled_tran)
            loss_1 = self.mine(obs_tau_excludes_goal, obs_tau_achieved_goal_1)
            loss_2 = self.mine(obs_tau_excludes_goal, obs_tau_achieved_goal_2)
            loss = loss_1 + loss_2

        neg_loss = -loss
        neg_loss_scalar = tf.reduce_mean(neg_loss)
        return LossInfo(scalar_loss=neg_loss_scalar)
