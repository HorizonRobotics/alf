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

import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.environments import trajectory
from tf_agents.environments.time_step import TimeStep
from tf_agents.networks.network import Network
from tf_agents.utils import eager_utils

from alf.algorithms import SimpleAlgorithm, LossStep


def run_rnn(rnn_func, inputs, states):
    """
    Args:
        rnn_func (callable): a function with signature run_func(input, state).
            It should return tuple of (output, next_state)
        inputs (nested Tensor) :
        states (nested Tensor) :
    Returns:
        outputs (nested Tensor): output of run_func concatenated in time
           dimesion.
        next_state (nested Tensor): last next_state output from rnn_func
    """
    raise NotImplementedError()


class SimpleAgent(tf_agent.TFAgent):
    """
    SimpleAgent encapsulate the common patterns for an agent with states.
    It also algorithm without states.
    """

    def __init__(self,
                 algorithm,
                 iterations_per_batch=1,
                 gradient_clipping=None,
                 debug_summaries=False,
                 summarize_grads_and_vars=False,
                 train_step_counter=None,
                 name=None):
        """
        Args:
          algorithm (SimpleAlgorithm):
          iterations_per_batch (int): number of training interations for each
            training batch
          gradient_clipping (None|0-D Tensor): Norm length to clip gradients.
          debug_summaries (bool): If true, subclasses should gather debug
            summaries.
          summarize_grads_and_vars (bool): If true, subclasses should
            additionally collect gradient and variable summaries.
          train_step_counter (Tensor): An optional counter to increment every
            time the train op is run.  Defaults to the global_step.
          name (str): The name of this agent. All variables in this module will
            fall under that name. Defaults to the class name.
        """
        tf.Module.__init__(self, name=name)
        self._algorithm = algorithm
        self._is_rnn_policy = bool(algorithm.policy_state_spec)
        self._iterations_per_batch = iterations_per_batch
        self._gradient_clipping = gradient_clipping
        super(SimpleAgent, self).__init__(
            algorithm.time_step_spec,
            algorithm.action_spec,
            policy=algorithm.policy,
            collect_policy=algorithm.collect_policy,
            train_sequence_length=None if self._is_rnn_policy else 2,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter)

    def _split_experience(self, trajectory):
        next_trajectory = tf.nest.map_structure(lambda x: x[:, 1:], trajectory)
        trajectory = tf.nest.map_structure(lambda x: x[:, :-1], trajectory)

        # Remove time dim if we are not using a recurrent network.
        if not self._is_rnn_policy:
            trajectory, next_trajectory = tf.nest.map_structure(
                lambda x: tf.squeeze(x, [1]), (trajectory, next_trajectory))

        return trajectory, next_trajectory

    # Use @common.function in graph mode or for speeding up.
    def _train(self, experience, weights):
        trajectory, next_trjectory = self._split_experience(experience)
        for _ in range(self._iterations_per_batch):
            loss_info = self._train_iter(trajectory, next_trjectory, weights)
        return loss_info

    def _train_iter(self, trajectory, next_trajectory, weights):
        batch_size = trajectory.discount.shape[0]

        if self._is_rnn_policy:
            critics, _ = run_rnn(self._algorithm.pre_critic, next_trajectory,
                                 self._algorithm.get_initial_state(batch_size))
        else:
            critics, _ = self._algorithm.pre_critic_step(next_trajectory)
        critics = self._algorithm.critic(trajectory, next_trajectory, critics)

        valid_mask = tf.cast(~trajectory.time_steps.is_last(), tf.float32)
        with tf.GradientTape() as tape:
            if self._is_rnn_policy:
                loss, _ = run_rnn(
                    self._algorithm.loss_step,
                    LossStep(trajectory=trajectory, critic=critics),
                    self._algorithm.get_initial_state(batch_size))
                loss = tf.nest.map_structure(lambda x: x * valid_mask, loss)
                loss = tf.nest.map_structure(
                    lambda x: tf.reduce_sum(x, axis=range(1, len(x.shape))),
                    loss)
            else:
                loss, _ = self._algorithm.loss_step(
                    LossStep(trajectory=trajectory, critic=critics))
                loss = tf.nest.map_structure(lambda x: x * valid_mask, loss)
            if weights is not None:
                loss = tf.nest.map_structure(lambda x: x * weights, loss)
            loss = tf.nest.map_structure(lambda x: x.reduce_mean(x), loss)
            total_loss = sum(tf.nest.flatten(loss))

        tf.debugging.check_numerics(total_loss, 'Loss is inf or nan')
        variables_to_train = self._algorithm.trainable_weights
        assert list(
            variables_to_train), "No variables in the agent's q_network."
        grads = tape.gradient(total_loss, variables_to_train)
        grads_and_vars = tuple(zip(grads, variables_to_train))
        if self._gradient_clipping is not None:
            grads_and_vars = eager_utils.clip_gradient_norms(
                grads_and_vars, self._gradient_clipping)

        if self._summarize_grads_and_vars:
            eager_utils.add_variables_summaries(grads_and_vars,
                                                self.train_step_counter)
            eager_utils.add_gradients_summaries(grads_and_vars,
                                                self.train_step_counter)

        self._algorithm.update(grads_and_vars)

        return tf_agent.LossInfo(total_loss, loss)
