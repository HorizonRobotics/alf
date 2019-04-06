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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow as tf

from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment
from tf_agents.environments.parallel_py_environment import ProcessPyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer

import alf
from alf.agents import SimpleAgent
from alf.algorithms import DQNAlgorithm

flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')

FLAGS = flags.FLAGS


@gin.configurable
def dqn_train_eval(
        root_dir,
        env_name='CartPole-v0',
        env_load_fn=suite_gym.load,
        num_environments=1,
        train_sequence_length=1,
        replay_buffer_capacity=100000,
        # Params for target update
        target_update_tau=0.05,
        target_update_period=5,
        # Params for train
        train_steps_per_iteration=1,
        learning_rate=1e-3,
        epsilon_greedy=0.1,
        gamma=0.99,
        nstep_reward=1,
        gradient_clipping=None,
        debug_summaries=False,
        summarize_grads_and_vars=False):
    """A simple train and eval for DQN."""

    train_env = TFPyEnvironment(
        ParallelPyEnvironment(
            [lambda: env_load_fn(env_name)] * num_environments))
    eval_env = TFPyEnvironment(
        ProcessPyEnvironment(lambda: env_load_fn(env_name)))

    global_step = tf.Variable(0, dtype=tf.int64)

    if train_sequence_length > 1:
        q_net = models.QRnnModel(
            tf_env.observation_spec(),
            tf_env.action_spec(),
            input_fc_layer_params=input_fc_layer_params,
            lstm_size=lstm_size,
            output_fc_layer_params=output_fc_layer_params)
    else:
        q_net = q_network.QNetwork(
            tf_env.observation_spec(),
            tf_env.action_spec(),
            fc_layer_params=fc_layer_params)

    model = alf.Model()

    algorithm = DQNAlgorithm(
        model,
        optimizer=tf.optimizers.AdamOptimizer(learning_rate=learning_rate),
        epsilon_greedy=epsilon_greedy,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        gamma=gamma,
        nstep_reward=nstep_reward)

    tf_agent = SimpleAgent(
        algorithm,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step)

    replay_buffer = TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity)

    alf.train_evel(
        root_dir=root_dir,
        train_env=train_env,
        eval_env=eval_env,
        tf_agent=tf_agent,
        replay_buffer=replay_buffer,
        global_step=global_step)


def main(_):
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.enable_v2_behavior()
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    dqn_train_eval(FLAGS.root_dir)


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
