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
from typing import Callable

from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow as tf

from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.metrics import metric_utils, tf_metrics
from tf_agents.agents.tf_agent import TFAgent
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.utils import common as tfa_common

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('num_iterations', 100000,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')

FLAGS = flags.FLAGS


@gin.configurable
def train_eval(
        root_dir,
        train_env: TFEnvironment,
        eval_env: TFEnvironment,
        tf_agent: TFAgent,
        replay_buffer: ReplayBuffer,
        num_iterations=2000000,
        train_sequence_length=2,
        # Params for collect
        initial_collect_steps=1000,
        collect_steps_per_iteration=1,
        # Params for train
        train_steps_per_iteration=1,
        batch_size=64,
        use_tf_functions=True,
        # Params for eval
        num_eval_episodes=10,
        eval_interval=10000,
        # Params for checkpoints, summaries, and logging
        log_interval=1000,
        summary_interval=1000,
        summaries_flush_secs=10,
        debug_summaries=False,
        summarize_grads_and_vars=False,
        eval_metrics_callback=None):
    """A simple train and eval for DDPG."""
    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')

    train_summary_writer = tf.summary.create_file_writer(
        train_dir, flush_millis=summaries_flush_secs * 1000)
    train_summary_writer.set_as_default()

    eval_summary_writer = tf.summary.create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000)
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
    ]

    with tf.summary.record_if(
            lambda: tf.math.equal(global_step % summary_interval, 0)):

        tf_agent.initialize()

        train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric(),
        ]

        eval_policy = tf_agent.policy
        collect_policy = tf_agent.collect_policy

        initial_collect_driver = DynamicStepDriver(
            train_env,
            collect_policy,
            observers=[replay_buffer.add_batch],
            num_steps=initial_collect_steps)

        collect_driver = DynamicStepDriver(
            train_env,
            collect_policy,
            observers=[replay_buffer.add_batch] + train_metrics,
            num_steps=collect_steps_per_iteration)

        if use_tf_functions:
            initial_collect_driver.run = tfa_common.function(
                initial_collect_driver.run)
            collect_driver.run = tfa_common.function(collect_driver.run)
            tf_agent.train = tfa_common.function(tf_agent.train)

        # Collect initial replay data.
        logging.info(
            'Initializing replay buffer by collecting experience for %d steps '
            'with a random policy.', initial_collect_steps)
        initial_collect_driver.run()

        results = metric_utils.eager_compute(
            eval_metrics,
            eval_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics',
        )
        if eval_metrics_callback is not None:
            eval_metrics_callback(results, global_step.numpy())
        metric_utils.log_metrics(eval_metrics)

        time_step = None
        policy_state = collect_policy.get_initial_state(train_env.batch_size)

        timed_at_step = global_step.numpy()
        time_acc = 0

        # Dataset generates trajectories with shape [Bx2x...]
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=train_sequence_length + 1).prefetch(3)
        iterator = iter(dataset)

        for _ in range(num_iterations):
            start_time = time.time()
            time_step, policy_state = collect_driver.run(
                time_step=time_step,
                policy_state=policy_state,
            )
            for _ in range(train_steps_per_iteration):
                experience, _ = next(iterator)
                train_loss = tf_agent.train(experience)
            time_acc += time.time() - start_time

            if global_step.numpy() % log_interval == 0:
                logging.info('step = %d, loss = %f', global_step.numpy(),
                             train_loss.loss)
                steps_per_sec = (
                    global_step.numpy() - timed_at_step) / time_acc
                logging.info('%.3f steps/sec', steps_per_sec)
                tf.summary.scalar(
                    name='global_steps_per_sec',
                    data=steps_per_sec,
                    step=global_step)
                timed_at_step = global_step.numpy()
                time_acc = 0

            for train_metric in train_metrics:
                train_metric.tf_summaries(
                    train_step=global_step, step_metrics=train_metrics[:2])

            if global_step.numpy() % eval_interval == 0:
                results = metric_utils.eager_compute(
                    eval_metrics,
                    eval_env,
                    eval_policy,
                    num_episodes=num_eval_episodes,
                    train_step=global_step,
                    summary_writer=eval_summary_writer,
                    summary_prefix='Metrics',
                )
                if eval_metrics_callback is not None:
                    eval_metrics_callback(results, global_step.numpy())
                metric_utils.log_metrics(eval_metrics)

        return train_loss
