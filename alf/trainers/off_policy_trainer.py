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

import math
import os
import sys
import time
from typing import Callable

from absl import logging
import gin.tf
import tensorflow as tf

from tf_agents.eval import metric_utils
from tf_agents.utils import common as tfa_common

from alf.drivers.async_off_policy_driver import AsyncOffPolicyDriver
from alf.drivers.sync_off_policy_driver import SyncOffPolicyDriver
from alf.utils.metric_utils import eager_compute
from tf_agents.metrics import tf_metrics
from alf.utils import common
from alf.utils.common import run_under_record_context, get_global_counter


@gin.configurable
def train(root_dir,
          env_f: Callable,
          algorithm,
          synchronous=True,
          eval_env=None,
          random_seed=0,
          initial_collect_steps=0,
          num_updates_per_train_step=4,
          unroll_length=8,
          mini_batch_length=20,
          mini_batch_size=256,
          clear_replay_buffer=True,
          num_iterations=1000,
          use_tf_functions=True,
          summary_interval=50,
          summaries_flush_secs=1,
          summary_max_queue=10,
          eval_interval=10,
          num_eval_episodes=10,
          checkpoint_interval=1000,
          debug_summaries=False,
          summarize_grads_and_vars=False):
    """Perform on-policy training using OnPolicyDriver.

    NOTE: currently, for use_tf_function=False, all the summary names have an
    additional prefix "driver_loop", it's might be a bug of tf2. We'll see.

    Args:
        root_dir (str): directory for saving summary and checkpoints
        env_f (Callable[TFEnvironment]): creates an environment for training
        algorithm (OnPolicyAlgorithm): the training algorithm
        synchronous (bool): whether use the synchronous driver or asynchronous
            driver. For their data pipeline differences, please refer to the
            docstring of AsyncOffPolicyDriver. Generally, you may use the
            synchronous driver for both on-policy (e.g., A2C) and off-policy
            (e.g., PPO) algorithms, and the rollout always uses the most up-to-date
            policy. However, the asynchronous driver always has some policy lag
            between the training policy and the rollout policy. It's used to
            potentially squeeze the waiting time before every two training updates
            and yield higher data throughput. Example algorithms: DQN, off-policy
            AC, PPO. If set to False, remember to set the arguments `num_envs`,
            `num_actor_queues`, `actor_queue_cap`, and `learn_queue_cap` for
            AsyncOffPolicyDriver.
        eval_env (TFEnvironment): environment for evaluating
        initial_collect_steps (int): if positive, number of steps each single
            environment steps before perform first update
        random_seed (int): random seed
        num_updates_per_train_step (int): number of optimization steps for
            one iteration
        unroll_length (int): number of time steps each environment proceeds per
            iteration. The total number of time steps from all environments per
            iteration can be computed as: `num_envs` * `env_batch_size`
            * `unroll_length`.
        mini_batch_size (int): number of sequences for each minibatch
        clear_replay_buffer (bool): whether use all data in replay buffer to
            perform one update and then wiped clean
        mini_batch_length (int): the length of the sequence for each
            sample in the minibatch

        num_iterations (int): number of update iterations
        use_tf_functions (bool): whether to use tf.function
        summary_interval (int): write summary every so many training steps (
            i.e. number of parameter updates)
        summaries_flush_secs (int): flush summary to disk every so many seconds.
        summary_max_queue (int): flush to disk every so mary summaries
        eval_interval (int): evaluate every so many iteration
        num_eval_episodes (int) : number of episodes for one evaluation
        checkpoint_interval (int): checkpoint every so many iterations
        debug_summaries (bool): A bool to gather debug summaries.
        summarize_grads_and_vars (bool): If True, gradient and network variable
            summaries will be written during training.
    """

    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')
    env = env_f()
    # make sure the length of samples from rollout can be divided by
    # `mini_batch_length`
    if clear_replay_buffer:
        unroll_length = math.ceil(
            unroll_length / mini_batch_length) * mini_batch_length

    eval_metrics = None
    eval_summary_writer = None
    if eval_env is not None:
        eval_metrics = [
            tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
            tf_metrics.AverageEpisodeLengthMetric(
                buffer_size=num_eval_episodes)
        ]
        eval_summary_writer = tf.summary.create_file_writer(
            eval_dir, flush_millis=summaries_flush_secs * 1000)

    def train_():
        tf.random.set_seed(random_seed)
        global_step = get_global_counter()

        if synchronous:
            driver = SyncOffPolicyDriver(
                env=env,
                algorithm=algorithm,
                debug_summaries=debug_summaries,
                summarize_grads_and_vars=summarize_grads_and_vars)
        else:
            driver = AsyncOffPolicyDriver(
                env_f=env_f,
                algorithm=algorithm,
                unroll_length=unroll_length,
                debug_summaries=debug_summaries,
                summarize_grads_and_vars=summarize_grads_and_vars)
        replayer = driver.exp_replayer

        checkpointer = tfa_common.Checkpointer(
            ckpt_dir=os.path.join(train_dir, 'algorithm'),
            algorithm=algorithm,
            metrics=metric_utils.MetricsGroup(driver.get_metrics(), 'metrics'),
            global_step=global_step)
        checkpointer.initialize_or_restore()

        if not use_tf_functions:
            tf.config.experimental_run_functions_eagerly(True)

        env.reset()
        driver.start()
        time_step = driver.get_initial_time_step()
        policy_state = driver.get_initial_policy_state()

        for iter in range(num_iterations):
            t0 = tf.timestamp()

            steps = 0
            while True:
                if synchronous:
                    max_num_steps = unroll_length * env.batch_size
                    time_step, policy_state = driver.run(
                        max_num_steps=max_num_steps,
                        time_step=time_step,
                        policy_state=policy_state)
                    steps += max_num_steps
                else:
                    steps += driver.run_async()
                if iter > 0 or steps >= initial_collect_steps:
                    break

            if clear_replay_buffer:
                experience = replayer.replay_all()
                replayer.clear()
            else:
                experience, _ = replayer.replay(
                    sample_batch_size=mini_batch_size,
                    mini_batch_length=mini_batch_length)

            t1 = tf.timestamp()
            driver.train(
                experience,
                num_updates=num_updates_per_train_step,
                mini_batch_length=mini_batch_length,
                mini_batch_size=mini_batch_size)

            logging.info('%s time=%.3f' % (iter, tf.timestamp() - t0))
            tf.summary.scalar(
                name='time/driver.train()', data=tf.timestamp() - t1)

            if (iter + 1) % checkpoint_interval == 0:
                checkpointer.save(global_step=global_step.numpy())

            if eval_env is not None and (iter + 1) % eval_interval == 0:
                with tf.summary.record_if(True):
                    eager_compute(
                        metrics=eval_metrics,
                        environment=eval_env,
                        state_spec=algorithm.predict_state_spec,
                        action_fn=lambda time_step, state: common.
                        algorithm_step(
                            algorithm=driver.algorithm,
                            ob_transformer=driver.observation_transformer,
                            time_step=time_step,
                            state=state,
                            greedy_predict=True,
                            training=False),
                        num_episodes=num_eval_episodes,
                        step_metrics=driver.get_step_metrics(),
                        train_step=global_step,
                        summary_writer=eval_summary_writer,
                        summary_prefix="Metrics")
                    metric_utils.log_metrics(eval_metrics)
            if iter == 0:
                with tf.summary.record_if(True):
                    common.summarize_gin_config()
                    tf.summary.text('commandline', ' '.join(sys.argv))

        driver.stop()

        checkpointer.save(global_step=global_step.numpy())
        env.pyenv.close()
        if eval_env:
            eval_env.pyenv.close()

    run_under_record_context(
        func=train_,
        summary_dir=train_dir,
        summary_interval=summary_interval,
        flush_millis=summaries_flush_secs * 1000,
        summary_max_queue=summary_max_queue)
