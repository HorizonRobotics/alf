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

import os
import time

from absl import logging
import gin.tf
import tensorflow as tf

from tf_agents.eval import metric_utils
from tf_agents.utils import common as tfa_common

from alf.drivers.on_policy_driver import OnPolicyDriver
from alf.utils.common import run_under_record_context


@gin.configurable
def train(train_dir,
          env,
          algorithm,
          random_seed=0,
          train_interval=20,
          num_steps_per_iter=10000,
          num_iterations=1000,
          use_tf_functions=True,
          summary_interval=50,
          summaries_flush_secs=1,
          checkpoint_interval=1000,
          debug_summaries=False,
          summarize_grads_and_vars=False):
    """Perform on-policy training using OnPolicyDriver.

    NOTE: currently, for use_tf_function=False, all the summary names have an
    additional prefix "driver_loop", it's might be a bug of tf2. We'll see.

    Args:
        env (TFEnvironment): the environment
        algorithm (OnPolicyAlgorithm): the training algorithm
        random_seed (int): random seed
        train_interval (int): update parameter every so many env.step().
        num_steps_per_iter (int): number of steps for one iteration. It is the 
            total steps from all individual environment in the batch
            environment.
        use_tf_functions (bool): whether to use tf.function
        summary_interval (int): write summary every so many training steps (
            i.e. number of parameter updates)
        summaries_flush_secs (int): flush summary to disk every so many seconds.
        checkpoint_interval (int): checkpoint every so many iterations
        debug_summaries (bool): A bool to gather debug summaries.
        summarize_grads_and_vars (bool): If True, gradient and network variable
            summaries will be written during training.
    """

    train_dir = os.path.expanduser(train_dir)

    def train_():
        tf.random.set_seed(random_seed)
        global_step = tf.summary.experimental.get_step()

        driver = OnPolicyDriver(
            env=env,
            algorithm=algorithm,
            train_interval=train_interval,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars)

        checkpointer = tfa_common.Checkpointer(
            ckpt_dir=os.path.join(train_dir, 'algorithm'),
            algorithm=algorithm,
            metrics=metric_utils.MetricsGroup(driver.get_metrics(), 'metrics'),
            global_step=global_step)
        checkpointer.initialize_or_restore()

        if use_tf_functions:
            driver.run = tf.function(driver.run)

        env.reset()
        time_step = driver.get_initial_time_step()
        policy_state = driver.get_initial_state()
        for iter in range(num_iterations):
            t0 = time.time()
            time_step, policy_state = driver.run(
                max_num_steps=num_steps_per_iter,
                time_step=time_step,
                policy_state=policy_state)

            logging.info('%s time=%.3f' % (iter, time.time() - t0))

            if (iter + 1) % checkpoint_interval == 0:
                checkpointer.save(global_step=global_step.numpy())

        checkpointer.save(global_step=global_step.numpy())

    run_under_record_context(
        func=train_,
        summary_dir=train_dir,
        summary_interval=summary_interval,
        flush_millis=summaries_flush_secs * 1000)
