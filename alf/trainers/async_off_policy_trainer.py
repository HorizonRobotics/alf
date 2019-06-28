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
import time

from absl import logging
import gin.tf
import tensorflow as tf

from alf.drivers.off_policy_async_driver import OffPolicyAsyncDriver
from alf.drivers.off_policy_async_driver import ExperienceReplayer
from alf.utils.common import run_under_record_context, get_global_counter


"""
A TEMPORARY async off-policy trainer. To be merged with off_policy_trainer.py.
"""


@gin.configurable
class OnetimeExperienceReplayer(ExperienceReplayer):
    """
    A simple one-time experience replayer. For each incoming `exp`,
    it stores it with a temporary variable which is used for training
    only once.

    Example algorithms: IMPALA, PPO2
    """
    def __init__(self):
        super().__init__()
        self._experience = None

    def observe(self, exp):
        self._experience = exp

    def replay(self):
        return self._experience


@gin.configurable
def train(train_dir,
          env_f,
          algorithm,
          random_seed=0,
          use_tf_functions=True,
          summary_interval=50,
          summaries_flush_secs=1,
          debug_summaries=False):
    """Perform Async off-policy training using OffPolicyAsyncDriver.

    NOTE: currently, for use_tf_function=False, all the summary names have an
    additional prefix "driver_loop", it's might be a bug of tf2. We'll see.

    Args:
        train_dir (str): directory for saving summary and checkpoints
        env (TFEnvironment): environment for training
        algorithm (OnPolicyAlgorithm): the training algorithm
        eval_env (TFEnvironment): environment for evaluating
        random_seed (int): random seed
    """

    train_dir = os.path.expanduser(train_dir)

    def train_():
        tf.random.set_seed(random_seed)

        driver = OffPolicyAsyncDriver(
            env_f=env_f,
            algorithm=algorithm,
            debug_summaries=debug_summaries)

        if not use_tf_functions:
            tf.config.experimental_run_functions_eagerly(True)

        driver.run()

    run_under_record_context(
        func=train_,
        summary_dir=train_dir,
        summary_interval=summary_interval,
        flush_millis=summaries_flush_secs * 1000)
