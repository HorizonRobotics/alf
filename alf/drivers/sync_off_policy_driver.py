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
"""Synchronous driver for off-policy training."""

import math

from absl import logging
import gin.tf
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.environments.tf_environment import TFEnvironment
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.drivers.off_policy_driver import OffPolicyDriver
from alf.experience_replayers.experience_replay import SyncUniformExperienceReplayer


@gin.configurable
class SyncOffPolicyDriver(OffPolicyDriver):
    """Synchronous driver for off-policy training.

    It provides two major interface functions.

    * run(): for collecting data into replay buffer using algorithm.predict
    * train(): for training with one batch of data.

    train() further divides a batch into multiple minibatches. For each
    mini-batch. It performs the following computation:
    ```python
        with tf.GradientTape() as tape:
            batched_training_info
            for experience in batch:
                policy_step = train_step(experience, state)
                collect necessary information and policy_step.info into
                training_info
            train_complete(tape, training_info)
    ```
    """

    def __init__(self,
                 env: TFEnvironment,
                 algorithm: OffPolicyAlgorithm,
                 exp_replayer="uniform",
                 observers=[],
                 metrics=[]):
        """Create an OffPolicyDriver.

        Args:
            env (TFEnvironment): A TFEnvironmnet
            algorithm (OffPolicyAlgorithm): The algorithm for training
            exp_replayer (str): a string that indicates which ExperienceReplayer
                to use.
            observers (list[Callable]): An optional list of observers that are
                updated after every step in the environment. Each observer is a
                callable(time_step.Trajectory).
            metrics (list[TFStepMetric]): An optiotional list of metrics.
        """
        # training=False because training info is always obtained from
        # replayed exps instead of current time_step prediction. So _step() in
        # policy_driver.py has nothing to do with training for off-policy
        # algorithms
        super(SyncOffPolicyDriver, self).__init__(
            env=env,
            algorithm=algorithm,
            exp_replayer=exp_replayer,
            observers=observers,
            metrics=metrics)
        algorithm.set_metrics(self.get_metrics())

    def _run(self, max_num_steps, time_step, policy_state):
        """Take steps in the environment for max_num_steps."""
        return self.predict(max_num_steps, time_step, policy_state)
