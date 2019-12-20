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

from absl import logging

import tensorflow as tf

from tf_agents.environments.tf_environment import TFEnvironment

from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm, Experience
from alf.drivers import policy_driver


class OffPolicyDriver(policy_driver.PolicyDriver):
    """
    A base class for SyncOffPolicyDriver and AsyncOffPolicyDriver
    """

    def __init__(self,
                 env: TFEnvironment,
                 algorithm: OffPolicyAlgorithm,
                 exp_replayer: str,
                 num_envs=1,
                 observers=[],
                 metrics=[]):
        """Create an OffPolicyDriver.

        Args:
            env (TFEnvironment): A TFEnvironment
            algorithm (OffPolicyAlgorithm): The algorithm for training
            exp_replayer (str): a string that indicates which ExperienceReplayer
                to use. Either "one_time" or "uniform".
            num_envs (int): the number of batched environments. The total number
                of single environment is `num_envs * env.batch_size`
            observers (list[Callable]): An optional list of observers that are
                updated after every step in the environment. Each observer is a
                callable(time_step.Trajectory).
            metrics (list[TFStepMetric]): An optional list of metrics.
        """
        super(OffPolicyDriver, self).__init__(
            env=env,
            algorithm=algorithm,
            observers=observers,
            metrics=metrics,
            mode="rollout")

        algorithm.set_exp_replayer(exp_replayer, num_envs * env.batch_size)

    def start(self):
        """
        Start the driver. Only valid for AsyncOffPolicyDriver.
        This empty function keeps OffPolicyDriver APIs consistent.
        """
        pass

    def stop(self):
        """
        Stop the driver. Only valid for AsyncOffPolicyDriver.
        This empty function keeps OffPolicyDriver APIs consistent.
        """
        pass
