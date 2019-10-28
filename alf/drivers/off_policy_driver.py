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
import tensorflow_probability as tfp

from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import StepType
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.specs.distribution_spec import DistributionSpec
from tf_agents.specs.tensor_spec import TensorSpec

from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm, Experience
from alf.drivers import policy_driver
from alf.utils import common


def warning_once(msg, *args):
    """Generate warning message once

    Args:
        msg: str, the message to be logged.
        *args: The args to be substitued into the msg.
    """
    logging.log_every_n(logging.WARNING, msg, 1 << 62, *args)


class OffPolicyDriver(policy_driver.PolicyDriver):
    """
    A base class for SyncOffPolicyDriver and AsyncOffPolicyDriver
    """

    def __init__(self,
                 env: TFEnvironment,
                 algorithm: OffPolicyAlgorithm,
                 exp_replayer: str,
                 observers=[],
                 use_rollout_state=False,
                 metrics=[]):
        """Create an OffPolicyDriver.

        Args:
            env (TFEnvironment): A TFEnvironment
            algorithm (OffPolicyAlgorithm): The algorithm for training
            exp_replayer (str): a string that indicates which ExperienceReplayer
                to use. Either "one_time" or "uniform".
            observers (list[Callable]): An optional list of observers that are
                updated after every step in the environment. Each observer is a
                callable(time_step.Trajectory).
            metrics (list[TFStepMetric]): An optional list of metrics.
        """
        super(OffPolicyDriver, self).__init__(
            env=env,
            algorithm=algorithm,
            observers=observers,
            use_rollout_state=use_rollout_state,
            metrics=metrics,
            training=True,
            greedy_predict=False)  # always use OnPolicyDriver for play/eval!

        self._prepare_specs(algorithm)
        algorithm.set_exp_replayer(exp_replayer)

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

    def _prepare_specs(self, algorithm):
        """Prepare various tensor specs."""

        time_step = self.get_initial_time_step()
        self._time_step_spec = common.extract_spec(time_step)
        self._action_spec = self._env.action_spec()

        policy_step = algorithm.rollout(
            algorithm.transform_timestep(time_step), self._initial_state)
        info_spec = common.extract_spec(policy_step.info)
        self._policy_step_spec = PolicyStep(
            action=self._action_spec,
            state=algorithm.train_state_spec,
            info=info_spec)

        def _to_distribution_spec(spec):
            if isinstance(spec, TensorSpec):
                return DistributionSpec(
                    tfp.distributions.Deterministic,
                    input_params_spec={"loc": spec},
                    sample_spec=spec)
            return spec

        self._action_distribution_spec = tf.nest.map_structure(
            _to_distribution_spec, algorithm.action_distribution_spec)
        self._action_dist_param_spec = tf.nest.map_structure(
            lambda spec: spec.input_params_spec,
            self._action_distribution_spec)

        algorithm.prepare_off_policy_specs(time_step)
