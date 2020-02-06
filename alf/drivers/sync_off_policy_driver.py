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

import alf.data_structures as ds
from alf.utils import nest_utils


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
        self._prepare_specs(algorithm)

    def _prepare_specs(self, algorithm):
        time_step_spec = algorithm.time_step_spec
        self._training_info_spec = ds.TrainingInfo(
            action=algorithm.action_spec,
            step_type=time_step_spec.step_type,
            reward=time_step_spec.reward,
            discount=time_step_spec.discount,
            info=algorithm.rollout_info_spec,
            env_id=time_step_spec.env_id)

    def _run(self, max_num_steps, time_step, policy_state):
        """Take steps in the environment for max_num_steps."""
        return self.rollout(max_num_steps, time_step, policy_state)

    @tf.function
    def rollout(self, max_num_steps, time_step, policy_state):
        counter = tf.zeros((), tf.int32)
        batch_size = self._env.batch_size
        maximum_iterations = math.ceil(max_num_steps / self._env.batch_size)

        def create_ta(s):
            return tf.TensorArray(
                dtype=s.dtype,
                size=maximum_iterations,
                element_shape=tf.TensorShape([batch_size]).concatenate(
                    s.shape))

        training_info_ta = tf.nest.map_structure(
            create_ta,
            self._training_info_spec._replace(
                info=nest_utils.to_distribution_param_spec(
                    self._training_info_spec.info)))

        [counter, time_step, policy_state, training_info_ta] = tf.while_loop(
            cond=lambda *_: True,
            body=self._rollout_loop_body,
            loop_vars=[counter, time_step, policy_state, training_info_ta],
            maximum_iterations=maximum_iterations,
            back_prop=False,
            name="rollout_loop")

        training_info = tf.nest.map_structure(lambda ta: ta.stack(),
                                              training_info_ta)

        training_info = nest_utils.params_to_distributions(
            training_info, self._training_info_spec)

        self._algorithm.summarize_rollout(training_info)
        self._algorithm.summarize_metrics()

        return time_step, policy_state

    def _rollout_loop_body(self, counter, time_step, policy_state,
                           training_info_ta):

        next_time_step, policy_step, transformed_time_step = self._step(
            time_step, policy_state)

        training_info = ds.TrainingInfo(
            action=policy_step.action,
            reward=transformed_time_step.reward,
            discount=transformed_time_step.discount,
            step_type=transformed_time_step.step_type,
            info=nest_utils.distributions_to_params(policy_step.info),
            env_id=transformed_time_step.env_id)

        training_info_ta = tf.nest.map_structure(
            lambda ta, x: ta.write(counter, x), training_info_ta,
            training_info)

        counter += 1

        return [counter, next_time_step, policy_step.state, training_info_ta]
