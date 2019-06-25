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
"""Driver for off-policy training."""

import math

from absl import logging
import gin.tf
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.specs.distribution_spec import DistributionSpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.specs.distribution_spec import nested_distributions_from_specs

from alf.algorithms.rl_algorithm import make_training_info
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.drivers import policy_driver
from alf.utils import common
from alf.algorithms.off_policy_algorithm import Experience


@gin.configurable
class OffPolicyDriver(policy_driver.PolicyDriver):
    """Driver for off-policy training.

    It provides two major interface functions.

    * run(): for collecting data into replay buffer using algorithm.predict
    * train(): for training with one batch of data.

    train() further divides a batch into multiple minibatches. For each mini-
    batch. It performs the following computation:
    ```python
        with tf.GradientTape() as tape:
            batched_training_info
            for experience in batch[:-1]:
                policy_step = train_step(experience, state)
                collect necessary information and policy_step.info into training_info
            final_policy_step = algorithm.train_step(training_info)
            train_complete(tape, training_info, batch[-1], final_policy_step.info)
    ```
    """

    def __init__(self,
                 env: TFEnvironment,
                 algorithm: OffPolicyAlgorithm,
                 observers=[],
                 metrics=[],
                 greedy_predict=False,
                 debug_summaries=False,
                 summarize_grads_and_vars=False,
                 train_step_counter=None):
        """Create an OffPolicyDriver.

        Args:
            env (TFEnvironment): A TFEnvoronmnet
            algorithm (OffPolicyAlgorithm): The algorithm for training
            observers (list[Callable]): An optional list of observers that are
                updated after every step in the environment. Each observer is a
                callable(time_step.Trajectory).
            metrics (list[TFStepMetric]): An optiotional list of metrics.
            greedy_predict (bool): use greedy action for evaluation (i.e.
                training==False).
            debug_summaries (bool): A bool to gather debug summaries.
            summarize_grads_and_vars (bool): If True, gradient and network
                variable summaries will be written during training.
            train_step_counter (tf.Variable): An optional counter to increment
                every time the a new iteration is started. If None, it will use
                tf.summary.experimental.get_step(). If this is still None, a
                counter will be created.
        """
        # training=False because training info is always obtained from
        # replayed exps instead of current time_step prediction. So _step() in
        # policy_driver.py has nothing to do with training for off-policy algorithms
        super(OffPolicyDriver, self).__init__(
            env=env,
            algorithm=algorithm,
            observers=observers,
            metrics=metrics,
            training=False,
            greedy_predict=greedy_predict,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter)

        self._prepare_specs(env, algorithm)
        self._trainable_variables = algorithm.trainable_variables

    def add_replay_buffer(self, replay_buffer=None):
        """Add a replay_buffer.

        Args:
            replay_buffer (ReplayBuffer): if None, a default
                TFUniformReplayBuffer will be created.
        Returns:
            replay_buffer
        """
        if replay_buffer is None:
            # use gin_param to set replay_buffer capacity
            replay_buffer = TFUniformReplayBuffer(self._experience_spec,
                                                  self._env.batch_size)
        self.add_experience_observer(replay_buffer.add_batch)
        return replay_buffer

    def _prepare_specs(self, env, algorithm):
        """Prepare various tensor specs."""

        def extract_spec(nest):
            return tf.nest.map_structure(
                lambda t: tf.TensorSpec(t.shape[1:], t.dtype), nest)

        time_step_spec = env.time_step_spec()
        policy_step = algorithm.predict(self.get_initial_time_step(),
                                        self.get_initial_state())
        collect_info_spec = extract_spec(policy_step.info)

        def _to_distribution_spec(spec):
            if isinstance(spec, tf.TensorSpec):
                return DistributionSpec(
                    tfp.distributions.Deterministic,
                    input_params_spec={"loc": spec},
                    sample_spec=spec)
            return spec

        self._action_distribution_spec = tf.nest.map_structure(
            _to_distribution_spec, algorithm.action_distribution_spec)
        action_dist_param_spec = tf.nest.map_structure(
            lambda spec: spec.input_params_spec,
            self._action_distribution_spec)
        self._experience_spec = Experience(
            step_type=time_step_spec.step_type,
            reward=time_step_spec.reward,
            discount=time_step_spec.discount,
            observation=time_step_spec.observation,
            prev_action=env.action_spec(),
            action=env.action_spec(),
            info=collect_info_spec,
            action_distribution=action_dist_param_spec)

        ts = self.get_initial_time_step()
        action_dist_params = common.zero_tensor_from_nested_spec(
            self._experience_spec.action_distribution, self.env.batch_size)
        action_dist = nested_distributions_from_specs(
            self._action_distribution_spec, action_dist_params)
        exp = Experience(
            step_type=ts.step_type,
            reward=ts.reward,
            discount=ts.discount,
            observation=ts.observation,
            prev_action=ts.prev_action,
            action=ts.prev_action,
            info=policy_step.info,
            action_distribution=action_dist)

        policy_step = self._train_step(exp, self.get_initial_train_state(3))
        info_spec = extract_spec(policy_step.info)
        self._training_info_spec = make_training_info(
            action=env.action_spec(),
            action_distribution=action_dist_param_spec,
            step_type=time_step_spec.step_type,
            reward=time_step_spec.reward,
            discount=time_step_spec.discount,
            info=info_spec,
            collect_info=self._experience_spec.info,
            collect_action_distribution=action_dist_param_spec)

    def get_initial_train_state(self, batch_size):
        return common.zero_tensor_from_nested_spec(
            self._algorithm.train_state_spec, batch_size)

    def _run(self, max_num_steps, time_step, policy_state):
        """Take steps in the environment for max_num_steps."""
        return self.predict(max_num_steps, time_step, policy_state)

    def _make_time_major(self, nest):
        def transpose2(x, dim1, dim2):
            perm = list(range(len(x.shape)))
            perm[dim1] = dim2
            perm[dim2] = dim1
            return tf.transpose(x, perm)

        return tf.nest.map_structure(lambda x: transpose2(x, 0, 1), nest)

    def train(self,
              experience: Experience,
              num_updates=1,
              mini_batch_size=None,
              mini_batch_length=None):
        """Train using `experience`

        Args:
            experience (Experience): experience from replay_buffer. It is
                assumed to be batch major.
            num_updates (int): number of optimization steps
            mini_batch_size (int): number of sequences for each minibatch
            mini_batch_length (int): the length of the sequence for each
                sample in the minibatch
        """
        length = experience.step_type.shape[1]
        if mini_batch_length is None:
            mini_batch_length = length
        if mini_batch_size is None:
            mini_batch_size = experience.step_type.shape[0]

        assert length % mini_batch_length == 0

        experience = tf.nest.map_structure(
            lambda x: tf.reshape(x, [-1, mini_batch_length] + list(x.shape[2:])
                                 ), experience)

        batch_size = experience.step_type.shape[0]
        for u in tf.range(num_updates):
            if mini_batch_size < batch_size:
                indices = tf.random.shuffle(
                    tf.range(experience.step_type.shape[0]))
                experience = tf.nest.map_structure(
                    lambda x: tf.gather(x, indices), experience)
            for b in tf.range(0, batch_size, mini_batch_size):
                batch = tf.nest.map_structure(
                    lambda x: x[b:tf.minimum(batch_size, b + mini_batch_size)],
                    experience)
                batch = self._make_time_major(batch)
                training_info, loss_info, grads_and_vars = self._update(
                    batch, weight=batch.step_type.shape[0] / mini_batch_size)
                # somehow tf.function autograph does not work correctly for the
                # following code:
                # if u == num_updates - 1 and b + mini_batch_size >= batch_size:
                if tf.logical_and(
                        tf.equal(u, num_updates - 1),
                        tf.greater_equal(b + mini_batch_size, batch_size)):
                    self._summary(training_info, loss_info, grads_and_vars)
        self._train_step_counter.assign_add(1)

    def _train_step(self, exp, state):
        policy_step = self.algorithm_step(exp, state=state, training=True)
        return policy_step._replace(
            action=self._to_distribution(policy_step.action))

    def _update(self, experience, weight):
        batch_size = experience.step_type.shape[1]
        counter = tf.zeros((), tf.int32)
        initial_train_state = self.get_initial_train_state(batch_size)
        num_steps = experience.step_type.shape[0] - 1

        def create_ta(s):
            return tf.TensorArray(
                dtype=s.dtype,
                size=num_steps,
                element_shape=tf.TensorShape([batch_size]).concatenate(
                    s.shape))

        experience_ta = tf.nest.map_structure(create_ta, self._experience_spec)
        experience_ta = tf.nest.map_structure(
            lambda elem, ta: ta.unstack(elem[0:-1]), experience, experience_ta)
        training_info_ta = tf.nest.map_structure(create_ta,
                                                 self._training_info_spec)

        def _train_loop_body(counter, policy_state, training_info_ta):
            exp = tf.nest.map_structure(lambda ta: ta.read(counter),
                                        experience_ta)
            collect_action_distribution_param = exp.action_distribution
            collect_action_distribution = nested_distributions_from_specs(
                self._action_distribution_spec,
                collect_action_distribution_param)
            exp = exp._replace(action_distribution=collect_action_distribution)

            policy_state = common.reset_state_if_necessary(
                policy_state, initial_train_state,
                tf.equal(exp.step_type, StepType.FIRST))

            policy_step = self._train_step(exp, state=policy_state)
            action_distribution_param = common.get_distribution_params(
                policy_step.action)

            training_info = make_training_info(
                action=exp.action,
                action_distribution=action_distribution_param,
                reward=exp.reward,
                discount=exp.discount,
                step_type=exp.step_type,
                info=policy_step.info,
                collect_info=exp.info,
                collect_action_distribution=collect_action_distribution_param)

            training_info_ta = tf.nest.map_structure(
                lambda ta, x: ta.write(counter, x), training_info_ta,
                training_info)

            counter += 1

            return [counter, policy_step.state, training_info_ta]

        with tf.GradientTape(
                persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(self._trainable_variables)
            [_, policy_state, training_info_ta] = tf.while_loop(
                cond=lambda counter, *_: tf.less(counter, num_steps),
                body=_train_loop_body,
                loop_vars=[counter, initial_train_state, training_info_ta],
                back_prop=True,
                name="train_loop")
            training_info = tf.nest.map_structure(lambda ta: ta.stack(),
                                                  training_info_ta)
            action_distribution = nested_distributions_from_specs(
                self._action_distribution_spec,
                training_info.action_distribution)
            collect_action_distribution = nested_distributions_from_specs(
                self._action_distribution_spec,
                training_info.collect_action_distribution)
            training_info = training_info._replace(
                action_distribution=action_distribution,
                collect_action_distribution=collect_action_distribution)

        final_exp = tf.nest.map_structure(lambda x: x[-1], experience)
        policy_step = self._train_step(final_exp, policy_state)

        loss_info, grads_and_vars = self._algorithm.train_complete(
            tape=tape,
            training_info=training_info,
            final_time_step=final_exp,
            final_info=policy_step.info,
            weight=weight)

        del tape

        return training_info, loss_info, grads_and_vars
