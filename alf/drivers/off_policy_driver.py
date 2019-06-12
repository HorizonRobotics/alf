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
import gin.tf
import tensorflow as tf

from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.trajectories import trajectory, policy_step
from tf_agents.trajectories.time_step import StepType
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer

from alf.algorithms.rl_algorithm import make_training_info
from alf.drivers import policy_driver
from alf.utils import common
from alf.algorithms.off_policy_algorithm import Experience


@gin.configurable
class OffPolicyDriver(policy_driver.PolicyDriver):
    def __init__(self,
                 env: TFEnvironment,
                 algorithm,
                 observers=[],
                 metrics=[],
                 debug_summaries=False,
                 summarize_grads_and_vars=False,
                 train_step_counter=None):
        """Create an OffPolicyDriver.

        Args:
            env (TFEnvironment): A TFEnvoronmnet
            algorithm (OnPolicyAlgorith): The algorithm for training
            observers (list[Callable]): An optional list of observers that are
                updated after every step in the environment. Each observer is a
                callable(time_step.Trajectory).
            metrics (list[TFStepMetric]): An optiotional list of metrics.
            debug_summaries (bool): A bool to gather debug summaries.
            summarize_grads_and_vars (bool): If True, gradient and network
                variable summaries will be written during training.
            train_step_counter (tf.Variable): An optional counter to increment
                every time the a new iteration is started. If None, it will use 
                tf.summary.experimental.get_step(). If this is still None, a
                counter will be created.
        """
        super(OffPolicyDriver, self).__init__(
            env=env,
            algorithm=algorithm,
            observers=observers,
            metrics=metrics,
            training=False,
            greedy_predict=False,
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

        self._experience_spec = Experience(
            step_type=time_step_spec.step_type,
            reward=time_step_spec.reward,
            discount=time_step_spec.discount,
            observation=time_step_spec.observation,
            prev_action=env.action_spec(),
            action=env.action_spec(),
            info=collect_info_spec)

        ts = self.get_initial_time_step()
        exp = Experience(
            step_type=ts.step_type,
            reward=ts.reward,
            discount=ts.discount,
            observation=ts.observation,
            prev_action=ts.prev_action,
            action=ts.prev_action,
            info=policy_step.info)

        _, info = algorithm.train_step(exp, self.get_initial_train_state(3))
        info_spec = extract_spec(info)
        self._training_info_spec = make_training_info(
            action=env.action_spec(),
            step_type=time_step_spec.step_type,
            reward=time_step_spec.reward,
            discount=time_step_spec.discount,
            info=info_spec,
            collect_info=self._experience_spec.info)

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

    def train(self, experience: Experience):
        """Train using `experience`

        Args:
            experience (Experience): experience from replay_buffer. It is
                assumed to be batch major.
        """
        experience = self._make_time_major(experience)
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
            policy_state = common.reset_state_if_necessary(
                policy_state, initial_train_state,
                tf.equal(exp.step_type, StepType.FIRST))
            state, info = self._algorithm.train_step(exp, state=policy_state)

            training_info = make_training_info(
                action=exp.action,
                reward=exp.reward,
                discount=exp.discount,
                step_type=exp.step_type,
                info=info,
                collect_info=exp.info)

            training_info_ta = tf.nest.map_structure(
                lambda ta, x: ta.write(counter, x), training_info_ta,
                training_info)

            counter += 1

            return [counter, state, training_info_ta]

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

        final_exp = tf.nest.map_structure(lambda x: x[-1], experience)
        policy_state, final_info = self._algorithm.train_step(
            final_exp, policy_state)

        loss_info, grads_and_vars = self._algorithm.train_complete(
            tape=tape,
            training_info=training_info,
            final_time_step=final_exp,
            final_info=final_info)

        del tape

        self.summary(loss_info, grads_and_vars)

        self._train_step_counter.assign_add(1)
