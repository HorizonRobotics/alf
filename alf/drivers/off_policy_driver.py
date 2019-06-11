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

from alf.algorithms.on_policy_algorithm import make_training_info
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
                 replay_buffer=None,
                 training=True,
                 train_interval=20,
                 trainig_batch_size=64,
                 initial_collect_steps=128,
                 collect_steps_per_iteration=1,
                 debug_summaries=False,
                 summarize_grads_and_vars=False,
                 train_step_counter=None):

        if training and replay_buffer is None:
            observers = observers + [replay_buffer.add_batch]
            self._prepare_specs(env, algorithm)
            if replay_buffer is None:
                # use gin_param to set replay_buffer capacity
                replay_buffer = TFUniformReplayBuffer(self._experience_spec,
                                                      env.batch_size)

        super(OffPolicyDriver, self).__init__(
            env, algorithm, observers, metrics, training, debug_summaries,
            summarize_grads_and_vars, train_step_counter)

        self._initial_collect_steps = initial_collect_steps
        self._collect_steps_per_iteration = collect_steps_per_iteration
        self._training_batch_size = trainig_batch_size
        self._replay_buffer = replay_buffer
        self._train_interval = train_interval

    def _prepare_specs(self, env, algorithm):
        def extract_spec(nest):
            return tf.nest.map_structure(
                lambda t: tf.TensorSpec(t.shape[1:], t.dtype), nest)

        time_step_spec = env.time_step_spec()
        policy_step = algorithm.collect_step(self.get_initial_time_step(),
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
            prev_action=ts.action,
            action=ts.action,
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

    # def train(self, max_num_steps, time_step, policy_state):
    #     # collect every iteration
    #     time_step, policy_state = self.predict(
    #         self._initial_collect_steps, time_step, policy_state)

    #     maximum_iterations = math.ceil(
    #         max_num_steps / (self._env.batch_size * self._collect_steps_per_iteration))
    #     [time_step, policy_state] = tf.while_loop(
    #         cond=lambda *_: True,
    #         body=self._iter,
    #         loop_vars=[time_step, policy_state],
    #         maximum_iterations=maximum_iterations,
    #         back_prop=False,
    #         name="driver_loop")

    #     return time_step, policy_state

    def train(self, experience):
        experience = make_time_major(experience)
        batch_size = experience.step_type.shape[1]
        counter = tf.zeros((), tf.int32)
        policy_state = self.get_initial_state()

        def create_ta(s):
            return tf.TensorArray(
                dtype=s.dtype,
                size=self._train_interval,
                element_shape=tf.TensorShape([batch_size]).concatenate(
                    s.shape))

        experience_ta = tf.nest.map_structory(create_ta, self._experience_spec)
        experience_ta = tf.nest.map_structury(
            lambda elem, ta: ta.unstack(elem[0:-1]), experience, experience_ta)
        training_info_ta = tf.nest.map_structure(create_ta,
                                                 self._training_info_spec)

        with tf.GradientTape() as tape:
            [_, policy_state, _, training_info_ta] = tf.while_loop(
                cond=lambda counter, *_: tf.less(counter, self._train_interval),
                body=self._train_loop_body,
                loop_vars=[
                    counter, policy_state, experience_ta, training_info_ta
                ],
                back_prop=True,
                name="iter_loop")
            training_info = tf.nest.map_structure(lambda ta: ta.stack(),
                                                  training_info_ta)

        final_exp = tf.nest.map_structure(lambda t: t[-1], experience)
        policy_state, final_info = self._algorithm.train_step(
            final_exp, policy_state)

        loss_info, grads_and_vars = self._algorithm.train_complete(
            tape=tape,
            training_info=training_info,
            final_time_step=final_exp,
            final_info=final_info)

        self.summary(loss_info, grads_and_vars)

        self._train_step_counter.assign_add(1)

    def _train_loop_body(self, counter, policy_state, experience_ta,
                         training_info_ta):

        exp = tf.nest.map_structure(lambda ta: ta.read(counter), experience_ta)
        policy_state = common.reset_state_if_necessary(
            policy_state, self._initial_state,
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

        return [counter, state, experience_ta, training_info_ta]
