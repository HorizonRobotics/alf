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
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer

from alf.drivers import policy_driver


@gin.configurable
class OffPolicyDriver(policy_driver.PolicyDriver):
    def __init__(self,
                 env: TFEnvironment,
                 algorithm,
                 observers=[],
                 metrics=[],
                 replay_buffer=None,
                 training=True,
                 trainig_batch_size=64,
                 initial_collect_steps=128,
                 collect_steps_per_iteration=1,
                 debug_summaries=False,
                 summarize_grads_and_vars=False,
                 train_step_counter=None):

        if training and replay_buffer is None:
            time_step_spec = env.time_step_spec()
            policy_step_spec = policy_step.PolicyStep(
                action=env.action_spec(),
                state=(),  # todo
                info=())
            trajectory_spec = trajectory.from_transition(
                time_step_spec,
                policy_step_spec,
                time_step_spec)
            if replay_buffer is None:
                # use gin_param to set replay_buffer capacity
                replay_buffer = TFUniformReplayBuffer(
                    trajectory_spec, env.batch_size)
            observers = observers + [replay_buffer.add_batch]

        super(OffPolicyDriver, self).__init__(
            env,
            algorithm,
            observers,
            metrics,
            training,
            debug_summaries,
            summarize_grads_and_vars,
            train_step_counter)

        self._initial_collect_steps = initial_collect_steps
        self._collect_steps_per_iteration = collect_steps_per_iteration
        self._training_batch_size = trainig_batch_size
        self._replay_buffer = replay_buffer

    def train(self, max_num_steps, time_step, policy_state):
        # collect every iteration
        time_step, policy_state = self.predict(
            self._initial_collect_steps, time_step, policy_state)

        maximum_iterations = math.ceil(
            max_num_steps / (self._env.batch_size * self._collect_steps_per_iteration))
        [time_step, policy_state] = tf.while_loop(
            cond=lambda *_: True,
            body=self._iter,
            loop_vars=[time_step, policy_state],
            maximum_iterations=maximum_iterations,
            back_prop=False,
            name="driver_loop")

        return time_step, policy_state

    def _iter(self, time_step, policy_state):
        [time_step, policy_state] = tf.while_loop(
            cond=lambda *_: True,
            body=self._eval_loop_body,
            loop_vars=[time_step, policy_state],
            maximum_iterations=self._collect_steps_per_iteration,
            back_prop=False,
            name="driver_loop")

        experience, _ = self._replay_buffer.get_next(
            sample_batch_size=self._training_batch_size, num_steps=2)
        loss_info, grads_and_vars = self._algorithm.train_complete(experience)
        self.summary(loss_info, grads_and_vars)
        self._train_step_counter.assign_add(1)
        return time_step, policy_state
