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

import time
import random
import unittest
from absl.testing import parameterized

from absl import logging
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.nest as nest
import threading

from tf_agents.trajectories.time_step import TimeStep, StepType
from tf_agents.trajectories.policy_step import PolicyStep
from alf.algorithms.rl_algorithm import make_action_time_step
from alf.drivers.threads import NestFIFOQueue
from alf.drivers.threads import flatten_once
from alf.drivers.threads import TFQueues, ActorThread, EnvThread
from alf.drivers.threads import repeat_shape_n
from alf.drivers.off_policy_async_driver import OffPolicyAsyncDriver
import collections


class Env(object):
    def __init__(self):
        self._step_type = StepType.FIRST
        self._episode_length = 0

    @property
    def batch_size(self):
        return 1

    def current_time_step(self):
        return TimeStep(step_type=tf.ones((1,), tf.int32) * self._step_type,
                        reward=tf.ones((1,)) * (self._step_type == StepType.LAST),
                        discount=tf.ones((1,)),
                        observation=tf.ones([1, 10]))

    def action_spec(self):
        return tf.TensorSpec(shape=[2], dtype=tf.float32)

    def time_step_spec(self):
        return tf.nest.map_structure(
            lambda t: tf.TensorSpec(t.shape[1:], t.dtype),
            self.current_time_step()
        )

    def step(self, action):
        self._episode_length += 1
        if self._step_type == StepType.LAST:
            self._step_type = StepType.FIRST
            self._episode_length = 0
        else:
            if self._episode_length == 5:
                self._step_type = StepType.LAST
            else:
                self._step_type = StepType.MID
        return self.current_time_step()


class Algorithm(object):
    def predict(self, time_step, state):
        return PolicyStep(action=tfp.distributions.Deterministic(
                            loc=tf.zeros([time_step.observation.shape[0], 2])),
                          state=state,
                          info=())

    @property
    def predict_state_spec(self):
        return ()

    @property
    def trainable_variables(self):
        return

    @property
    def action_distribution_spec(self):
        return tf.TensorSpec(shape=[2], dtype=tf.float32)


class OffPolicyAsyncDriverTest(parameterized.TestCase,
                               unittest.TestCase):

    def test_nest_fifo(self):
        NamedTuple = collections.namedtuple('tuple', 'x y')
        t0 = NamedTuple(x=tf.ones([2, 3]),
                        y=tf.ones([2]))
        # test whether any field can be an empty tuple
        t = NamedTuple(x=t0, y=())

        queue = NestFIFOQueue(capacity=2, sample_element=t)

        # assert that we can safely enqueue/dequeue a tf.nest structure
        queue.enqueue(t)
        tf.nest.assert_same_structure(t, queue.dequeue())

        # assert that we can safely dequeue multiple tf.nest structures
        queue.enqueue(t)
        queue.enqueue(t)
        t2 = nest.map_structure(lambda x, y: tf.stack([x, y], axis=0), t, t)
        tf.nest.assert_same_structure(queue.dequeue_many(2), t2)

    def test_nest_pack_and_unpack(self):
        NamedTuple = collections.namedtuple('tuple', 'x y')
        t0 = NamedTuple(x=tf.ones([2, 3]),
                        y=tf.ones([2, 10]))
        t1 = NamedTuple(x=t0, y=tf.zeros([2, 3, 2]))
        nested = NamedTuple(x=t0, y=t1)
        nested_ = nest.map_structure(flatten_once, nested)
        nested_ = nest.map_structure(
            lambda e: tf.reshape(e, [2, -1] + list(e.shape[1:])),
            nested_)
        nest.map_structure(
            lambda e1, e2: self.assertTrue(tf.reduce_all(tf.equal(e1, e2))),
            nested, nested_)

    @parameterized.parameters(
        (50, 20, 5, 10, 5, 30, 20)
    )
    def test_alf_metrics(self,
                         num_envs,
                         learn_queue_cap,
                         act_queue_cap,
                         unroll_length,
                         num_actors,
                         max_num_env_steps,
                         max_num_iterations):
        alg = Algorithm()
        driver = OffPolicyAsyncDriver(lambda: Env(),
                                      alg,
                                      num_envs,
                                      num_actors,
                                      max_num_iterations,
                                      max_num_env_steps,
                                      unroll_length,
                                      learn_queue_cap,
                                      act_queue_cap,
                                      training=False)
        driver.run()

        total_num_steps = int(driver._log_thread.metrics[1].result())
        self.assertGreaterEqual(num_envs * max_num_env_steps, total_num_steps)
        self.assertGreaterEqual(
            total_num_steps, # multiply by 2/3 because 1/3 of steps are StepType.LAST
            int(max_num_env_steps // unroll_length * 0.9) * unroll_length * num_envs * 2//3)
        average_reward = int(driver._log_thread.metrics[2].result())
        self.assertEqual(average_reward, 1)
        episode_length = int(driver._log_thread.metrics[3].result())
        self.assertEqual(episode_length, 5)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    from alf.utils.common import set_per_process_memory_growth

    set_per_process_memory_growth()
    tf.config.experimental_run_functions_eagerly(True)
    unittest.main()
