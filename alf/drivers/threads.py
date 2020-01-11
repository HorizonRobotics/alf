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
"""Thread classes for asynchronous training."""

from collections import namedtuple
import threading
import traceback
from threading import Thread

from typing import Callable
import tensorflow as tf
import tensorflow_probability as tfp

from alf.utils import common, nest_utils
from alf.data_structures import make_action_time_step

from tf_agents.trajectories.trajectory import from_transition

from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.metrics.tf_metrics import NumberOfEpisodes
from alf.metrics.tf_metrics import EnvironmentSteps
from alf.metrics.tf_metrics import AverageReturnMetric
from alf.metrics.tf_metrics import AverageEpisodeLengthMetric


class NestFIFOQueue(object):
    """
    The original tf.FIFOQueue doesn't support dequeue_all.
    And it doesn't support enqueue nested structures. So we write a wrapper.
    """

    def __init__(self, capacity, sample_element):
        """
        Args:
            capacity (int): maximum number of elements
            sample_element (tf.nest): an example of elements to be stored in the
                queue. It's just used to infer dtypes and shapes without being
                actually stored.
        """
        dtypes = tf.nest.map_structure(lambda e: e.dtype, sample_element)
        shapes = tf.nest.map_structure(lambda e: e.shape, sample_element)
        flat_dtypes = tf.nest.flatten(dtypes)
        flat_shapes = tf.nest.flatten(shapes)
        # FIFOQueue will strip [val] as a single val when enqueue()
        # what we really want is to preserve this list after dequeue()
        # so we manually unsqueeze in this special case
        self._unsqueeze = (len(flat_dtypes) == 1)
        self._structure = dtypes
        self._queue = tf.queue.FIFOQueue(capacity, flat_dtypes, flat_shapes)
        self._capacity = capacity

    def enqueue(self, vals):
        """
        Enqueue a nested structure into the queue. The structure will first be
        flattened.

        Args:
            vals (nested structure): a single structure with nested tensors
        """
        flat_vals = tf.nest.flatten(vals)
        self._queue.enqueue(flat_vals)

    def dequeue(self):
        """Dequeue an element from the queue.

        The flat element will be packed into the original nested structure.

        Returns:
            vals (nested structure):
        """
        flat_vals = self._queue.dequeue()
        if self._unsqueeze:
            flat_vals = [flat_vals]
        return tf.nest.pack_sequence_as(self._structure, flat_vals)

    def dequeue_many(self, n):
        """Dequeue `n` elements from the queue.

        Returns:
            vals (nested structure): Each item of `vals` will have an additional
            dim at axis=0 because of stacking the `n` elements. See tf.queue for
            more information.
        """
        vals = self._queue.dequeue_many(n)
        return tf.nest.pack_sequence_as(self._structure, vals)

    def dequeue_all(self):
        """Dequeue all elements."""
        return self.dequeue_many(self._capacity)

    def close(self, cancel_pending_enqueues=True):
        """A wrapper for tf.queue.FIFOQueue.close()."""
        self._queue.close(cancel_pending_enqueues)

    def is_closed(self):
        """A wrapper for tf.queue.FIFOQueue.is_closed()."""
        return self._queue.is_closed()

    def size(self):
        """A wrapper for tf.queue.FIFOQueue.size()."""
        return self._queue.size()


def repeat_shape_n(nested_spec, n):
    """
    Repeat `nested`'s shape `n` times along axis=0
    """
    return tf.nest.map_structure(
        lambda t: tf.TensorSpec([n] + list(t.shape), t.dtype), nested_spec)


LearningBatch = namedtuple(
    "LearningBatch", ["time_step", "state", "policy_step", "next_time_step"])


class TFQueues(object):
    """Structure for various queues for async training."""

    def __init__(self,
                 num_envs,
                 env_batch_size,
                 learn_queue_cap,
                 actor_queue_cap,
                 time_step_spec,
                 policy_step_spec,
                 unroll_length,
                 num_actor_queues=1):
        """
        Create five kinds of queues:
        1. one learner queue
            stores batches of training trajectories
            all agent threads should enqueue unrolled trajectories into it
        2.`num_actor_queues` actor queues
            each queue stores batches of observations from some envs to act upon
            all agent threads should enqueue current observations into one of
            the actor queues to get predicted actions
        3. `num_envs` action-returning queues
            each env holds one such queue for receiving the returned action
            predicted the by actor
        4. one log queue
            the logging thread retrieves trajectory data from this queue
        5. `num_envs` env-unroll queues
            there is a one-to-one mapping from a queue to an env. Each queue
            accumulates `unroll_length` time steps before they are used for
            training.

        These queues are used for communications between learner&actor threads
        and actor&logging threads. We manage them in a centralized way to
        facilitate closing.

        Args:
            num_envs (int): number of tf_agents batched environments running in
                parallel. Each environment could be a batch of environments!
            env_batch_size (int): number of envs contained by each batched env
            learn_queue_cap (int): the capacity of the learner queue
            actor_queue_cap (int): the capacity of a actor queue
            time_step_spec (tf.nest): see OffPolicyAsyncDriver._prepare_specs();
                used for creating queues
            policy_step_spec (tf.nest): see OffPolicyAsyncDriver._prepare_specs();
                used for creating queues
            unroll_length (int): how many time steps each environment proceeds
                before training
            num_actor_queues (int): number of actor queues running in parallel
        """
        batch_time_step_spec = repeat_shape_n(time_step_spec, env_batch_size)
        batch_policy_step_spec = repeat_shape_n(
            nest_utils.to_distribution_param_spec(policy_step_spec),
            env_batch_size)
        unrolled_time_step_spec = repeat_shape_n(batch_time_step_spec,
                                                 unroll_length)
        unrolled_policy_step_spec = repeat_shape_n(batch_policy_step_spec,
                                                   unroll_length)

        self._batch_state_spec = batch_policy_step_spec.state

        self.learn_queue = NestFIFOQueue(
            capacity=learn_queue_cap,
            sample_element=LearningBatch(
                time_step=unrolled_time_step_spec,
                state=unrolled_policy_step_spec.state,
                policy_step=unrolled_policy_step_spec,
                next_time_step=unrolled_time_step_spec))

        self.log_queue = NestFIFOQueue(
            capacity=num_envs,
            sample_element=[
                unrolled_time_step_spec, unrolled_policy_step_spec,
                unrolled_time_step_spec,
                tf.ones((), dtype=tf.int32)
            ])

        tf.debugging.assert_greater_equal(
            num_envs,
            num_actor_queues * actor_queue_cap,
            message="not enough environments!")

        self.actor_queues = [
            NestFIFOQueue(
                capacity=actor_queue_cap,
                sample_element=[
                    batch_time_step_spec, batch_policy_step_spec.state,
                    tf.ones((), dtype=tf.int32)
                ]) for i in range(num_actor_queues)
        ]

        self.action_return_queues = [
            NestFIFOQueue(capacity=1, sample_element=batch_policy_step_spec)
            for i in range(num_envs)
        ]

        self.env_unroll_queues = [
            NestFIFOQueue(
                capacity=unroll_length,
                sample_element=LearningBatch(
                    time_step=batch_time_step_spec,
                    state=batch_policy_step_spec.state,
                    policy_step=batch_policy_step_spec,
                    next_time_step=batch_time_step_spec))
            for i in range(num_envs)
        ]

    def close_all(self):
        try:
            self.learn_queue.close()
            self.log_queue.close()
            for aq in self.actor_queues:
                aq.close()
            for arq in self.action_return_queues:
                arq.close()
            for euq in self.env_unroll_queues:
                euq.close()
        except tf.errors.CancelledError:
            # Ignore this because it is typically because those queues have
            # already been closed
            pass
        except tf.errors.OutOfRangeError:
            # Ignore this because it is typically because those queues have
            # already been closed
            pass


class ActorThread(Thread):
    """
    An actor thread is responsible for taking out time steps from its
    corresponding actor queue, calling the algorithm's prediction, and putting
    the results back in a return queue.

    An actor thread will keep running forever until the coordinator requests a
    stop (from another thread).
    """

    def __init__(self, name, coord, algorithm, tf_queues, id):
        """
        Args:
            name (str): the name of the actor thread
            coord (tf.train.Coordinator): coordinate among threads
            algorithm (OffPolicyAlgorithm): for prediction
            tf_queues (TFQueues): for storing all the tf.FIFOQueues for
                communicating between threads
            id (int): thread id
        """
        super().__init__(name=name, target=self._run, args=(coord, algorithm))
        self._tfq = tf_queues
        self._id = id
        self._actor_q = self._tfq.actor_queues[id]

    @tf.function
    def _enqueue_actions(self, policy_step, i, return_q):
        i_policy_step = tf.nest.map_structure(lambda e: e[i], policy_step)
        return_q.enqueue(i_policy_step)

    def _send_actions_back(self, i, env_ids, policy_step):
        """
        Send back the policy step to the i-th id in `env_ids`
        """
        env_id = env_ids[i]
        # the following line has to be in the eager mode because we want
        # to index a python list of non-tensor objects with tf.scalar
        action_return_queue = self._tfq.action_return_queues[env_id]

        self._enqueue_actions(policy_step, i, action_return_queue)
        return [i + 1, env_ids, policy_step]

    @tf.function
    def _dequeue_and_step(self, algorithm):
        time_step, policy_state, env_ids = self._actor_q.dequeue_all()
        # pack
        time_step = tf.nest.map_structure(common.flatten_once, time_step)
        policy_state = tf.nest.map_structure(common.flatten_once, policy_state)

        # prediction forward
        transformed_time_step = algorithm.transform_timestep(time_step)
        policy_step = algorithm.rollout(
            transformed_time_step, policy_state, mode=RLAlgorithm.ROLLOUT)

        # unpack
        policy_step = nest_utils.distributions_to_params(policy_step)
        policy_step = tf.nest.map_structure(
            lambda e: tf.reshape(e, [env_ids.shape[0], -1] + list(e.shape[1:])
                                 ), policy_step)
        return policy_step, env_ids

    def _acting_body(self, algorithm):
        policy_step, env_ids = self._dequeue_and_step(algorithm)
        i = tf.zeros((), tf.int32)
        tf.while_loop(
            cond=lambda *_: True,
            body=self._send_actions_back,
            loop_vars=[i, env_ids, policy_step],
            back_prop=False,
            parallel_iterations=env_ids.shape[0],
            maximum_iterations=env_ids.shape[0])

    def _run(self, coord, algorithm):
        # do not apply tf.function to any code containing coord!
        # (it won't work)
        with coord.stop_on_exception():
            try:
                while not coord.should_stop():
                    self._acting_body(algorithm)
            except tf.errors.CancelledError as e:
                raise e
            except tf.errors.OutOfRangeError as e:
                raise e
            except Exception as e:
                traceback.print_exc()
                raise e

        # Whoever stops first, cancel all pending requests
        # (including enqueues and dequeues),
        # so that no thread hangs before calling coord.should_stop()
        self._tfq.close_all()


class EnvThread(Thread):
    """
    An environment thread

    NOTE: because we potentially have *many* env threads, and
    Python threads share a CPU, so make sure the thread is lightweight
    and IO bound!
    If the env simulation is computation-heavy, consider moving the env
    simulator to an external process
    """

    def __init__(self,
                 name,
                 coord,
                 env,
                 tf_queues,
                 unroll_length,
                 id,
                 actor_id,
                 first_env_id=None):
        """
        Args:
            name (str): name of the thread
            coord (tf.train.Coordinator): coordinate among threads
            env (TFEnvironment):  A TFEnvironment
            tf_queues (TFQueues): an object for storing all the tf.FIFOQueues
                for communicating between threads
            unroll_length (int): Each env unrolls for so many steps before sending
                the steps to the learning queue. If the env is batched, then
                the total number would be `unroll_length` * `batch_size`.
            id (int): an integer identifies the env thread
            first_env_id (int): the id for the first environment of the
                batched environment `env`. If there are multiple `EnvThread`s.
                The `first_env_id` should be set in such way that the IDs for
                individual environments are different. If None, it's assumed
                that all the `env` have same batch_size and first_env_id will
                be set as `id * env.batch_size`
            actor_id (int): indicates which actor thread the env thread should
                send time steps to.
        """
        super().__init__(
            name=name, target=self._run, args=(coord, unroll_length))
        self._env = env
        self._tfq = tf_queues
        self._id = id
        if first_env_id is None:
            first_env_id = self._id * env.batch_size
        self._first_env_id = first_env_id
        self._actor_q = self._tfq.actor_queues[actor_id]
        self._action_return_q = self._tfq.action_return_queues[id]
        self._unroll_queue = self._tfq.env_unroll_queues[id]
        self._initial_policy_state = common.get_initial_policy_state(
            self._env.batch_size,
            tf.nest.map_structure(
                lambda t: tf.TensorSpec(t.shape[1:], t.dtype),
                self._tfq._batch_state_spec))

    def _step(self, time_step, policy_state):
        policy_state = common.reset_state_if_necessary(
            policy_state, self._initial_policy_state, time_step.is_first())
        self._actor_q.enqueue([time_step, policy_state, self._id])
        policy_step = self._action_return_q.dequeue()
        action = policy_step.action
        next_time_step = make_action_time_step(
            self._env.step(action), action, first_env_id=self._first_env_id)
        # temporarily store the transition into a local queue
        self._unroll_queue.enqueue(
            LearningBatch(
                time_step=time_step,
                state=policy_state,
                policy_step=policy_step,
                next_time_step=next_time_step))
        return [next_time_step, policy_step.state]

    def _unroll_env(self, time_step, policy_state, unroll_length):
        time_step, policy_state = tf.while_loop(
            cond=lambda *_: True,
            body=self._step,
            loop_vars=[time_step, policy_state],
            maximum_iterations=unroll_length,
            back_prop=False,
            name="eval_loop")
        return time_step, policy_state

    @tf.function
    def _unroll_and_learn(self, time_step, policy_state, unroll_length):
        time_step, policy_state = self._unroll_env(time_step, policy_state,
                                                   unroll_length)
        # Dump transitions from the local queue and put into
        # the learner queue and the log queue
        unrolled = self._unroll_queue.dequeue_all()
        self._tfq.learn_queue.enqueue(unrolled)
        self._tfq.log_queue.enqueue([
            unrolled.time_step, unrolled.policy_step, unrolled.next_time_step,
            self._id
        ])
        return time_step, policy_state

    def _run(self, coord, unroll_length):
        with coord.stop_on_exception():
            try:
                time_step = common.get_initial_time_step(
                    self._env, first_env_id=self._first_env_id)
                policy_state = self._initial_policy_state
                while not coord.should_stop():
                    time_step, policy_state = self._unroll_and_learn(
                        time_step, policy_state, unroll_length)
            except tf.errors.CancelledError as e:
                raise e
            except tf.errors.OutOfRangeError as e:
                raise e
            except Exception as e:
                traceback.print_exc()
                raise e
        # Whoever stops first, cancel all pending requests
        # (including enqueues and dequeues),
        # so that no thread hangs before calling coord.should_stop()
        self._tfq.close_all()


class LogThread(Thread):
    """
    A logging thread, responsible for summarizing game related metrics
    """

    def __init__(self, name, num_envs, env_batch_size, observers, metrics,
                 coord, queue):
        """
        Args:
            name (str): name of the thread
            num_envs (int): number of env threads
            env_batch_size (int): batch size of each env
            observers (list[Callable]): A list of observers that are
                updated after every step in the environment. Each observer is a
                callable(time_step.Trajectory).
            metrics (list[TFStepMetric]): A list of metrics.
            coord (tf.train.Coordinator): coordinate among threads
            queue (NestFIFOQueue): the queue containing data to be logged
        """
        super().__init__(name=name, target=self._run, args=(coord, queue))
        standard_metrics = [
            NumberOfEpisodes(),
            EnvironmentSteps(),
            AverageReturnMetric(num_envs, env_batch_size),
            AverageEpisodeLengthMetric(num_envs, env_batch_size),
        ]
        self._metrics = standard_metrics + metrics
        self._observers = self._metrics + observers

    @property
    def metrics(self):
        return self._metrics

    @tf.function
    def _summary(self, batch):
        time_step, policy_step, next_time_step, id = batch
        traj = from_transition(time_step, policy_step, next_time_step)
        for ob in self._observers:
            ob(traj, id)

    def _run(self, coord, queue):
        with coord.stop_on_exception():
            try:
                while not coord.should_stop():
                    self._summary(queue.dequeue())
            except tf.errors.CancelledError as e:
                raise e
            except tf.errors.OutOfRangeError as e:
                raise e
            except Exception as e:
                traceback.print_exc()
                raise e
