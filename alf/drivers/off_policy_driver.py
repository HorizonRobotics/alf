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

import os
import psutil
from typing import Callable
from absl import logging
from collections import namedtuple

from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.utils import common
from alf.algorithms.off_policy_algorithm import Experience, make_experience

import gin.tf
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.trajectories.trajectory import from_transition
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import StepType
from tf_agents.metrics import tf_metrics
from tf_agents.utils import eager_utils
from tf_agents.specs.distribution_spec import DistributionSpec
from tf_agents.specs.distribution_spec import nested_distributions_from_specs

from alf.drivers.policy_driver import PolicyDriver
from alf.drivers.threads import TFQueues, ActorThread, EnvThread, LogThread
from alf.drivers.threads import get_act_dist_param
from alf.drivers.experience_replay import OnetimeExperienceReplayer

from alf.algorithms.rl_algorithm import make_training_info


def transpose2(x, dim1, dim2):
    perm = list(range(len(x.shape)))
    perm[dim1] = dim2
    perm[dim2] = dim1
    return tf.transpose(x, perm)


LearningBatch = namedtuple(
    "LearningBatch",
    "time_step policy_step act_dist_param next_time_step env_id")


def make_learning_batch(time_step, policy_step, act_dist_param, next_time_step,
                        env_id):
    return LearningBatch(
        time_step=time_step,
        policy_step=policy_step,
        act_dist_param=act_dist_param,
        next_time_step=next_time_step,
        env_id=env_id)


@gin.configurable
class OffPolicyDriver(PolicyDriver):
    """
    A driver that enables async training. The definition of 'async' is that prediction
    and learning are not intervened like below:
        pred -> learn -> pred -> learn -> ...
    Instead they are decoupled:
        pred -> pred -> pred -> ...
                   |
            (sync periodically)
                   |
        learn -> learn -> learn -> ...
    And more importantly, a learner or predictor may only operate on a subset of
    environments at a time.
    """

    def __init__(self,
                 env_f: Callable,
                 algorithm: OffPolicyAlgorithm,
                 num_envs=32,
                 num_actor_queues=4,
                 unroll_length=8,
                 learn_queue_cap=8,
                 actor_queue_cap=8,
                 greedy_predict=False,
                 observers=[],
                 metrics=[],
                 exp_replayer_class=OnetimeExperienceReplayer,
                 debug_summaries=False,
                 summarize_grads_and_vars=False,
                 train_step_counter=None):
        """
        Args:
            env_f (Callable): a function with 0 args that creates an environment
            algorithm (OffPolicyAlgorithm):
            num_envs (int): the number of environments to run asynchronously. Note:
                    each environment itself could be a tf_agent batched environment.
                    So the actual total number of environments is `num_envs` * `env_f().batch_size`.
                    However, `env_f().batch_size` is transparent to this driver. So all
                    the queues operate on the assumption of `num_envs` environments.
                    Each environment is exclusively owned by only one `EnvThread`.
            num_actor_queues (int): number of actor queues. Each queue is exclusivly owned by just one actor
                            thread.
            unroll_length (int): number of time steps each environment proceeds before sending the steps
                            to the learner queue
            learn_queue_cap (int): the learner queue capacity determines how many environments contribute
                            to the training data for each training iteration
            actor_queue_cap (int): the actor queue capacity determines how many environments contribute to
                            the data for each prediction forward in an `ActorThread`. To prevent deadlock,
                            it's required that `actor_queue_cap` * `num_actor_queues` <= `num_envs`.
            greedy_predict (bool): used by the algorithm for prediction
            observers (list[Callable]): An optional list of observers that are
                                        updated after every step in the environment. Each observer
                                        is a callable(time_step.Trajectory).
            metrics (list[TFStepMetric]): An optiotional list of metrics.
            exp_replayer_class (ExperienceReplayer): a class that implements how to storie and replay
                                experiences. See `OnetimeExperienceRelayer` for example. The replayed
                                experiences are used for parameter updating.
            debug_summaries (bool): A bool to gather debug summaries.
            summarize_grads_and_vars (bool): If True, gradient and network
                                            variable summaries will be written during training.
            train_step_counter (tf.Variable): An optional counter to increment every time the a new
                                            iteration is started. If None, it will use
                                            tf.summary.experimental.get_step(). If this is still None,
                                            a counter will be created.
        """
        super(OffPolicyDriver, self).__init__(
            env=env_f(),
            algorithm=algorithm,
            observers=[],
            training=False,
            greedy_predict=greedy_predict,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter)
        self._coord = tf.train.Coordinator()
        self._exp_replayer = exp_replayer_class()
        self._prepare_specs(algorithm)
        self._trainable_variables = algorithm.trainable_variables

        # create actor and env threads
        self._tfq = TFQueues(
            num_envs,
            self._env.batch_size,
            learn_queue_cap,
            actor_queue_cap,
            time_step_spec=self._time_step_spec,
            policy_step_spec=self._pred_policy_step_spec,
            act_dist_param_spec=self._action_dist_param_spec,
            unroll_length=unroll_length,
            num_actor_queues=num_actor_queues)
        actor_threads = [
            ActorThread(
                name="actor{}".format(i),
                coord=self._coord,
                algorithm=self._algorithm,
                tf_queues=self._tfq,
                id=i,
                greedy_predict=greedy_predict,
                observation_transformer=self._observation_transformer)
            for i in range(num_actor_queues)
        ]
        env_threads = [
            EnvThread(
                name="env{}".format(i),
                coord=self._coord,
                env_f=env_f,
                tf_queues=self._tfq,
                unroll_length=unroll_length,
                id=i,
                actor_id=i % num_actor_queues) for i in range(num_envs)
        ]
        self._log_thread = LogThread(
            name="logging",
            num_envs=num_envs,
            env_batch_size=self._env.batch_size,
            observers=observers,
            metrics=metrics,
            coord=self._coord,
            queue=self._tfq.log_queue)
        self._threads = actor_threads + env_threads + [self._log_thread]

    def _prepare_specs(self, algorithm):
        """Prepare various tensor specs."""

        def extract_spec(nest):
            return tf.nest.map_structure(
                lambda t: tf.TensorSpec(t.shape[1:], t.dtype), nest)

        time_step = common.get_initial_time_step(self._env)
        self._time_step_spec = extract_spec(time_step)
        self._action_spec = self._env.action_spec()

        policy_step = algorithm.predict(
            time_step,
            common.get_initial_policy_state(self._env.batch_size,
                                            algorithm.predict_state_spec))
        info_spec = extract_spec(policy_step.info)
        self._pred_policy_step_spec = PolicyStep(
            action=self._action_spec,
            state=algorithm.predict_state_spec,
            info=info_spec)

        def _to_distribution_spec(spec):
            if isinstance(spec, tf.TensorSpec):
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

        self._experience_spec = Experience(
            step_type=self._time_step_spec.step_type,
            reward=self._time_step_spec.reward,
            discount=self._time_step_spec.discount,
            observation=self._time_step_spec.observation,
            prev_action=self._action_spec,
            action=self._action_spec,
            info=info_spec,
            action_distribution=self._action_dist_param_spec)

        action_dist_params = common.zero_tensor_from_nested_spec(
            self._experience_spec.action_distribution, self._env.batch_size)
        action_dist = nested_distributions_from_specs(
            self._action_distribution_spec, action_dist_params)
        exp = Experience(
            step_type=time_step.step_type,
            reward=time_step.reward,
            discount=time_step.discount,
            observation=time_step.observation,
            prev_action=time_step.prev_action,
            action=time_step.prev_action,
            info=policy_step.info,
            action_distribution=action_dist)

        processed_exp = algorithm.preprocess_experience(exp)
        self._processed_experience_spec = self._experience_spec._replace(
            info=extract_spec(processed_exp.info))

        policy_step = common.algorithm_step(
            algorithm,
            ob_transformer=None,
            time_step=exp,
            state=common.get_initial_policy_state(self._env.batch_size,
                                                  algorithm.train_state_spec),
            training=True)
        info_spec = extract_spec(policy_step.info)
        self._training_info_spec = make_training_info(
            action=self._action_spec,
            action_distribution=self._action_dist_param_spec,
            step_type=self._time_step_spec.step_type,
            reward=self._time_step_spec.reward,
            discount=self._time_step_spec.discount,
            info=info_spec,
            collect_info=self._processed_experience_spec.info,
            collect_action_distribution=self._action_dist_param_spec)

    @property
    def exp_replayer(self):
        return self._exp_replayer

    def get_step_metrics(self):
        """See PolicyDriver.get_step_metrics()"""
        return self._log_thread.metrics[:2]

    def get_metrics(self):
        """See PolicyDriver.get_metrics()"""
        return self._log_thread.metrics

    def start(self):
        """Starts all env, actor, and log threads."""
        for th in self._threads:
            th.setDaemon(True)
            th.start()
        logging.info("All threads started")

    def run(self):
        """
        Each call of run() will wait for a learning batch to be filled in by the
        env threads.

        Output:
            steps (int): the total number of unrolled steps
        """
        t0 = tf.timestamp()
        batch = make_learning_batch(*self._tfq.learn_queue.dequeue_all())
        common.summarize_time(name="Learning queue", data=tf.timestamp() - t0)
        # convert the batch to the experience format
        exp = make_experience(batch.time_step, batch.policy_step,
                              batch.act_dist_param)
        # make the exp batch major for each environment
        exp = tf.nest.map_structure(lambda e: transpose2(e, 1, 2), exp)
        self._exp_replayer.observe(exp, batch.env_id)

        num_envs, unroll_length, env_batch_size = batch.time_step.observation.shape[:
                                                                                    3]
        return num_envs * unroll_length * env_batch_size

    def stop(self):
        # finishes the entire program
        self._coord.request_stop()
        # Cancel all pending requests (including enqueues and dequeues),
        # so that no thread hangs before calling coord.should_stop()
        self._tfq.close_all()
        self._coord.join(self._threads)
        logging.info("All threads stopped")

    def _make_time_major(self, nest):
        return tf.nest.map_structure(lambda x: transpose2(x, 0, 1), nest)

    @tf.function
    def train(self,
              experience: Experience,
              num_updates=1,
              mini_batch_size=None,
              mini_batch_length=None):
        """Train using `experience`. Copied from off_policy_driver.py.

        Args:
            experience (Experience): experience from replay_buffer. It is
                assumed to be batch major.
            num_updates (int): number of optimization steps
            mini_batch_size (int): number of sequences for each minibatch
            mini_batch_length (int): the length of the sequence for each
                sample in the minibatch
        """

        experience = self._algorithm.preprocess_experience(experience)

        length = experience.step_type.shape[1]
        if mini_batch_length is None:
            mini_batch_length = length

        experience = tf.nest.map_structure(
            lambda x: tf.reshape(x, [-1, mini_batch_length] + list(x.shape[2:])
                                 ), experience)

        if mini_batch_size is None:
            mini_batch_size = experience.step_type.shape[0]

        assert length % mini_batch_length == 0

        batch_size = experience.step_type.shape[0]
        # The reason of this constraint is at L244
        # TODO: remove this constraint.
        assert batch_size % mini_batch_size == 0, (
            "batch_size=%s mini_batch_size=%s" % (batch_size, mini_batch_size))
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
                # Make the shape explicit. The shapes of tensors from the
                # previous line depend on tensor `b`, which is replaced with
                # None by tf. This makes some operations depending on the shape
                # of tensor fail. (Currently, it's alf.common.tensor_extend)
                # TODO: Find a way to work around with shapes containing None
                # at common.tensor_extend()
                batch = tf.nest.map_structure(
                    lambda x: tf.reshape(x, [mini_batch_size] + list(x.shape)[
                        1:]), batch)
                batch = self._make_time_major(batch)
                training_info, loss_info, grads_and_vars = self._update(
                    batch, weight=batch.step_type.shape[0] / mini_batch_size)
                # somehow tf.function autograph does not work correctly for the
                # following code:
                # if u == num_updates - 1 and b + mini_batch_size >= batch_size:
                is_last_mini_batch = tf.logical_and(
                    tf.equal(u, num_updates - 1),
                    tf.greater_equal(b + mini_batch_size, batch_size))
                if is_last_mini_batch:
                    self._training_summary(training_info, loss_info,
                                           grads_and_vars)

        self._train_step_counter.assign_add(1)

    def _update(self, experience, weight):
        batch_size = experience.step_type.shape[1]
        counter = tf.zeros((), tf.int32)
        initial_train_state = common.get_initial_policy_state(
            batch_size, self._algorithm.train_state_spec)
        num_steps = experience.step_type.shape[0]

        def create_ta(s):
            return tf.TensorArray(
                dtype=s.dtype,
                size=num_steps,
                element_shape=tf.TensorShape([batch_size]).concatenate(
                    s.shape))

        experience_ta = tf.nest.map_structure(create_ta,
                                              self._processed_experience_spec)
        experience_ta = tf.nest.map_structure(
            lambda elem, ta: ta.unstack(elem), experience, experience_ta)
        training_info_ta = tf.nest.map_structure(create_ta,
                                                 self._training_info_spec)

        def _train_loop_body(counter, policy_state, training_info_ta):
            exp = tf.nest.map_structure(lambda ta: ta.read(counter),
                                        experience_ta)
            collect_action_distribution_param = exp.action_distribution

            policy_state = common.reset_state_if_necessary(
                policy_state, initial_train_state,
                tf.equal(exp.step_type, StepType.FIRST))

            policy_step = common.algorithm_step(
                self._algorithm,
                self._observation_transformer,
                exp,
                policy_state,
                training=True)
            policy_step, action_dist_param = get_act_dist_param(policy_step)

            training_info = make_training_info(
                action=exp.action,
                action_distribution=action_dist_param,
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
            [_, _, training_info_ta] = tf.while_loop(
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

        loss_info, grads_and_vars = self._algorithm.train_complete(
            tape=tape, training_info=training_info, weight=weight)

        del tape

        return training_info, loss_info, grads_and_vars
