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

from typing import Callable
from absl import logging

from alf.utils import common
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.off_policy_algorithm import make_experience

import gin.tf
import tensorflow as tf

from tf_agents.environments.tf_environment import TFEnvironment
from alf.drivers.off_policy_driver import OffPolicyDriver
from alf.drivers.threads import TFQueues, ActorThread, EnvThread, LogThread
from alf.experience_replayers.experience_replay import OnetimeExperienceReplayer


@gin.configurable
class AsyncOffPolicyDriver(OffPolicyDriver):
    """
    A driver that enables async training. The definition of 'async' is that
    prediction and learning are not intervened like below (synchronous):
        pred -> learn -> pred -> learn -> ...
    Instead they are decoupled:
        pred -> pred -> pred -> ...
                   |
            (synchronize periodically)
                   |
        learn -> learn -> learn -> ...
    And more importantly, a learner or predictor may only operate on a subset of
    environments at a time.
    """

    def __init__(self,
                 envs,
                 algorithm: OffPolicyAlgorithm,
                 num_actor_queues=1,
                 unroll_length=8,
                 learn_queue_cap=1,
                 actor_queue_cap=1,
                 observers=[],
                 use_rollout_state=False,
                 metrics=[],
                 exp_replayer="one_time"):
        """
        Args:
            envs (list[TFEnvironment]):  list of TFEnvironment
            algorithm (OffPolicyAlgorithm):
            num_actor_queues (int): number of actor queues. Each queue is
                exclusivly owned by just one actor thread.
            unroll_length (int): number of time steps each environment proceeds
                before sending the steps to the learner queue
            learn_queue_cap (int): the learner queue capacity determines how many
                environments contribute to the training data for each training
                iteration
            actor_queue_cap (int): the actor queue capacity determines how many
                environments contribute to the data for each prediction forward
                in an `ActorThread`. To prevent deadlock, it's required that
                `actor_queue_cap` * `num_actor_queues` <= `num_envs`.
            observers (list[Callable]): An optional list of observers that are
                updated after every step in the environment. Each observer is a
                callable(time_step.Trajectory).
            use_rollout_state (bool): Include the RNN state for the experiences
                used for off-policy training
            metrics (list[TFStepMetric]): An optiotional list of metrics.
            exp_replayer (str): a string that indicates which ExperienceReplayer
                to use.
        """
        super(AsyncOffPolicyDriver, self).__init__(
            env=envs[0],
            algorithm=algorithm,
            exp_replayer=exp_replayer,
            observers=observers,
            use_rollout_state=use_rollout_state,
            metrics=metrics)

        # create threads
        self._coord = tf.train.Coordinator()
        num_envs = len(envs)
        self._tfq = TFQueues(
            num_envs,
            self._env.batch_size,
            learn_queue_cap,
            actor_queue_cap,
            time_step_spec=self._time_step_spec,
            policy_step_spec=self._policy_step_spec,
            act_dist_param_spec=self._action_dist_param_spec,
            unroll_length=unroll_length,
            store_state=use_rollout_state,
            num_actor_queues=num_actor_queues)
        actor_threads = [
            ActorThread(
                name="actor{}".format(i),
                coord=self._coord,
                algorithm=self._algorithm,
                tf_queues=self._tfq,
                id=i) for i in range(num_actor_queues)
        ]
        env_threads = [
            EnvThread(
                name="env{}".format(i),
                coord=self._coord,
                env=envs[i],
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
        algorithm.set_metrics(self.get_metrics())

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

    @tf.function
    def get_training_exps(self):
        """
        Get training experiences from the learning queue

        Returns:
            exp (Experience):
            env_id (tf.tensor): if not None, has the shape of (`num_envs`). Each
                element of `env_ids` indicates which batched env the data come from.
            steps (int): how many environment steps this batch of exps contain
        """
        batch = self._tfq.learn_queue.dequeue_all()
        # convert the batch to the experience format
        exp = make_experience(
            batch.time_step,
            batch.policy_step,
            batch.act_dist_param,
            state=batch.state)
        # make the exp batch major for each environment
        exp = tf.nest.map_structure(lambda e: common.transpose2(e, 1, 2), exp)
        num_envs, unroll_length, env_batch_size \
            = batch.time_step.reward.shape[:3]
        steps = num_envs * unroll_length * env_batch_size
        return exp, batch.env_id, steps

    def run_async(self):
        """
        Each call of run_async() will wait for a learning batch to be filled in
        by the env threads.
        Running in the eager mode. The reason is that currently
        OnetimeExperienceReplayer is incompatible with Graph mode because it
        replays by a temporary variable.

        Output:
            steps (int): the total number of unrolled steps
        """
        exp, env_id, steps = self.get_training_exps()
        for ob in self._algorithm.exp_observers:
            ob(exp, env_id)
        return steps

    def _run(self, *args, **kwargs):
        raise RuntimeError(
            "You should call self.run_async instead for async drivers")

    def stop(self):
        # finishes the entire program
        self._coord.request_stop()
        # Cancel all pending requests (including enqueues and dequeues),
        # so that no thread hangs before calling coord.should_stop()
        self._tfq.close_all()
        self._coord.join(self._threads)
        logging.info("All threads stopped")
