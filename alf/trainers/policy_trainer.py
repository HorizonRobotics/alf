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
"""Trainer for training an Algorithm on given environments."""

import abc
from absl import logging
import gin
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import math
import os
import pprint
import sys
import time
import torch
import torch.nn as nn

import alf
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.config import TrainerConfig
from alf.algorithms.data_transformer import create_data_transformer
from alf.environments.utils import create_environment
from alf.utils import common
from alf.utils import git_utils
from alf.utils.checkpoint_utils import Checkpointer
from alf.utils.summary_utils import record_time


class _TrainerProgress(nn.Module):
    def __init__(self):
        super(_TrainerProgress, self).__init__()
        self.register_buffer("_iter_num", torch.zeros((), dtype=torch.int64))
        self.register_buffer("_env_steps", torch.zeros((), dtype=torch.int64))
        self._num_iterations = None
        self._num_env_steps = None
        self._progress = None

    def set_termination_criterion(self, num_iterations, num_env_steps):
        self._num_iterations = float(num_iterations)
        self._num_env_steps = float(num_env_steps)
        # might be loaded from a checkpoint, so we update first
        self.update()

    def update(self, iter_num=None, env_steps=None):
        if iter_num is not None:
            self._iter_num.fill_(iter_num)
        if env_steps is not None:
            self._env_steps.fill_(env_steps)

        assert not (self._num_iterations is None
                    and self._num_env_steps is None), (
                        "You must first call set_terimination_criterion()!")
        iter_progress, env_steps_progress = 0, 0
        if self._num_iterations > 0:
            iter_progress = float(
                self._iter_num.to(torch.float64) / self._num_iterations)
        if self._num_env_steps > 0:
            env_steps_progress = float(
                self._env_steps.to(torch.float64) / self._num_env_steps)
        # If either criterion is met, the training ends
        self._progress = max(iter_progress, env_steps_progress)

    @property
    def progress(self):
        assert self._progress is not None, "Must call update() first!"
        return self._progress


class Trainer(object):
    """Trainer.

    Trainer is responsible for creating algorithm and environment, setting up
    summary, checkpointing, running training iterations, and evaluating periodically.
    """

    _trainer_progress = _TrainerProgress()

    def __init__(self, config: TrainerConfig):
        """

        Args:
            config (TrainerConfig): configuration used to construct this trainer
        """
        root_dir = os.path.expanduser(config.root_dir)
        os.makedirs(root_dir, exist_ok=True)
        logging.get_absl_handler().use_absl_log_file(log_dir=root_dir)
        self._root_dir = root_dir
        self._train_dir = os.path.join(root_dir, 'train')
        self._eval_dir = os.path.join(root_dir, 'eval')

        self._envs = []
        self._algorithm_ctor = config.algorithm_ctor
        self._algorithm = None

        self._random_seed = config.random_seed
        self._num_iterations = config.num_iterations
        self._num_env_steps = config.num_env_steps
        assert (self._num_iterations + self._num_env_steps > 0
                and self._num_iterations * self._num_env_steps == 0), \
            "Must provide #iterations or #env_steps exclusively for training!"
        self._trainer_progress.set_termination_criterion(
            self._num_iterations, self._num_env_steps)

        self._num_checkpoints = config.num_checkpoints
        self._checkpointer = None

        self._evaluate = config.evaluate
        self._eval_interval = config.eval_interval
        self._num_eval_episodes = config.num_eval_episodes

        self._summary_interval = config.summary_interval
        self._summaries_flush_secs = config.summaries_flush_secs
        self._summary_max_queue = config.summary_max_queue
        self._debug_summaries = config.debug_summaries
        self._summarize_grads_and_vars = config.summarize_grads_and_vars
        alf.summary.should_summarize_output(config.summarize_output)
        self._config = config

        self._random_seed = common.set_random_seed(self._random_seed)

        env = self._create_environment(random_seed=self._random_seed)
        logging.info(
            "observation_spec=%s" % pprint.pformat(env.observation_spec()))
        logging.info("action_spec=%s" % pprint.pformat(env.action_spec()))
        common.set_global_env(env)

        data_transformer = create_data_transformer(
            config.data_transformer_ctor, env.observation_spec())
        self._config.data_transformer = data_transformer
        observation_spec = data_transformer.transformed_observation_spec
        common.set_transformed_observation_spec(observation_spec)

        self._algorithm = self._algorithm_ctor(
            observation_spec=observation_spec,
            action_spec=env.action_spec(),
            env=env,
            config=self._config,
            debug_summaries=self._debug_summaries)

        # Create an unwrapped env to expose subprocess gin confs which otherwise
        # will be marked as "inoperative". This env should be created last.
        # DO NOT register this env in self._envs because AsyncOffPolicyTrainer
        # will use all self._envs to init AsyncOffPolicyDriver!
        self._unwrapped_env = self._create_environment(
            nonparallel=True, random_seed=self._random_seed, register=False)
        self._eval_env = None
        self._eval_metrics = None
        self._eval_summary_writer = None
        if self._evaluate:
            self._eval_env = self._unwrapped_env
            self._eval_metrics = [
                alf.metrics.AverageReturnMetric(
                    buffer_size=self._num_eval_episodes),
                alf.metrics.AverageEpisodeLengthMetric(
                    buffer_size=self._num_eval_episodes),
                alf.metrics.AverageEnvInfoMetric(
                    example_env_info=self._eval_env.reset().env_info,
                    batch_size=self._eval_env.batch_size,
                    buffer_size=self._num_eval_episodes)
            ]
            self._eval_summary_writer = alf.summary.create_summary_writer(
                self._eval_dir, flush_secs=config.summaries_flush_secs)

    @gin.configurable('alf.trainers.Trainer._create_environment')
    def _create_environment(self,
                            nonparallel=False,
                            random_seed=None,
                            register=True):
        """Create and register an env."""
        env = create_environment(nonparallel=nonparallel, seed=random_seed)
        if register:
            self._register_env(env)
        return env

    def _register_env(self, env):
        """Register env so that later its resource will be recycled."""
        self._envs.append(env)

    def _close_envs(self):
        """Close all envs to release their resources."""
        for env in self._envs:
            env.close()
        self._unwrapped_env.close()

    @staticmethod
    def progress():
        """A static method that returns the current training progress, provided
        that only one trainer will be used for training.

        Returns:
            float: a number in :math:`[0,1]` indicating the training progress.
        """
        return Trainer._trainer_progress.progress

    @staticmethod
    def current_iterations():
        return Trainer._trainer_progress._iter_num

    @staticmethod
    def current_env_steps():
        return Trainer._trainer_progress._env_step

    def train(self):
        """Perform training."""
        self._restore_checkpoint()
        alf.summary.enable_summary()
        try:
            if self._config.profiling:
                import cProfile, pstats, io
                from pstats import SortKey
                pr = cProfile.Profile()
                pr.enable()

            common.run_under_record_context(
                self._train,
                summary_dir=self._train_dir,
                summary_interval=self._summary_interval,
                flush_secs=self._summaries_flush_secs,
                summary_max_queue=self._summary_max_queue)

            if self._config.profiling:
                pr.disable()
                s = io.StringIO()
                ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.TIME)
                ps.print_stats()
                ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
                ps.print_stats()
                ps.print_callees()

                logging.info(s.getvalue())

            self._save_checkpoint()
        finally:
            self._close_envs()

    def _train(self):
        for env in self._envs:
            env.reset()
        if self._eval_env:
            self._eval_env.reset()

        begin_iter_num = int(self._trainer_progress._iter_num)
        iter_num = begin_iter_num

        checkpoint_interval = math.ceil(
            (self._num_iterations
             or self._num_env_steps) / self._num_checkpoints)
        time_to_checkpoint = checkpoint_interval

        while True:
            t0 = time.time()
            with record_time("time/train_iter"):
                train_steps = self._algorithm.train_iter()
            t = time.time() - t0
            logging.log_every_n_seconds(
                logging.INFO,
                '%s -> %s: %s time=%.3f throughput=%0.2f' %
                (common.get_gin_file(), [
                    os.path.basename(self._root_dir.strip('/'))
                ], iter_num, t, int(train_steps) / t),
                n_seconds=1)

            if self._evaluate and (iter_num + 1) % self._eval_interval == 0:
                self._eval()
            if iter_num == begin_iter_num:
                # We need to wait for one iteration to get the operative args
                # Right just give a fixed gin file name to store operative args
                common.write_gin_configs(self._root_dir, "configured.gin")

                with alf.summary.record_if(lambda: True):

                    def _markdownify(paragraph):
                        return "    ".join(
                            (os.linesep + paragraph).splitlines(keepends=True))

                    common.summarize_gin_config()
                    alf.summary.text('commandline', ' '.join(sys.argv))
                    alf.summary.text(
                        'optimizers',
                        _markdownify(self._algorithm.get_optimizer_info()))
                    alf.summary.text('revision', git_utils.get_revision())
                    alf.summary.text('diff',
                                     _markdownify(git_utils.get_diff()))
                    alf.summary.text('seed', str(self._random_seed))

            # check termination
            env_steps_metric = self._algorithm.get_step_metrics()[1]
            total_time_steps = env_steps_metric.result()
            iter_num += 1

            self._trainer_progress.update(iter_num, total_time_steps)

            if ((self._num_iterations and iter_num >= self._num_iterations)
                    or (self._num_env_steps
                        and total_time_steps >= self._num_env_steps)):
                # Evaluate before exiting so that the eval curve shown in TB
                # will align with the final iter/env_step.
                if self._evaluate:
                    self._eval()
                break

            if ((self._num_iterations and iter_num >= time_to_checkpoint)
                    or (self._num_env_steps
                        and total_time_steps >= time_to_checkpoint)):
                self._save_checkpoint()
                time_to_checkpoint += checkpoint_interval

    def _restore_checkpoint(self):
        checkpointer = Checkpointer(
            ckpt_dir=os.path.join(self._train_dir, 'algorithm'),
            algorithm=self._algorithm,
            metrics=nn.ModuleList(self._algorithm.get_metrics()),
            trainer_progress=self._trainer_progress)

        if checkpointer.has_checkpoint():
            # Some objects (e.g. ReplayBuffer) are constructed lazily in algorithm.
            # They only appear after one training iteration. So we need to run
            # train_iter() once before loading the checkpoint
            self._algorithm.train_iter()

        try:
            recovered_global_step = checkpointer.load()
        except Exception as e:
            raise RuntimeError(
                ("Checkpoint loading failed from the provided root_dir={}. "
                 "Typically this is caused by using a wrong checkpoint. \n"
                 "Please make sure the root_dir is set correctly. "
                 "Use a new value for it if "
                 "planning to train from scratch. \n"
                 "Detailed error message: {}").format(self._root_dir, e))
        if recovered_global_step != -1:
            alf.summary.set_global_counter(recovered_global_step)

        self._checkpointer = checkpointer

    def _save_checkpoint(self):
        global_step = alf.summary.get_global_counter()
        self._checkpointer.save(global_step=global_step)

    @common.mark_eval
    def _eval(self):
        self._algorithm.eval()
        time_step = common.get_initial_time_step(self._eval_env)
        policy_state = self._algorithm.get_initial_predict_state(
            self._eval_env.batch_size)
        trans_state = self._algorithm.get_initial_transform_state(
            self._eval_env.batch_size)
        episodes = 0
        while episodes < self._num_eval_episodes:
            time_step, policy_state, trans_state = _step(
                algorithm=self._algorithm,
                env=self._eval_env,
                time_step=time_step,
                policy_state=policy_state,
                trans_state=trans_state,
                epsilon_greedy=self._config.epsilon_greedy,
                metrics=self._eval_metrics)
            if time_step.is_last():
                episodes += 1

        step_metrics = self._algorithm.get_step_metrics()
        with alf.summary.push_summary_writer(self._eval_summary_writer):
            for metric in self._eval_metrics:
                metric.gen_summaries(
                    train_step=alf.summary.get_global_counter(),
                    step_metrics=step_metrics)

        common.log_metrics(self._eval_metrics)
        self._algorithm.train()


@torch.no_grad()
def _step(algorithm, env, time_step, policy_state, trans_state, epsilon_greedy,
          metrics):
    policy_state = common.reset_state_if_necessary(
        policy_state, algorithm.get_initial_predict_state(env.batch_size),
        time_step.is_first())
    transformed_time_step, trans_state = algorithm.transform_timestep(
        time_step, trans_state)
    policy_step = algorithm.predict_step(transformed_time_step, policy_state,
                                         epsilon_greedy)
    next_time_step = env.step(policy_step.output)
    for metric in metrics:
        metric(time_step.cpu())
    return next_time_step, policy_step.state, trans_state


def play(root_dir,
         env,
         algorithm,
         checkpoint_step="latest",
         epsilon_greedy=0.1,
         num_episodes=10,
         max_episode_length=0,
         sleep_time_per_step=0.01,
         record_file=None,
         ignored_parameter_prefixes=['_exp_replayer.']):
    """Play using the latest checkpoint under `train_dir`.

    The following example record the play of a trained model to a mp4 video:
    .. code-block:: bash

        python -m alf.bin.play \
        --root_dir=~/tmp/bullet_humanoid/ppo2/ppo2-11 \
        --num_episodes=1 \
        --record_file=ppo_bullet_humanoid.mp4

    Args:
        root_dir (str): same as the root_dir used for `train()`
        env (AlfEnvironment): the environment
        algorithm (RLAlgorithm): the training algorithm
        checkpoint_step (int|str): the number of training steps which is used to
            specify the checkpoint to be loaded. If checkpoint_step is 'latest',
            the most recent checkpoint named 'latest' will be loaded.
        epsilon_greedy (float): a floating value in [0,1], representing the
            chance of action sampling instead of taking argmax. This can
            help prevent a dead loop in some deterministic environment like
            Breakout.
        num_episodes (int): number of episodes to play
        max_episode_length (int): if >0, each episode is limited to so many
            steps.
        sleep_time_per_step (float): sleep so many seconds for each step
        record_file (str): if provided, video will be recorded to a file
            instead of shown on the screen.
        ignored_parameter_prefixes (list[str]): ignore the parameters whose
            name has one of these prefixes in the checkpoint. This is useful
            for skipping loading the checkpoint of ReplayBuffer.
"""
    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')

    ckpt_dir = os.path.join(train_dir, 'algorithm')
    checkpointer = Checkpointer(ckpt_dir=ckpt_dir, algorithm=algorithm)
    checkpointer.load(
        checkpoint_step, ignored_parameter_prefixes=ignored_parameter_prefixes)

    recorder = None
    if record_file is not None:
        recorder = VideoRecorder(env, path=record_file)
    else:
        # pybullet_envs need to render() before reset() to enable mode='human'
        env.render(mode='human')
    env.reset()
    if recorder:
        recorder.capture_frame()
    time_step = common.get_initial_time_step(env)
    algorithm.eval()
    policy_state = algorithm.get_initial_predict_state(env.batch_size)
    trans_state = algorithm.get_initial_transform_state(env.batch_size)
    episode_reward = 0.
    episode_length = 0
    episodes = 0
    while episodes < num_episodes:
        time_step, policy_state, trans_state = _step(
            algorithm=algorithm,
            env=env,
            time_step=time_step,
            policy_state=policy_state,
            trans_state=trans_state,
            epsilon_greedy=epsilon_greedy,
            metrics=[])
        episode_length += 1
        if recorder:
            recorder.capture_frame()
        else:
            env.render(mode='human')
            time.sleep(sleep_time_per_step)

        episode_reward += float(time_step.reward)

        if time_step.is_last() or episode_length >= max_episode_length > 0:
            logging.info("episode_length=%s episode_reward=%s" %
                         (episode_length, episode_reward))
            episode_reward = 0.
            episode_length = 0.
            episodes += 1
            time_step = env.reset()

    if recorder:
        recorder.close()
    env.reset()
