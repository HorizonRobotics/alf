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
import math
import os
import pprint
import signal
import sys
import time
import torch
import torch.nn as nn

import alf
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.config import TrainerConfig
from alf.algorithms.data_transformer import create_data_transformer
from alf.environments.utils import create_environment
from alf.nest import map_structure
from alf.tensor_specs import TensorSpec
from alf.utils import common
from alf.utils import git_utils
from alf.utils import math_ops
from alf.utils.checkpoint_utils import Checkpointer
import alf.utils.datagen as datagen
from alf.utils.summary_utils import record_time
from alf.utils.video_recorder import VideoRecorder


@gin.configurable
def create_dataset(dataset_name='mnist',
                   dataset_loader=datagen,
                   train_batch_size=100,
                   test_batch_size=100):
    """Create a pytorch data loaders.

    Args:
        dataset_name (str): dataset_name
        dataset_loader (Callable) : callable that create pytorch data
            loaders for both training and testing.
        train_batch_size (int): batch_size for training.
        test_batch_size (int): batch_size for testing.

    Returns:
        trainset (torch.utils.data.DataLoaderr):
        testset (torch.utils.data.DataLoaderr):
    """

    trainset, testset = getattr(dataset_loader,
                                'load_{}'.format(dataset_name))(
                                    train_bs=train_batch_size,
                                    test_bs=test_batch_size)
    return trainset, testset


class _TrainerProgress(nn.Module):
    def __init__(self):
        super(_TrainerProgress, self).__init__()
        self.register_buffer("_iter_num", torch.zeros((), dtype=torch.int64))
        self.register_buffer("_env_steps", torch.zeros((), dtype=torch.int64))
        self._num_iterations = None
        self._num_env_steps = None
        self._progress = None

    def set_termination_criterion(self, num_iterations, num_env_steps=0):
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
    """Base class for trainers.

    Trainer is responsible for creating algorithm and dataset/environment, setting up
    summary, checkpointing, running training iterations, and evaluating periodically.
    """

    _trainer_progress = _TrainerProgress()

    def __init__(self, config: TrainerConfig):
        """

        Args:
            config (TrainerConfig): configuration used to construct this trainer
        """
        Trainer._trainer_progress = _TrainerProgress()
        root_dir = os.path.expanduser(config.root_dir)
        os.makedirs(root_dir, exist_ok=True)
        logging.get_absl_handler().use_absl_log_file(log_dir=root_dir)
        self._root_dir = root_dir
        self._train_dir = os.path.join(root_dir, 'train')
        self._eval_dir = os.path.join(root_dir, 'eval')

        self._algorithm_ctor = config.algorithm_ctor
        self._algorithm = None

        self._num_checkpoints = config.num_checkpoints
        self._checkpointer = None

        self._evaluate = config.evaluate
        self._eval_interval = config.eval_interval

        self._summary_interval = config.summary_interval
        self._summaries_flush_secs = config.summaries_flush_secs
        self._summary_max_queue = config.summary_max_queue
        self._debug_summaries = config.debug_summaries
        self._summarize_grads_and_vars = config.summarize_grads_and_vars
        self._config = config

        self._random_seed = common.set_random_seed(config.random_seed)

    def train(self):
        """Perform training."""
        self._restore_checkpoint()
        alf.summary.enable_summary()

        self._checkpoint_requested = False
        signal.signal(signal.SIGUSR2, self._request_checkpoint)
        logging.info("Use `kill -%s %s` to request checkpoint during training."
                     % (int(signal.SIGUSR2), os.getpid()))

        self._debug_requested = False
        signal.signal(signal.SIGUSR1, self._request_debug)
        logging.info("Use `kill -%s %s` to request debugging." % (int(
            signal.SIGUSR1), os.getpid()))

        checkpoint_saved = False
        try:
            if self._config.profiling:
                import cProfile, pstats, io
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
                ps = pstats.Stats(pr, stream=s).sort_stats('time')
                ps.print_stats()
                ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
                ps.print_stats()
                ps.print_callees()

                logging.info(s.getvalue())
            self._save_checkpoint()
            checkpoint_saved = True
        finally:
            if self._config.confirm_checkpoint_upon_crash and not checkpoint_saved:
                ans = input("Do you want to save checkpoint? (y/n): ")
                if ans.lower().startswith('y'):
                    self._save_checkpoint()
            self._close()

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

    def _train(self):
        """Perform training according the the learning type. """
        pass

    def _close(self):
        """Closing operations after training. """
        pass

    def _summarize_training_setting(self):
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
            alf.summary.text('diff', _markdownify(git_utils.get_diff()))
            alf.summary.text('seed', str(self._random_seed))

    def _request_checkpoint(self, signum, frame):
        self._checkpoint_requested = True

    def _request_debug(self, signum, frame):
        self._debug_requested = True

    def _save_checkpoint(self):
        global_step = alf.summary.get_global_counter()
        self._checkpointer.save(global_step=global_step)

    def _restore_checkpoint(self, checkpointer):
        """Retore from saved checkpoint.

            Args:
                checkpointer (Checkpointer):
        """
        if checkpointer.has_checkpoint():
            # Some objects (e.g. ReplayBuffer) are constructed lazily in algorithm.
            # They only appear after one training iteration. So we need to run
            # train_iter() once before loading the checkpoint
            self._algorithm.train_iter()

        try:
            recovered_global_step = checkpointer.load()
            self._trainer_progress.update()
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


class RLTrainer(Trainer):
    """Trainer for reinforcement learning. """

    def __init__(self, config: TrainerConfig):
        """

        Args:
            config (TrainerConfig): configuration used to construct this trainer
        """
        super().__init__(config)

        self._envs = []
        self._num_env_steps = config.num_env_steps
        self._num_iterations = config.num_iterations
        assert (self._num_iterations + self._num_env_steps > 0
                and self._num_iterations * self._num_env_steps == 0), \
            "Must provide #iterations or #env_steps exclusively for training!"
        self._trainer_progress.set_termination_criterion(
            self._num_iterations, self._num_env_steps)

        self._num_eval_episodes = config.num_eval_episodes
        alf.summary.should_summarize_output(config.summarize_output)

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
        if self._evaluate or isinstance(
                env,
                alf.environments.parallel_environment.ParallelAlfEnvironment):
            self._unwrapped_env = self._create_environment(
                nonparallel=True,
                random_seed=self._random_seed,
                register=False)
        else:
            self._unwrapped_env = None
        self._eval_env = None
        self._eval_metrics = None
        self._eval_summary_writer = None
        if self._evaluate:
            self._eval_env = self._unwrapped_env
            self._eval_metrics = [
                alf.metrics.AverageReturnMetric(
                    buffer_size=self._num_eval_episodes,
                    reward_shape=self._eval_env.reward_spec().shape),
                alf.metrics.AverageEpisodeLengthMetric(
                    buffer_size=self._num_eval_episodes),
                alf.metrics.AverageEnvInfoMetric(
                    example_env_info=self._eval_env.reset().env_info,
                    batch_size=self._eval_env.batch_size,
                    buffer_size=self._num_eval_episodes)
            ]
            self._eval_summary_writer = alf.summary.create_summary_writer(
                self._eval_dir, flush_secs=config.summaries_flush_secs)

    @gin.configurable('alf.trainers.RLTrainer._create_environment')
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
        if self._unwrapped_env is not None:
            self._unwrapped_env.close()

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

        if self._num_iterations:
            time_to_checkpoint = self._trainer_progress._iter_num + checkpoint_interval
        else:
            time_to_checkpoint = self._trainer_progress._num_env_steps + checkpoint_interval

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
                self._summarize_training_setting()

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
            elif self._checkpoint_requested:
                logging.info("Saving checkpoint upon request...")
                self._save_checkpoint()
                self._checkpoint_requested = False

            if self._debug_requested:
                self._debug_requested = False
                import pdb
                pdb.set_trace()

    def _close(self):
        """Closing operations after training. """
        self._close_envs()

    def _restore_checkpoint(self):
        checkpointer = Checkpointer(
            ckpt_dir=os.path.join(self._train_dir, 'algorithm'),
            algorithm=self._algorithm,
            metrics=nn.ModuleList(self._algorithm.get_metrics()),
            trainer_progress=self._trainer_progress)

        super()._restore_checkpoint(checkpointer)

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
            time_step, policy_step, trans_state = _step(
                algorithm=self._algorithm,
                env=self._eval_env,
                time_step=time_step,
                policy_state=policy_state,
                trans_state=trans_state,
                epsilon_greedy=self._config.epsilon_greedy,
                metrics=self._eval_metrics)
            policy_state = policy_step.state

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


class SLTrainer(Trainer):
    """Trainer for supervised learning. """

    def __init__(self, config: TrainerConfig):
        """Create a SLTrainer

        Args:
            config (TrainerConfig): configuration used to construct this trainer
        """
        super().__init__(config)

        assert config.num_iterations > 0, \
            "Must provide num_iterations for training!"

        self._num_epochs = config.num_iterations
        self._trainer_progress.set_termination_criterion(self._num_epochs)

        trainset, testset = self._create_dataset()
        input_tensor_spec = TensorSpec(shape=trainset.dataset[0][0].shape)
        if hasattr(trainset.dataset, 'classes'):
            output_dim = len(trainset.dataset.classes)
        else:
            output_dim = len(trainset.dataset[0][1])

        self._algorithm = config.algorithm_ctor(
            input_tensor_spec=input_tensor_spec,
            last_layer_param=(output_dim, True),
            last_activation=math_ops.identity,
            config=config)

        self._algorithm.set_data_loader(trainset, testset)

    def _create_dataset(self):
        """Create data loaders."""
        return create_dataset()

    def _train(self):
        begin_epoch_num = int(self._trainer_progress._iter_num)
        epoch_num = begin_epoch_num

        checkpoint_interval = math.ceil(
            self._num_epochs / self._num_checkpoints)
        time_to_checkpoint = checkpoint_interval

        logging.info("==> Begin Training")
        while True:
            logging.info("-" * 68)
            logging.info("Epoch: {}".format(epoch_num + 1))
            with record_time("time/train_iter"):
                self._algorithm.train_iter()

            if self._evaluate and (epoch_num + 1) % self._eval_interval == 0:
                self._algorithm.evaluate()

            if epoch_num == begin_epoch_num:
                self._summarize_training_setting()

            # check termination
            epoch_num += 1
            self._trainer_progress.update(epoch_num)

            if (self._num_epochs and epoch_num >= self._num_epochs):
                if self._evaluate:
                    self._algorithm.evaluate()
                break

            if self._num_epochs and epoch_num >= time_to_checkpoint:
                self._save_checkpoint()
                time_to_checkpoint += checkpoint_interval
            elif self._checkpoint_requested:
                logging.info("Saving checkpoint upon request...")
                self._save_checkpoint()
                self._checkpoint_requested = False

            if self._debug_requested:
                self._debug_requested = False
                import pdb
                pdb.set_trace()

    def _restore_checkpoint(self):
        checkpointer = Checkpointer(
            ckpt_dir=os.path.join(self._train_dir, 'algorithm'),
            algorithm=self._algorithm,
            trainer_progress=self._trainer_progress)

        super()._restore_checkpoint(checkpointer)


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
    return next_time_step, policy_step, trans_state


def play(root_dir,
         env,
         algorithm,
         checkpoint_step="latest",
         epsilon_greedy=0.1,
         num_episodes=10,
         max_episode_length=0,
         sleep_time_per_step=0.01,
         record_file=None,
         future_steps=0,
         ignored_parameter_prefixes=[]):
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
        future_steps (int): whether to encode some information from future steps
            into the current frame. If future_steps is larger than zero,
            then the episodic information will be cached and the encoding of
            them to video frames is deferred to the end of an episode.
            This defer mode is potentially useful to display for each frame
            some information that expands beyond a single time step to the future.
            Currently this mode only support offline rendering, i.e. rendering
            and saving the video to ``record_file``. If a non-positive value is
            provided, it is treated as not using the defer mode and the plots
            for displaying future information will not be displayed.
        ignored_parameter_prefixes (list[str]): ignore the parameters whose
            name has one of these prefixes in the checkpoint.
"""
    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')

    ckpt_dir = os.path.join(train_dir, 'algorithm')
    checkpointer = Checkpointer(ckpt_dir=ckpt_dir, algorithm=algorithm)
    checkpointer.load(
        checkpoint_step,
        ignored_parameter_prefixes=ignored_parameter_prefixes,
        including_optimizer=False,
        including_replay_buffer=False)

    recorder = None
    if record_file is not None:
        recorder = VideoRecorder(env, path=record_file)
    else:
        # pybullet_envs need to render() before reset() to enable mode='human'
        env.render(mode='human')
    env.reset()

    time_step = common.get_initial_time_step(env)
    algorithm.eval()
    policy_state = algorithm.get_initial_predict_state(env.batch_size)
    trans_state = algorithm.get_initial_transform_state(env.batch_size)
    episode_reward = 0.
    episode_length = 0
    episodes = 0
    metrics = [
        alf.metrics.AverageReturnMetric(
            buffer_size=num_episodes, reward_shape=env.reward_spec().shape),
        alf.metrics.AverageEpisodeLengthMetric(buffer_size=num_episodes),
    ]

    if future_steps > 0:
        defer_mode = True
        observations = []
        rewards = []
        actions = []
    else:
        defer_mode = False

    while episodes < num_episodes:

        # transform time step
        transformed_time_step, trans_state = algorithm.transform_timestep(
            time_step, trans_state)

        time_step, policy_step, trans_state = _step(
            algorithm=algorithm,
            env=env,
            time_step=time_step,
            policy_state=policy_state,
            trans_state=trans_state,
            epsilon_greedy=epsilon_greedy,
            metrics=metrics)
        policy_state = policy_step.state
        action = policy_step.output
        info = policy_step.info

        if defer_mode:
            observations.append(transformed_time_step.observation)
            rewards.append(transformed_time_step.reward)
            actions.append(action)

        episode_length += 1
        if recorder:
            recorder.capture_frame(info, encode_frame=not defer_mode)
        else:
            env.render(mode='human')
            time.sleep(sleep_time_per_step)

        time_step_reward = time_step.reward.view(-1).float().cpu().numpy()

        episode_reward += time_step_reward

        # start defer encoding
        if defer_mode and (episode_length > future_steps):
            _record_future_info(
                recorder,
                observations,
                actions,
                rewards,
                future_steps=future_steps)
            observations.pop(0)
            actions.pop(0)
            rewards.pop(0)

        if time_step.is_last() or episode_length >= max_episode_length > 0:
            logging.info("episode_length=%s episode_reward=%s" %
                         (episode_length, episode_reward))

            # defer encoding all the rest till the end
            if defer_mode:
                _record_future_info(
                    recorder,
                    observations,
                    actions,
                    rewards,
                    future_steps=future_steps,
                    encode_all=True)
                observations = []
                actions = []
                rewards = []

            episode_reward = 0.
            episode_length = 0.
            episodes += 1
            # observe the last step
            for m in metrics:
                m(time_step.cpu())
            time_step = env.reset()

    for m in metrics:
        logging.info(
            "%s: %s", m.name,
            map_structure(
                lambda x: x.numpy().item() if x.ndim == 0 else x.numpy(),
                m.result()))
    if recorder:
        recorder.close()
    env.reset()


def _record_future_info(recorder,
                        observations,
                        actions,
                        rewards,
                        info_func=None,
                        future_steps=5,
                        encode_all=False):
    """Record future information and encode with the recoder.
    This function extracts some information of ``future_steps`` into
    the future, based on the input observations/actions/rewards.
    By default, the reward and actions from the current time step
    to the next ``future_steps`` will be displayed for each frame.
    User can use ``info_func`` to add customized predictive quantities
    to be shown in the video frames.

    Args:
        recorder (VideoRecorder):
        observations (list[Tensor]): list of observations captured in an episode.
            It can be used as the input to some algorithms to get the information
            to be displayed in the video, e.g. a predictive algorithm predicting
            future observations/rewards/values.
        actions (list[Tensor]): list of actions captured in an episode.
        rewards (list[Tensor]: list of rewards captured in an episode.
        info_func (None|callable): a callable for calculating some customized
            information (e.g. predicted future reward) based on the observation
            at each time step and action sequences from the current time step
            to the next ``future_steps`` steps (if available). It is called
            as ``pred_info=info_func(current_observation, action_sequences)``.
            Currently only support displaying scalar predictive information
            returned from info_func.
        future_steps (int): number of future steps to show.
        encode_all (bool): whether to encode all the steps in the episode
            buffer (i.e. the list of observations/actions/rewards).
            - If False, only encode one step. In this case, ``future_steps``
                should be no larger than the length of the episode buffer.
            - If True, encode all the steps in episode_buffer. In this case,
                the actual ``future_steps`` is upper-bounded by the
                length of the episode buffer - 1.
    """
    # [episode_buffer_length, reward_dim]
    rewards = torch.cat(rewards, dim=0)
    episode_buffer_length = rewards.shape[0]

    if not encode_all:
        assert future_steps < episode_buffer_length, (
            "future steps should be smaller than "
            "the episode buffer length {a}, but got {b}".format(
                a=episode_buffer_length, b=future_steps))

    if rewards.ndim > 1:
        # slice the multi-dimensional rewards
        # assume the first dimension is the overall reward
        rewards = rewards[..., 0]

    actions = torch.cat(actions, dim=0)

    num_steps = future_steps + 1

    reward_curve_set = []
    action_curve_set = []
    predictive_curve_set = []

    encoding_steps = episode_buffer_length if encode_all else 1
    for t in range(encoding_steps):
        H = min(num_steps, episode_buffer_length - t)  # total display steps
        if H > 0:
            t_actions = actions[t:t + H]
            t_rewards = rewards[t:t + H]

            if info_func is not None:
                predictions = info_func(observations[t], t_actions)
                assert predictions.ndim == 1 or predictions.shape[1] == 1, \
                    "only support displaying scalar predictive information"
                predictions = predictions.view(-1).detach().cpu().numpy()

                pred_curve = recorder._plot_value_curve([predictions],
                                                        legends=["Prediction"],
                                                        size=6,
                                                        linewidth=5,
                                                        name="prediction")
                predictive_curve_set.append(pred_curve)

            reward_gt = t_rewards.view(-1).cpu().numpy()
            action_cpu = t_actions.detach().cpu().numpy()

            reward_curve = recorder._plot_value_curve([reward_gt],
                                                      legends=["GroundTruth"],
                                                      size=6,
                                                      linewidth=5,
                                                      name="rewards")
            reward_curve_set.append(reward_curve)

            action_curve = recorder._plot_value_curve(
                [action_cpu[..., i] for i in range(action_cpu.shape[-1])],
                legends=["a" + str(i) for i in range(action_cpu.shape[-1])],
                size=6,
                linewidth=5,
                name="actions")
            action_curve_set.append(action_curve)

    # encode all frames
    recorder.encode_frames_in_buffer_with_external(
        [reward_curve_set, action_curve_set, predictive_curve_set])
