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

import abc
from absl import logging
import gin.tf
import os
import sys
import time
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import alf
from alf.algorithms.agent import Agent
from alf.environments.utils import create_environment
from alf.utils.metric_utils import eager_compute
from alf.utils import common
from alf.utils.checkpoint_utils import Checkpointer
from alf.utils.summary_utils import record_time
from alf.utils import git_utils


@gin.configurable
class TrainerConfig(object):
    """TrainerConfig

    Note: This is a mixture collection configuration for all trainers,
    not all parameter is operative and its not necessary to config it.

    1. `num_steps_per_iter` is only for on_policy_trainer.

    2. `initial_collect_steps`, `num_updates_per_train_step`, `mini_batch_length`,
    `mini_batch_size`, `clear_replay_buffer`, `num_envs` are used by sync_off_policy_trainer and
    async_off_policy_trainer.
    """

    def __init__(self,
                 root_dir,
                 algorithm_ctor=None,
                 random_seed=None,
                 num_iterations=1000,
                 num_env_steps=0,
                 unroll_length=8,
                 use_rollout_state=False,
                 num_checkpoints=10,
                 evaluate=False,
                 eval_interval=10,
                 epsilon_greedy=0.1,
                 num_eval_episodes=10,
                 summary_interval=50,
                 update_counter_every_mini_batch=False,
                 summaries_flush_secs=1,
                 summary_max_queue=10,
                 debug_summaries=False,
                 summarize_grads_and_vars=False,
                 summarize_action_distributions=False,
                 num_steps_per_iter=10000,
                 initial_collect_steps=0,
                 num_updates_per_train_step=4,
                 mini_batch_length=None,
                 mini_batch_size=None,
                 clear_replay_buffer=True,
                 num_envs=1):
        """Configuration for Trainers

        Args:
            root_dir (str): directory for saving summary and checkpoints
            algorithm_ctor (Callable): callable that create an
                `OffPolicyAlgorithm` or `OnPolicyAlgorithm` instance
            random_seed (None|int): random seed, a random seed is used if None
            num_iterations (int): number of update iterations (ignored if 0)
            num_env_steps (int): number of environment steps (ignored if 0). The
                total number of FRAMES will be (`num_env_steps`*`frame_skip`) for
                calculating sample efficiency. See alf/environments/wrappers.py
                for the definition of FrameSkip.
            unroll_length (int):  number of time steps each environment proceeds per
                iteration. The total number of time steps from all environments per
                iteration can be computed as: `num_envs` * `env_batch_size`
                * `unroll_length`.
            use_rollout_state (bool): Include the RNN state for the experiences
                used for off-policy training
            checkpoint_interval (int): checkpoint every so many iterations
            checkpoint_max_to_keep (int): Maximum number of checkpoints to keep
                (if greater than the max are saved, the oldest checkpoints are
                deleted). If None, all checkpoints will be kept.
            evaluate (bool): A bool to evaluate when training
            eval_interval (int): evaluate every so many iteration
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation.
            num_eval_episodes (int) : number of episodes for one evaluation
            summary_interval (int): write summary every so many training steps
            update_counter_every_mini_batch (bool): whether to update counter
                for every mini batch. The `summary_interval` is based on this
                counter. Typically, this should be False. Set to True if you
                want to have summary for every mini batch for the purpose of
                debugging.
            summaries_flush_secs (int): flush summary to disk every so many seconds
            summary_max_queue (int): flush to disk every so mary summaries
            debug_summaries (bool): A bool to gather debug summaries.
            summarize_grads_and_vars (bool): If True, gradient and network variable
                summaries will be written during training.
            num_steps_per_iter (int): number of steps for one iteration. It is the
                total steps from all individual environment in the batch
                environment.
            initial_collect_steps (int): if positive, number of steps each single
                environment steps before perform first update
            num_updates_per_train_step (int): number of optimization steps for
                one iteration
            mini_batch_size (int): number of sequences for each minibatch. If None,
                it's set to the replayer's `batch_size`.
            mini_batch_length (int): the length of the sequence for each
                sample in the minibatch. If None, it's set to `unroll_length`.
            clear_replay_buffer (bool): whether use all data in replay buffer to
                perform one update and then wiped clean
            num_envs (int): the number of environments to run asynchronously.
        """
        self._parameters = dict(
            root_dir=root_dir,
            algorithm_ctor=algorithm_ctor,
            random_seed=random_seed,
            num_iterations=num_iterations,
            num_env_steps=num_env_steps,
            unroll_length=unroll_length,
            use_rollout_state=use_rollout_state,
            num_checkpoints=num_checkpoints,
            evaluate=evaluate,
            eval_interval=eval_interval,
            epsilon_greedy=epsilon_greedy,
            num_eval_episodes=num_eval_episodes,
            summary_interval=summary_interval,
            update_counter_every_mini_batch=update_counter_every_mini_batch,
            summaries_flush_secs=summaries_flush_secs,
            summary_max_queue=summary_max_queue,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            summarize_action_distributions=summarize_action_distributions,
            num_steps_per_iter=num_steps_per_iter,
            initial_collect_steps=initial_collect_steps,
            num_updates_per_train_step=num_updates_per_train_step,
            mini_batch_length=mini_batch_length,
            mini_batch_size=mini_batch_size,
            clear_replay_buffer=clear_replay_buffer,
            num_envs=num_envs)

        self._trainer = trainer

    def __getattr__(self, param_name):
        return self._parameters.get(param_name)


class Trainer(object):
    """Abstract base class for on-policy and off-policy trainer."""

    def __init__(self, config: TrainerConfig):
        """Create a Trainer instance.

        Args:
            config (TrainerConfig): configuration used to construct this trainer
        """
        root_dir = os.path.expanduser(config.root_dir)
        self._root_dir = root_dir
        self._train_dir = os.path.join(root_dir, 'train')
        self._eval_dir = os.path.join(root_dir, 'eval')

        self._envs = []
        self._algorithm_ctor = config.algorithm_ctor
        self._algorithm = None
        self._driver = None

        self._random_seed = config.random_seed
        self._num_iterations = config.num_iterations
        self._num_env_steps = config.num_env_steps
        assert self._num_iterations + self._num_env_steps > 0, \
            "Must provide #iterations or #env_steps for training!"
        self._unroll_length = config.unroll_length

        self._checkpoint_interval = config.checkpoint_interval
        self._checkpoint_max_to_keep = config.checkpoint_max_to_keep
        self._checkpointer = None

        self._evaluate = config.evaluate
        self._eval_interval = config.eval_interval
        self._num_eval_episodes = config.num_eval_episodes
        eval_metrics = None
        eval_summary_writer = None
        if self._evaluate:
            eval_metrics = [
                metrics.AverageReturnMetric(
                    buffer_size=self._num_eval_episodes),
                metrics.AverageEpisodeLengthMetric(
                    buffer_size=self._num_eval_episodes)
            ]
            eval_summary_writer = alf.summary.create_file_writer(
                self._eval_dir,
                flush_millis=config.summaries_flush_secs * 1000)
        self._eval_env = None
        self._eval_summary_writer = eval_summary_writer
        self._eval_metrics = eval_metrics

        self._summary_interval = config.summary_interval
        self._summaries_flush_mills = config.summaries_flush_secs * 1000
        self._summary_max_queue = config.summary_max_queue
        self._debug_summaries = config.debug_summaries
        self._summarize_grads_and_vars = config.summarize_grads_and_vars
        self._config = config
        self._initialize()

    def _initialize(self):
        """Initializes the Trainer."""
        self._random_seed = common.set_random_seed(self._random_seed)

        env = self._create_environment(random_seed=self._random_seed)
        common.set_global_env(env)

        self._algorithm = self._algorithm_ctor(
            observation_spec=env.observation_spec(),
            action_spec=env.action_spec(),
            debug_summaries=self._debug_summaries)
        self._algorithm.set_summary_settings(
            summarize_grads_and_vars=self._summarize_grads_and_vars,
            summarize_action_distributions=self._config.
            summarize_action_distributions)
        self._algorithm.use_rollout_state = self._config.use_rollout_state

        # Create an unwrapped env to expose subprocess gin confs which otherwise
        # will be marked as "inoperative". This env should be created last.
        # DO NOT register this env in self._envs because AsyncOffPolicyTrainer
        # will use all self._envs to init AsyncOffPolicyDriver!
        self._unwrapped_env = self._create_environment(
            nonparallel=True, random_seed=self._random_seed, register=False)
        if self._evaluate:
            self._eval_env = self._unwrapped_env

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
            env.pyenv.close()
        self._unwrapped_env.pyenv.close()

    @abc.abstractmethod
    def _init_driver(self):
        """Initialize driver

        Sub class should implement this method and return an instance of `PolicyDriver`

        Returns:
            driver (PolicyDriver): driver used for collecting data and training
        """
        pass

    def train(self):
        """Perform training."""
        self._restore_checkpoint()
        common.enable_summary(True)
        common.run_under_record_context(
            self._train,
            summary_dir=self._train_dir,
            summary_interval=self._summary_interval,
            flush_millis=self._summaries_flush_mills,
            summary_max_queue=self._summary_max_queue)
        self._save_checkpoint()
        self._close_envs()

    def _train(self):
        for env in self._envs:
            env.reset()
        iter_num = 0
        while True:
            t0 = time.time()
            with record_time("time/train_iter"):
                train_steps = self._algorithm.train_iter()
            t = time.time() - t0
            logging.log_every_n_seconds(
                logging.INFO,
                '%s time=%.3f throughput=%0.2f' % (iter_num, t,
                                                   int(train_steps) / t),
                n_seconds=1)
            if (iter_num + 1) % self._checkpoint_interval == 0:
                self._save_checkpoint()
            if self._evaluate and (iter_num + 1) % self._eval_interval == 0:
                self._eval()
            if iter_num == 0:
                # We need to wait for one iteration to get the operative args
                # Right just give a fixed gin file name to store operative args
                common.write_gin_configs(self._root_dir, "configured.gin")

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

            # check termination
            env_steps_metric = self._driver.get_step_metrics()[1]
            total_time_steps = env_steps_metric.result().numpy()
            iter_num += 1
            if (self._num_iterations and iter_num >= self._num_iterations) \
                or (self._num_env_steps and total_time_steps >= self._num_env_steps):
                break

    def _restore_checkpoint(self):
        global_step = alf.summary.get_global_counter()
        checkpointer = Checkpointer(
            ckpt_dir=os.path.join(self._train_dir, 'algorithm'),
            algorithm=self._algorithm,
            metrics=metric_utils.MetricsGroup(self._driver.get_metrics(),
                                              'metrics'),
            global_step=global_step)
        checkpointer.load()
        self._checkpointer = checkpointer

    def _save_checkpoint(self):
        global_step = alf.summary.get_global_counter()
        self._checkpointer.save(global_step=global_step.numpy())

    def _eval(self):
        time_step = get_initial_time_step(self._eval_env)
        policy_state = self._algorithm.get_initial_predict_state(
            self._eval_env.batch_size)
        episodes = 0
        while episodes < self._num_eval_episodes:
            time_step, policy_state = _step(
                algortihm=self._algorithm,
                env=self._eval_env,
                time_ste=time_step,
                policy_state=policy_state,
                epsilon_greedy=self._config.epsilon_greedy,
                metrics=self._eval_metrics)
            if time_step.is_last():
                episodes += 1
        metric_utils.log_metrics(self._eval_metrics)


def get_initial_time_step(env):
    return common.zero_tensor_from_nested_spec(env.time_step_spec,
                                               env.batch_size)


def _step(algorithm, env, time_step, policy_state, epsilon_greedy, metrics):
    policy_state = common.reset_state_if_necessary(
        policy_state, algorithm.get_initial_predict_state(env.batch_size),
        time_step.is_first())
    transformed_time_step = algorithm.transform_timestep(time_step)
    policy_step = algorithm.predict(transformed_time_step, policy_state,
                                    epsilon_greedy)
    next_time_step = env.step(policy_step.output)

    exp = alf.data_structures.make_experience(time_step, policy_step,
                                              policy_state)
    for metric in metrics:
        metric.observe(exp)
    return next_time_step, policy_step.state


def play(root_dir,
         env,
         algorithm,
         checkpoint_name="latest",
         epsilon_greedy=0.1,
         num_episodes=10,
         sleep_time_per_step=0.01,
         record_file=None):
    """Play using the latest checkpoint under `train_dir`.

    The following example record the play of a trained model to a mp4 video:
    ```bash
    python -m alf.bin.play \
    --root_dir=~/tmp/bullet_humanoid/ppo2/ppo2-11 \
    --num_episodes=1 \
    --record_file=ppo_bullet_humanoid.mp4
    ```
    Args:
        root_dir (str): same as the root_dir used for `train()`
        env (TFEnvironment): the environment
        algorithm (OnPolicyAlgorithm): the training algorithm
        checkpoint_name (str): name of the checkpoint (e.g. 'ckpt-12800`).
            If None, the latest checkpoint under train_dir will be used.
        epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout.
        num_episodes (int): number of episodes to play
        sleep_time_per_step (float): sleep so many seconds for each step
        record_file (str): if provided, video will be recorded to a file
            instead of shown on the screen.
        use_tf_functions (bool): whether to use tf.function
    """
    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')

    global_step = alf.summary.get_global_counter()

    ckpt_dir = os.path.join(train_dir, 'algorithm')
    checkpointer = Checkpointer(
        ckpt_dir=ckpt_dir,
        algorithm=algorithm,
        metrics=metrics,
        global_step=global_step)
    checkpointer.load(checkpoint_name)

    recorder = None
    if record_file is not None:
        recorder = VideoRecorder(env.pyenv.envs[0], path=record_file)
    else:
        # pybullet_envs need to render() before reset() to enable mode='human'
        env.pyenv.envs[0].render(mode='human')
    env.reset()
    if recorder:
        recorder.capture_frame()
    time_step = get_initial_time_step(env)
    policy_state = algorithm.get_initial_predict_state(env.batch_size)
    episode_reward = 0.
    episode_length = 0
    episodes = 0
    while episodes < num_episodes:
        time_step, policy_state = _step(
            algorithm=algorithm,
            env=env,
            time_ste=time_step,
            policy_state=policy_state,
            epsilon_greedy=epsilon_greedy,
            metrics=[])
        if recorder:
            recorder.capture_frame()
        else:
            env.pyenv.envs[0].render(mode='human')
            time.sleep(sleep_time_per_step)

        episode_reward += float(time_step.reward)

        if time_step.is_last():
            logging.info("episode_length=%s episode_reward=%s" %
                         (episode_length, episode_reward))
            episode_reward = 0.
            episode_length = 0.
            episodes += 1
        else:
            episode_length += 1
    if recorder:
        recorder.close()
    env.reset()
