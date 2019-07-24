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
import sys
import time
import abc
from absl import logging
import gin.tf
import tensorflow as tf
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from tf_agents.eval import metric_utils
from tf_agents.utils import common as tfa_common

from alf.drivers.on_policy_driver import OnPolicyDriver
from alf.utils.metric_utils import eager_compute
from tf_agents.metrics import tf_metrics
from alf.utils import common
from alf.utils.common import run_under_record_context, get_global_counter
from alf.environments.utils import create_environment


@gin.configurable
class TrainerConfig(object):
    def __init__(self, trainer_cls, **trainer_kwargs):
        """

        Args:
            trainer_cls (class): cls that used for creating a `Trainer` instance,
                and it should subclass of `Trainer`
            trainer_kwargs (dict): kwargs that used for construct trainer,
                see `Trainer` and its subclasses for details
        """
        self._trainer_cls = trainer_cls
        self._trainer_kwargs = trainer_kwargs

    def create_trainer(self, root_dir):
        """Create a policy trainer from config

        Args:
            root_dir (str):  directory for saving summary and checkpoints
        Returns:
            An instance of `PolicyTrainer`
        """
        return self._trainer_cls(root_dir=root_dir, **self._trainer_kwargs)


class Trainer(object):
    def __init__(self,
                 root_dir,
                 algorithm_ctor=None,
                 random_seed=0,
                 num_iterations=1000,
                 use_tf_functions=True,
                 checkpoint_interval=1000,
                 evaluate=False,
                 eval_interval=10,
                 num_eval_episodes=10,
                 summary_interval=50,
                 summaries_flush_secs=1,
                 summary_max_queue=10,
                 debug_summaries=False,
                 summarize_grads_and_vars=False):
        """Abstract base class for on-policy and off-policy trainer

        Args:
            root_dir (str): directory for saving summary and checkpoints
            algorithm_ctor (Callable): callable that create an
                `OffPolicyAlgorithm` or `OnPolicyAlgorithm` instance
            random_seed (int): random seed
            num_iterations (int): number of update iterations
            use_tf_functions (bool): whether to use tf.function
            checkpoint_interval (int): checkpoint every so many iterations
            evaluate (bool): A bool to evaluate when training
            eval_interval (int): evaluate every so many iteration
            num_eval_episodes (int) : number of episodes for one evaluation
            summary_interval (int): write summary every so many training steps
            summaries_flush_secs (int): flush summary to disk every so many seconds
            summary_max_queue (int): flush to disk every so mary summaries
            debug_summaries (bool): A bool to gather debug summaries.
            summarize_grads_and_vars (bool): If True, gradient and network variable
                summaries will be written during training.
        """
        root_dir = os.path.expanduser(root_dir)
        self._train_dir = os.path.join(root_dir, 'train')
        self._eval_dir = os.path.join(root_dir, 'eval')

        self._env = None
        self._algorithm_ctor = algorithm_ctor
        self._algorithm = None
        self._driver = None

        self._random_seed = random_seed
        self._num_iterations = num_iterations
        self._use_tf_functions = use_tf_functions

        self._checkpoint_interval = checkpoint_interval
        self._checkpointer = None

        self._evaluate = evaluate
        self._eval_interval = eval_interval
        self._num_eval_episodes = num_eval_episodes
        eval_metrics = None
        eval_summary_writer = None
        if self._evaluate:
            eval_metrics = [
                tf_metrics.AverageReturnMetric(
                    buffer_size=self._num_eval_episodes),
                tf_metrics.AverageEpisodeLengthMetric(
                    buffer_size=self._num_eval_episodes)
            ]
            eval_summary_writer = tf.summary.create_file_writer(
                self._eval_dir, flush_millis=summaries_flush_secs * 1000)
        self._eval_env = None
        self._eval_summary_writer = eval_summary_writer
        self._eval_metrics = eval_metrics

        self._summary_interval = summary_interval
        self._summaries_flush_mills = summaries_flush_secs * 1000
        self._summary_max_queue = summary_max_queue
        self._debug_summaries = debug_summaries
        self._summarize_grads_and_vars = summarize_grads_and_vars

    def initialize(self):
        """Initializes the Trainer."""

        tf.random.set_seed(self._random_seed)
        tf.config.experimental_run_functions_eagerly(
            not self._use_tf_functions)
        self._env = create_environment()
        if self._evaluate:
            self._eval_env = create_environment(num_parallel_environments=1)
        self._algorithm = self._algorithm_ctor(
            self._env, debug_summaries=self._debug_summaries)
        self._driver = self.init_driver()

    @abc.abstractmethod
    def init_driver(self):
        """Initialize driver

        Sub class should implement this method and return an instance of `PolicyDriver`

        Returns:
            driver (PolicyDriver): driver used for collecting data and training
        """
        pass

    def train(self):
        """Perform training"""
        assert None not in (self._env, self._algorithm,
                            self._driver), "Trainer not initialized"
        self._restore_checkpoint()
        run_under_record_context(
            self._train,
            summary_dir=self._train_dir,
            summary_interval=self._summary_interval,
            flush_millis=self._summaries_flush_mills,
            summary_max_queue=self._summary_max_queue)
        self._save_checkpoint()
        if self._evaluate:
            self._eval_env.pyenv.close()
        self._env.pyenv.close()

    @abc.abstractmethod
    def train_iter(self, iter_num, policy_state, time_step):
        """Perform one training iteration.

        Args:
            iter_num (int): iteration number
            policy_state (nested Tensor): should be consistent with train_state_spec
            time_step (ActionTimeStep): initial time_step
        Returns:
            policy_state (nested Tensor): final step policy state.
            time_step (ActionTimeStep): named tuple with final observation, reward, etc.
        """
        pass

    def _train(self):
        self._env.reset()
        time_step = self._driver.get_initial_time_step()
        policy_state = self._driver.get_initial_policy_state()
        for iter_num in range(self._num_iterations):
            t0 = time.time()
            time_step, policy_state = self.train_iter(
                iter_num=iter_num,
                policy_state=policy_state,
                time_step=time_step)
            logging.info('%s time=%.3f' % (iter_num, time.time() - t0))
            if (iter_num + 1) % self._checkpoint_interval == 0:
                self._save_checkpoint()
            if self._evaluate and (iter_num + 1) % self._eval_interval == 0:
                self._eval()
            if iter_num == 0:
                with tf.summary.record_if(True):
                    common.summarize_gin_config()
                    tf.summary.text('commandline', ' '.join(sys.argv))

    def _restore_checkpoint(self):
        global_step = get_global_counter()
        checkpointer = tfa_common.Checkpointer(
            ckpt_dir=os.path.join(self._train_dir, 'algorithm'),
            algorithm=self._algorithm,
            metrics=metric_utils.MetricsGroup(self._driver.get_metrics(),
                                              'metrics'),
            global_step=global_step)
        checkpointer.initialize_or_restore()
        self._checkpointer = checkpointer

    def _save_checkpoint(self):
        global_step = get_global_counter()
        self._checkpointer.save(global_step=global_step.numpy())

    def _eval(self):
        global_step = get_global_counter()
        with tf.summary.record_if(True):
            eager_compute(
                metrics=self._eval_metrics,
                environment=self._eval_env,
                state_spec=self._algorithm.predict_state_spec,
                action_fn=lambda time_step, state: common.algorithm_step(
                    algorithm=self._algorithm,
                    ob_transformer=self._driver.observation_transformer,
                    time_step=time_step,
                    state=state,
                    greedy_predict=True,
                    training=False),
                num_episodes=self._num_eval_episodes,
                step_metrics=self._driver.get_step_metrics(),
                train_step=global_step,
                summary_writer=self._eval_summary_writer,
                summary_prefix="Metrics")
            metric_utils.log_metrics(self._eval_metrics)


@gin.configurable
def play(root_dir,
         env,
         algorithm,
         checkpoint_name=None,
         greedy_predict=True,
         random_seed=0,
         num_episodes=10,
         sleep_time_per_step=0.01,
         record_file=None,
         use_tf_functions=True):
    """Play using the latest checkpoint under `train_dir`.

    The following example record the play of a trained model to a mp4 video:
    ```bash
    python -m alf.bin.main --play \
    --root_dir=~/tmp/bullet_humanoid/ppo2/ppo2-11 \
    --gin_param='on_policy_trainer.play.num_episodes=1' \
    --gin_param='on_policy_trainer.play.record_file="ppo_bullet_humanoid.mp4"'
    ```
    Args:
        root_dir (str): same as the root_dir used for `train()`
        env (TFEnvironment): the environment
        algorithm (OnPolicyAlgorithm): the training algorithm
        checkpoint_name (str): name of the checkpoint (e.g. 'ckpt-12800`).
            If None, the latest checkpoint unber train_dir will be used.
        greedy_predict (bool): use greedy action for evaluation.
        random_seed (int): random seed
        num_episodes (int): number of episodes to play
        sleep_time_per_step (float): sleep so many seconds for each step
        record_file (str): if provided, video will be recorded to a file
            instead of shown on the screen.
        use_tf_functions (bool): whether to use tf.function
    """
    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')

    tf.random.set_seed(random_seed)
    global_step = get_global_counter()

    driver = OnPolicyDriver(
        env=env,
        algorithm=algorithm,
        training=False,
        greedy_predict=greedy_predict)

    ckpt_dir = os.path.join(train_dir, 'algorithm')
    checkpoint = tf.train.Checkpoint(
        algorithm=algorithm,
        metrics=metric_utils.MetricsGroup(driver.get_metrics(), 'metrics'),
        global_step=global_step)
    if checkpoint_name is not None:
        ckpt_path = os.path.join(ckpt_dir, checkpoint_name)
    else:
        ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
    if ckpt_path is not None:
        logging.info("Restore from checkpoint %s" % ckpt_path)
        checkpoint.restore(ckpt_path)
    else:
        logging.info("Checkpoint is not found at %s" % ckpt_dir)

    if not use_tf_functions:
        tf.config.experimental_run_functions_eagerly(True)

    recorder = None
    if record_file is not None:
        recorder = VideoRecorder(env.pyenv.envs[0], path=record_file)
    else:
        # pybullet_envs need to render() before reset() to enable mode='human'
        env.pyenv.envs[0].render(mode='human')
    env.reset()
    if recorder:
        recorder.capture_frame()
    time_step = driver.get_initial_time_step()
    policy_state = driver.get_initial_policy_state()
    episode_reward = 0.
    episode_length = 0
    episodes = 0
    while episodes < num_episodes:
        time_step, policy_state = driver.run(
            max_num_steps=1, time_step=time_step, policy_state=policy_state)
        if recorder:
            recorder.capture_frame()
        else:
            env.pyenv.envs[0].render(mode='human')
            time.sleep(sleep_time_per_step)
        if time_step.is_last():
            logging.info("episode_length=%s episode_reward=%s" %
                         (episode_length, episode_reward))
            episode_reward = 0.
            episode_length = 0.
            episodes += 1
        else:
            episode_reward += float(time_step.reward)
            episode_length += 1
    if recorder:
        recorder.close()
    env.reset()
