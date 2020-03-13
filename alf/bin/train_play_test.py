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
"""Test the training and playing of the gin files under alf/examples."""

from absl import logging
import os
import numpy as np
from pathlib import Path
import subprocess
import sys
import tempfile
from tensorboard.backend.event_processing import event_file_loader
import unittest
from unittest import SkipTest

import alf

SKIP_TODO_MESSAGE = "TODO: convert and test this gin file to pytorch version"


def run_cmd(cmd, cwd=None):
    """Run cmd in a new process and check its exit status

    Args:
        cmd (list[str]): command and args to run
        cwd (str): working directory for the process
    """
    logging.info("Running %s", " ".join(cmd))

    new_env = os.environ.copy()

    process = subprocess.Popen(
        cmd, stdout=sys.stderr, stderr=sys.stderr, cwd=cwd, env=new_env)

    process.communicate()

    assert process.returncode == 0, ("cmd: {0} exit abnormally".format(
        " ".join(cmd)))


def get_metrics_from_eval_tfevents(eval_dir):
    """Get metrics from tfevents in eval dir

    Args:
        eval_dir (str): Root directory where eval summaries are stored

    Returns:
        list[float], list[int]: average returns, and average episode lengths
    """
    event_file = None
    for root, dirs, files in os.walk(eval_dir):
        for file_name in files:
            if "events" in file_name and 'profile' not in file_name:
                event_file = os.path.join(root, file_name)
                break

    assert event_file is not None

    logging.info("Parse event file:%s", event_file)
    episode_returns = []
    episode_lengths = []
    for event_str in event_file_loader.EventFileLoader(event_file).Load():
        if event_str.summary.value:
            for item in event_str.summary.value:
                if item.tag == 'Metrics/AverageReturn':
                    episode_returns.append(item.simple_value)
                elif item.tag == 'Metrics/AverageEpisodeLength':
                    episode_lengths.append(item.simple_value)

    assert len(episode_returns) > 0
    logging.info("Episode returns, %s, episode lengths: %s", episode_returns,
                 episode_lengths)
    return episode_returns, episode_lengths


def get_examples_dir():
    """Get examples directory"""
    bin_dir = Path(os.path.abspath(__file__)).parent
    examples_dir = os.path.join(Path(bin_dir).parent, 'examples')
    return examples_dir


def _to_gin_params(parameters):
    """A helper function that convert key-value parameters to gin parameters"""
    return ['--gin_param=%s' % e for e in parameters]


COMMON_TRAIN_CONF = [
    # create only one env
    'create_environment.num_parallel_environments=1',
    # disable summaries
    'TrainerConfig.debug_summaries=False',
    'TrainerConfig.summarize_grads_and_vars=False',
    'TrainerConfig.summarize_action_distributions=False',
    # train two iterations
    'TrainerConfig.num_iterations=2',
    # minimal env steps
    'TrainerConfig.num_steps_per_iter=1',
    # only save checkpoint after train iteration finished
    'TrainerConfig.num_checkpoints=1',
    # disable evaluate
    'TrainerConfig.evaluate=False',
]
COMMON_TRAIN_PARAMS = _to_gin_params(COMMON_TRAIN_CONF)

ON_POLICY_TRAIN_CONF = COMMON_TRAIN_CONF + [
    'TrainerConfig.unroll_length=4',
]
ON_POLICY_TRAIN_PARAMS = _to_gin_params(ON_POLICY_TRAIN_CONF)

OFF_POLICY_TRAIN_CONF = COMMON_TRAIN_CONF + [
    'TrainerConfig.unroll_length=1',
    'TrainerConfig.initial_collect_steps=8',
    'TrainerConfig.num_updates_per_train_step=1',
    'TrainerConfig.mini_batch_length=2',
    'TrainerConfig.mini_batch_size=4',
    'TrainerConfig.num_envs=2',
    'ReplayBuffer.max_length=64',
]
OFF_POLICY_TRAIN_PARAMS = _to_gin_params(OFF_POLICY_TRAIN_CONF)

ON_POLICY_ALG_OFF_POLICY_TRAIN_CONF = OFF_POLICY_TRAIN_CONF + [
    'TrainerConfig.unroll_length=2',
    'TrainerConfig.initial_collect_steps=0',
]
ON_POLICY_ALG_OFF_POLICY_TRAIN_PARAMS = _to_gin_params(
    ON_POLICY_ALG_OFF_POLICY_TRAIN_CONF)

PPO_TRAIN_CONF = OFF_POLICY_TRAIN_CONF + [
    'TrainerConfig.unroll_length=2', 'TrainerConfig.initial_collect_steps=0',
    'TrainerConfig.num_updates_per_train_step=2'
]
PPO_TRAIN_PARAMS = _to_gin_params(PPO_TRAIN_CONF)

# Run COMMAND in a virtual X server environment
XVFB_RUN = ['xvfb-run', '-a', '-e', '/dev/stderr']


class TrainPlayTest(alf.test.TestCase):
    """Train and play test for examples located in directory
    `$PROJECT_ROOT/alf/examples`

    NOTE: It's not reasonable to train all the examples until they reaches
    desired performance, we just test if they can run and play correctly
    with minimal configuration (train a few iterations, play a few steps,
    disable summary ...) for most of the examples.
    """

    # They are common configuration files, not complete test, exclude them
    _excluded_ = {
        'atari.gin',
        'ppo.gin',
    }

    # All gin files list in directory `$PROJECT_ROOT/alf/examples`
    _all_ = set()

    _tested_ = set()

    _skipped_ = set()

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        examples_dir = get_examples_dir()
        for root, dirs, files in os.walk(examples_dir):
            for filename in files:
                if filename.endswith('.gin'):
                    cls._all_.add(filename)

    @classmethod
    def markTested(cls, test):
        cls._tested_.add(test)

    @classmethod
    def markSkipped(cls, test):
        cls._skipped_.add(test)

    def _skip_if_socialbot_unavailable(self):
        from alf.environments import suite_socialbot
        if not suite_socialbot.is_available():
            self.skipTest("Socialbot env is not available.")

    def _skip_if_mario_unavailable(self):
        from alf.environments import suite_mario
        if not suite_mario.is_available():
            self.skipTest("SuperMario is not available.")

    def _skip_if_mujoco_unavailable(self):
        from tf_agents.environments import suite_mujoco
        if not suite_mujoco.is_available():
            self.skipTest("Mujoco env is not available.")

    def _skip_if_dmlab_unavailable(self):
        from alf.environments import suite_dmlab
        if not suite_dmlab.is_available():
            self.skipTest("DeepMindLab env is not available.")

    def _skip_if_bullet_unavailable(self):
        try:
            import pybullet_envs
        except ImportError:
            self.skipTest("PyBullet env is not available.")

    def _skip_if_atari_unavailable(self):
        try:
            import atari_py
        except ImportError:
            self.skipTest("Atari env is not available.")

    def _test(self,
              gin_file,
              skip_checker=None,
              extra_train_params=None,
              test_play=True,
              extra_play_params=None,
              test_perf=True,
              test_perf_func=None):
        """Test train, play and check performance

        Args:
            gin_file (str): Path to the gin-config file.
            skip_checker (Callable|list[Callable]): callables that will raise a `SkipTest`
                exception when the test is not available.
            extra_train_params (list[str]): extra params used for training.
            test_play (bool): A bool for test play.
            extra_play_params (list[str]): extra param used for play.
            test_perf (bool): A bool for check performance.
            test_perf_func (Callable): called as test_perf_func(episode_returns, episode_lengths)
                which checks whether `episode_returns` and `episode_lengths` meet expectations
                performance.
        """
        skip_checker = skip_checker or []
        if not isinstance(skip_checker, list):
            skip_checker = [skip_checker]
        try:
            for checker in skip_checker:
                checker()
        except SkipTest:
            self.markSkipped(gin_file)
            raise

        self.markTested(gin_file)
        with tempfile.TemporaryDirectory() as root_dir:
            self._test_train(gin_file, extra_train_params, root_dir)
            if test_play:
                self._test_play(root_dir, extra_play_params)
            if test_perf and test_perf_func:
                self._test_performance(root_dir, test_perf_func)

    def _test_train(self, gin_file, extra_params, root_dir):
        """Test if the training configured by gin_file and extra_params can run
        successfully.

        Args:
            gin_file (str): Path to the gin-config file.
            extra_params (list[str]): extra parameters used for training
            root_dir (str): Root directory for writing logs/summaries/checkpoints.
        """
        examples_dir = get_examples_dir()
        cmd = [
            'python3',
            '-m',
            'alf.bin.train',
            '--root_dir=%s' % root_dir,
            '--gin_file=%s' % gin_file,
            '--gin_param=TrainerConfig.random_seed=1',
        ]
        if 'DISPLAY' not in os.environ:
            cmd = XVFB_RUN + cmd
        cmd.extend(extra_params or [])
        run_cmd(cmd=cmd, cwd=examples_dir)

    def _test_play(self, root_dir, extra_params):
        """Test if it can play successfully using configuration and checkpoints
        saved in root_dir.

        Args:
            root_dir (str): Root directory where configuration and checkpoints are saved
            extra_params (list[str]): extra parameters used for play
        """
        cmd = [
            'python3', '-m', 'alf.bin.play',
            '--root_dir=%s' % root_dir, '--num_episodes=1'
        ]
        if 'DISPLAY' not in os.environ:
            cmd = XVFB_RUN + cmd
        cmd.extend(extra_params or [])
        run_cmd(cmd=cmd)

    def _test_performance(self, root_dir, test_func):
        """Test if the performance meet expectations

        Args:
             root_dir (str): Root directory where logs/summaries are saved
             test_func (Callable): called as test_func(episode_returns, episode_lengths)
                which checks whether `episode_returns` and `episode_lengths` meet desired
                performance.
        """
        eval_dir = os.path.join(root_dir, 'eval')
        episode_returns, episode_lengths = get_metrics_from_eval_tfevents(
            eval_dir)
        test_func(episode_returns, episode_lengths)

    def test_ac_breakout(self):
        self._test(
            gin_file='ac_breakout.gin',
            skip_checker=self._skip_if_atari_unavailable,
            extra_train_params=ON_POLICY_TRAIN_PARAMS)

    def test_ac_cart_pole(self):
        def _test_func(returns, lengths):
            self.assertGreater(returns[-1], 195)
            self.assertGreater(lengths[-1], 195)

        self._test(gin_file='ac_cart_pole.gin', test_perf_func=_test_func)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_ac_simple_navigation(self):
        self._test(
            gin_file='ac_simple_navigation.gin',
            skip_checker=self._skip_if_socialbot_unavailable,
            extra_train_params=ON_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_ddpg_pendulum(self):
        def _test_func(returns, lengths):
            self.assertGreater(returns[-1], -200)

        self._test(gin_file='ddpg_pendulum.gin', test_perf_func=_test_func)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_diayn_pendulum(self):
        self._test(
            gin_file='diayn_pendulum.gin',
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_icm_mountain_car(self):
        self._test(
            gin_file='icm_mountain_car.gin',
            extra_train_params=ON_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_icm_playground(self):
        self._test(
            gin_file='icm_playground.gin',
            skip_checker=self._skip_if_socialbot_unavailable,
            extra_train_params=ON_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_icm_super_mario(self):
        self._test(
            gin_file='icm_super_mario.gin',
            skip_checker=self._skip_if_mario_unavailable,
            extra_train_params=ON_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_icm_super_mario_intrinsic_only(self):
        self._test(
            gin_file="icm_super_mario_intrinsic_only.gin",
            skip_checker=self._skip_if_mario_unavailable,
            extra_train_params=ON_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_merlin_dmlab_collect_good_objects(self):
        self._test(
            gin_file='icm_super_mario_intrinsic_only.gin',
            skip_checker=self._skip_if_dmlab_unavailable,
            extra_train_params=ON_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_off_policy_ac_breakout(self):
        self._test(
            gin_file='off_policy_ac_breakout.gin',
            skip_checker=self._skip_if_atari_unavailable,
            extra_train_params=ON_POLICY_ALG_OFF_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_off_policy_ac_cart_pole(self):
        self._test(
            gin_file='off_policy_ac_cart_pole.gin',
            extra_train_params=ON_POLICY_ALG_OFF_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_ppo_bullet_humanoid(self):
        self._test(
            gin_file='ppo_bullet_humanoid.gin',
            skip_checker=self._skip_if_bullet_unavailable,
            extra_train_params=PPO_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_ppo_cart_pole(self):
        def _test_func(returns, lengths):
            self.assertGreater(returns[-1], 195)

        self._test(gin_file='ppo_cart_pole.gin', test_perf_func=_test_func)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_ppo_icm_super_mario_intrinsic_only(self):
        self._test(
            gin_file='ppo_icm_super_mario_intrinsic_only.gin',
            skip_checker=self._skip_if_mario_unavailable(),
            extra_train_params=PPO_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_ppo_icubwalk(self):
        self._test(
            gin_file='ppo_icubwalk.gin',
            skip_checker=self._skip_if_socialbot_unavailable,
            extra_train_params=PPO_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_ppo_pr2(self):
        self._test(
            gin_file='ppo_pr2.gin',
            skip_checker=self._skip_if_socialbot_unavailable,
            extra_train_params=PPO_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_rnd_super_mario(self):
        self._test(
            gin_file='rnd_super_mario.gin',
            skip_checker=self._skip_if_mario_unavailable,
            extra_train_params=ON_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_ppo_rnd_mrevenge(self):
        self._test(
            gin_file='ppo_rnd_mrevenge.gin',
            extra_train_params=PPO_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_sac_bipedal_walker(self):
        self._test(
            gin_file='sac_bipedal_walker.gin',
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_sac_cart_pole(self):
        self._test(
            gin_file='sac_cart_pole.gin',
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_sac_humanoid(self):
        self._test(
            gin_file='sac_humanoid.gin',
            skip_checker=self._skip_if_mujoco_unavailable,
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_sac_pendulum(self):
        def _test_func(returns, lengths):
            self.assertGreater(returns[-1], -200)

        self._test(gin_file='sac_pendulum.gin', test_perf_func=_test_func)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_sarsa_pendulum(self):
        self._test(
            gin_file='sarsa_pendulum.gin',
            extra_train_params=ON_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_trac_breakout(self):
        self._test(
            gin_file='trac_breakout.gin',
            skip_checker=self._skip_if_atari_unavailable,
            extra_train_params=ON_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_trac_ddpg_pendulum(self):
        self._test(
            gin_file='trac_ddpg_pendulum.gin',
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_trac_ppo_pr2(self):
        self._test(
            gin_file='trac_ppo_pr2.gin',
            skip_checker=self._skip_if_socialbot_unavailable,
            extra_train_params=PPO_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_trac_sac_pendulum(self):
        self._test(
            gin_file='trac_sac_pendulum.gin',
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    @classmethod
    def tearDownClass(cls):
        not_tested = cls._all_.difference(cls._tested_)
        missed = not_tested.difference(cls._excluded_.union(cls._skipped_))
        if missed:
            logging.warning(
                'Missing test for [%s], highly recommended to add them to testes.',
                ','.join(list(missed)))

        cls._all_.clear()
        cls._tested_.clear()
        cls._skipped_.clear()
        super().tearDownClass()


if __name__ == '__main__':
    alf.test.main()
