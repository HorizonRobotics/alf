# Copyright (c) 2021 Horizon Robotics. All Rights Reserved.
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
"""Test the training and playing of the alf conf files under alf/examples."""

from absl import logging
import os
import numpy as np
from pathlib import Path
import subprocess
import sys
import tempfile
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import unittest
from unittest import SkipTest

import alf

SKIP_TODO_MESSAGE = "TODO: convert and test this case to alf conf!"


def run_cmd(cmd, cwd=None):
    """Run cmd in a new process and check its exit status

    Args:
        cmd (list[str]): command and args to run
        cwd (str): working directory for the process
    """

    def format_error_message(cmd: list, stdout: str, stderr: str):
        cmd_inline = ' '.join(cmd)
        return f'\ncmd: {cmd_inline} exit abnormally, with\n' \
            f'OUT: {stdout}\n' \
            f'ERR: {stderr}'

    new_env = os.environ.copy()

    ret = subprocess.run(
        cmd,
        cwd=cwd,
        env=new_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    assert ret.returncode == 0, format_error_message(cmd, ret.stdout,
                                                     ret.stderr)


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
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()
    ret_events = event_acc.Scalars('Metrics/AverageReturn')
    episode_returns = [e.value for e in ret_events]
    len_events = event_acc.Scalars('Metrics/AverageEpisodeLength')
    episode_lengths = [e.value for e in len_events]

    assert len(episode_returns) > 0
    logging.info("Episode returns, %s, episode lengths: %s", episode_returns,
                 episode_lengths)
    return episode_returns, episode_lengths


def get_examples_dir():
    """Get examples directory"""
    bin_dir = Path(os.path.abspath(__file__)).parent
    examples_dir = os.path.join(Path(bin_dir).parent, 'examples')
    return examples_dir


def _to_conf_params(parameters):
    """A helper function that convert key-value parameters to conf parameters"""
    # TODO: remove --gin_param after all conf files are converted
    return ['--conf_param=%s' % e for e in parameters] \
        + ['--gin_param=%s' % e for e in parameters]


BASE_TRAIN_CONF = [
    # Do not get stuck there and asking for confirmation upon crash.
    # Die and give feedback immediately for unit tests.
    'confirm_checkpoint_upon_crash=False',
]
BASE_TRAIN_PARAMS = _to_conf_params(BASE_TRAIN_CONF)

COMMON_TRAIN_CONF = BASE_TRAIN_CONF + [
    # create only one env
    'create_environment.num_parallel_environments=1',
    # disable summaries
    'TrainerConfig.debug_summaries=False',
    'TrainerConfig.summarize_grads_and_vars=False',
    'TrainerConfig.summarize_action_distributions=False',
    # train two iterations
    'TrainerConfig.num_iterations=3',
    'TrainerConfig.num_env_steps=0',
    # only save checkpoint after train iteration finished
    'TrainerConfig.num_checkpoints=1',
    # disable evaluate
    'TrainerConfig.evaluate=False',
]
COMMON_TRAIN_PARAMS = _to_conf_params(COMMON_TRAIN_CONF)

ON_POLICY_TRAIN_CONF = COMMON_TRAIN_CONF + [
    'TrainerConfig.unroll_length=4',
]
ON_POLICY_TRAIN_PARAMS = _to_conf_params(ON_POLICY_TRAIN_CONF)

OFF_POLICY_TRAIN_CONF = COMMON_TRAIN_CONF + [
    # Make sure initial_collect_steps <= (num_iterations - 1) * unroll_length * num_parallel_environments
    # so there are some real training
    'TrainerConfig.unroll_length=2',
    'TrainerConfig.initial_collect_steps=2',
    'TrainerConfig.num_updates_per_train_iter=1',
    'TrainerConfig.mini_batch_length=2',
    'TrainerConfig.mini_batch_size=4',
    'TrainerConfig.replay_buffer_length=64',
    'FrameStacker.stack_size=1'
]
OFF_POLICY_TRAIN_PARAMS = _to_conf_params(OFF_POLICY_TRAIN_CONF)

MUZERO_TRAIN_CONF = COMMON_TRAIN_CONF + [
    # Make sure initial_collect_steps <= (num_iterations - 1) * unroll_length * num_parallel_environments
    # so there are some real training
    'TrainerConfig.unroll_length=2',
    'TrainerConfig.initial_collect_steps=2',
    'TrainerConfig.num_updates_per_train_iter=1',
    'TrainerConfig.mini_batch_size=4',
    'TrainerConfig.replay_buffer_length=64',
    'TrainerConfig.whole_replay_buffer_training=False',
    'TrainerConfig.clear_replay_buffer=False',
]
MUZERO_TRAIN_PARAMS = _to_conf_params(MUZERO_TRAIN_CONF)

ON_POLICY_ALG_OFF_POLICY_TRAIN_CONF = OFF_POLICY_TRAIN_CONF + [
    'TrainerConfig.unroll_length=2',
    'TrainerConfig.initial_collect_steps=0',
]
ON_POLICY_ALG_OFF_POLICY_TRAIN_PARAMS = _to_conf_params(
    ON_POLICY_ALG_OFF_POLICY_TRAIN_CONF)

PPO_TRAIN_CONF = OFF_POLICY_TRAIN_CONF + [
    'TrainerConfig.unroll_length=2', 'TrainerConfig.initial_collect_steps=0',
    'TrainerConfig.num_updates_per_train_iter=2'
]
PPO_TRAIN_PARAMS = _to_conf_params(PPO_TRAIN_CONF)

PPO_RND_ATARI_TRAIN_CONF = COMMON_TRAIN_CONF + [
    # Make sure initial_collect_steps <= (num_iterations - 1) *
    # unroll_length * num_parallel_environments so there are some real
    # training
    'TrainerConfig.unroll_length=2',
    'TrainerConfig.initial_collect_steps=3',
    'TrainerConfig.num_updates_per_train_iter=1',
    'TrainerConfig.mini_batch_length=2',
    'TrainerConfig.mini_batch_size=4',
    'TrainerConfig.replay_buffer_length=64',
]
PPO_RND_ATARI_TRAIN_PARAMS = _to_conf_params(PPO_RND_ATARI_TRAIN_CONF)

MBRL_TRAIN_CONF = OFF_POLICY_TRAIN_CONF + [
    'TrainerConfig.unroll_length=4',
    'TrainerConfig.whole_replay_buffer_training=True',
    'TrainerConfig.clear_replay_buffer=False',
]
MBRL_TRAIN_PARAMS = _to_conf_params(MBRL_TRAIN_CONF)

# Run COMMAND in a virtual X server environment
XVFB_RUN = ['xvfb-run', '-a', '-e', '/dev/stderr']


class TrainPlayTest(alf.test.TestCase):
    """Train and play test for alf conf examples located in directory
    `$PROJECT_ROOT/alf/examples`

    NOTE: It's not reasonable to train all the examples until they reaches
    desired performance, we just test if they can run and play correctly
    with minimal configuration (train a few iterations, play a few steps,
    disable summary ...) for most of the examples.
    """

    # They are common configuration files, not complete test, exclude them
    _excluded_ = {
        'carla_conf.py',
        'sac_conf.py',
        'carla.gin',
        'sac.gin',
        'atari_conf.py',
    }

    # All alf conf files list in directory `$PROJECT_ROOT/alf/examples`
    _all_ = set()

    _tested_ = set()

    _skipped_ = set()

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        examples_dir = get_examples_dir()
        for root, dirs, files in os.walk(examples_dir):
            for filename in files:
                if filename.endswith(('_conf.py', '.gin')):
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
        from alf.environments import suite_robotics
        # import other mujoco suites here
        if not suite_robotics.is_available():
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

    def _skip_if_carla_unavailable(self):
        from alf.environments import suite_carla
        if not suite_carla.is_available():
            self.skipTest("Carla is not available.")

    def _skip_if_safety_gym_unavailable(self):
        from alf.environments import suite_safety_gym
        if not suite_safety_gym.is_available():
            self.skipTest("Safety Gym is not available.")

    def _skip_if_dmc_unavailable(self):
        from alf.environments import suite_dmc
        if not suite_dmc.is_available():
            self.skipTest("DM control is not available.")

    def _test(self,
              conf_file,
              skip_checker=None,
              extra_train_params=BASE_TRAIN_PARAMS,
              test_play=True,
              test_video_recording=False,
              extra_play_params=None,
              test_perf=True,
              test_perf_func=None):
        """Test train, play and check performance

        Args:
            conf_file (str): Path to the gin or alf conf file.
            skip_checker (Callable|list[Callable]): callables that will raise a `SkipTest`
                exception when the test is not available.
            extra_train_params (list[str]): extra params used for training.
            test_play (bool): A bool for test play.
            test_video_recording (bool): whether test video recording for play.
            extra_play_params (list[str]): extra param used for play.
            test_perf (bool): A bool for check performance.
            test_perf_func (Callable): called as test_perf_func(episode_returns, episode_lengths)
                which checks whether `episode_returns` and `episode_lengths` meet expectations
                performance.
        """
        # Print which test function is calling this _test
        logging.info(self.id())

        skip_checker = skip_checker or []
        if not isinstance(skip_checker, list):
            skip_checker = [skip_checker]
        try:
            for checker in skip_checker:
                checker()
        except SkipTest:
            self.markSkipped(conf_file)
            raise

        self.markTested(conf_file)
        with tempfile.TemporaryDirectory() as root_dir:
            self._test_train(conf_file, extra_train_params, root_dir)
            if test_play:
                self._test_play(root_dir, extra_play_params,
                                test_video_recording)
            if test_perf and test_perf_func:
                self._test_performance(root_dir, test_perf_func)

    def _test_train(self, conf_file, extra_params, root_dir):
        """Test if the training configured by ``conf_file`` and ``extra_params``
        can run successfully.

        Args:
            conf_file (str): Path to the gin or alf conf file.
            extra_params (list[str]): extra parameters used for training
            root_dir (str): Root directory for writing logs/summaries/checkpoints.
        """
        examples_dir = get_examples_dir()
        cmd = [
            'python3',
            '-m',
            'alf.bin.train',
            '--nostore_snapshot',
            '--root_dir=%s' % root_dir,
            '--conf_param=TrainerConfig.random_seed=1',
            '--gin_param=TrainerConfig.random_seed=1'  # TODO: remove --gin_param
        ]
        if conf_file.endswith('.gin'):
            cmd.append('--gin_file=%s' % conf_file)
        else:
            cmd.append('--conf=%s' % conf_file)
        if 'DISPLAY' not in os.environ:
            cmd = XVFB_RUN + cmd
        cmd.extend(extra_params or [])
        run_cmd(cmd=cmd, cwd=examples_dir)

    def _test_play(self, root_dir, extra_params, test_video_recording):
        """Test if it can play successfully using configuration and checkpoints
        saved in ``root_dir``.

        Args:
            root_dir (str): Root directory where configuration and checkpoints are saved
            extra_params (list[str]): extra parameters used for play
            test_video_recording (bool): if True, also test if a video can be
                recorded.
        """
        examples_dir = get_examples_dir()
        cmd = [
            'python3', '-m', 'alf.bin.play',
            '--root_dir=%s' % root_dir, '--num_episodes=1'
        ]
        if test_video_recording:
            cmd.append('--record_file=%s/play.mp4' % root_dir)
        if 'DISPLAY' not in os.environ:
            cmd = XVFB_RUN + cmd
        cmd.extend(extra_params or [])
        run_cmd(cmd=cmd, cwd=examples_dir)

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
            conf_file='ac_breakout_conf.py',
            skip_checker=self._skip_if_atari_unavailable,
            extra_train_params=ON_POLICY_TRAIN_PARAMS)

    def test_ac_cart_pole(self):
        def _test_func(returns, lengths):
            self.assertGreater(returns[-1], 195)
            self.assertGreater(lengths[-1], 195)

        self._test(
            conf_file='ac_cart_pole_conf.py',
            test_perf_func=_test_func,
            test_video_recording=True)

    def test_ac_simple_navigation(self):
        self._test(
            conf_file='ac_simple_navigation.gin',
            skip_checker=self._skip_if_socialbot_unavailable,
            extra_train_params=ON_POLICY_TRAIN_PARAMS)

    def test_sac_breakout(self):
        self._test(
            conf_file='sac_breakout_conf.py',
            skip_checker=self._skip_if_atari_unavailable,
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_her_target_navigation(self):
        self._test(
            conf_file='her_target_navigation_states.gin',
            skip_checker=self._skip_if_socialbot_unavailable,
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_ddpg_pendulum(self):
        def _test_func(returns, lengths):
            self.assertGreater(returns[-1], -200)

        self._test(
            conf_file='ddpg_pendulum_conf.py', test_perf_func=_test_func)

    def test_ddpg_fetchslide(self):
        self._test(
            conf_file="ddpg_fetchslide_conf.py",
            skip_checker=self._skip_if_mujoco_unavailable,
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_her_fetchpush(self):
        self._test(
            conf_file='her_fetchpush_conf.py',
            skip_checker=self._skip_if_mujoco_unavailable,
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_diayn_pendulum(self):
        self._test(
            conf_file='diayn_pendulum.gin',
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_dyna_actrepeat_sac_bipedal_walker(self):
        self._test(
            conf_file='dyna_actrepeat_sac_bipedalwalker_conf.py',
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_bipedal_walker_w_capacity_schedule(self):
        self._test(
            conf_file='sac_bipedal_walker_capacity_schedule_conf.py',
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_dyna_actrepeat_sac_pickplace(self):
        self._test(
            conf_file="dyna_actrepeat_sac_pickplace.gin",
            skip_checker=self._skip_if_mujoco_unavailable,
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_icm_mountain_car(self):
        self._test(
            conf_file='icm_mountain_car.gin',
            extra_train_params=ON_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_icm_playground(self):
        self._test(
            conf_file='icm_playground.gin',
            skip_checker=self._skip_if_socialbot_unavailable,
            extra_train_params=ON_POLICY_TRAIN_PARAMS)

    def test_icm_super_mario(self):
        self._test(
            conf_file='icm_super_mario.gin',
            skip_checker=self._skip_if_mario_unavailable,
            extra_train_params=ON_POLICY_TRAIN_PARAMS)

    def test_icm_super_mario_intrinsic_only(self):
        self._test(
            conf_file="icm_super_mario_intrinsic_only.gin",
            skip_checker=self._skip_if_mario_unavailable,
            extra_train_params=ON_POLICY_TRAIN_PARAMS)

    def test_mbrl_pendulum(self):
        self._test(
            conf_file='mbrl_pendulum_conf.py',
            extra_train_params=MBRL_TRAIN_PARAMS)

    @unittest.skip("Segfault at the end of a successful training.")
    def test_mbrl_latent_pendulum(self):
        self._test(
            conf_file='mbrl_latent_pendulum_conf.py',
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_mdq_pendulum(self):
        self._test(
            conf_file='mdq_pendulum.gin',
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_mdq_halfcheetah(self):
        self._test(
            conf_file="mdq_halfcheetah.gin",
            skip_checker=self._skip_if_mujoco_unavailable,
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_merlin_dmlab_collect_good_objects(self):
        self._test(
            conf_file='merlin_dmlab_collect_good_objects.gin',
            skip_checker=self._skip_if_dmlab_unavailable,
            test_play=False,  # render mode 'human' is not implemented
            extra_train_params=ON_POLICY_TRAIN_PARAMS)

    def test_monet_bouncing_squares(self):
        self._test(
            conf_file='monet_bouncing_squares_conf.py',
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_muzero_tic_tac_toe(self):
        self._test(
            conf_file='muzero_tic_tac_toe.gin',
            extra_train_params=MUZERO_TRAIN_PARAMS)

    def test_muzero_pendulum(self):
        self._test(
            conf_file='muzero_pendulum_conf.py',
            extra_train_params=MUZERO_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_off_policy_ac_breakout(self):
        self._test(
            conf_file='off_policy_ac_breakout.gin',
            skip_checker=self._skip_if_atari_unavailable,
            extra_train_params=ON_POLICY_ALG_OFF_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_off_policy_ac_cart_pole(self):
        self._test(
            conf_file='off_policy_ac_cart_pole.gin',
            extra_train_params=ON_POLICY_ALG_OFF_POLICY_TRAIN_PARAMS)

    def test_taacl_fetch(self):
        self._test(
            conf_file='taacl_fetch_conf.py',
            skip_checker=self._skip_if_mujoco_unavailable,
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_ppo_bullet_humanoid(self):
        self._test(
            conf_file='ppo_bullet_humanoid.gin',
            skip_checker=self._skip_if_bullet_unavailable,
            extra_train_params=PPO_TRAIN_PARAMS)

    def test_ppo_cart_pole(self):
        def _test_func(returns, lengths):
            self.assertGreater(returns[-1], 195)

        self._test(
            conf_file='ppo_cart_pole_conf.py', test_perf_func=_test_func)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_ppo_icm_super_mario_intrinsic_only(self):
        self._test(
            conf_file='ppo_icm_super_mario_intrinsic_only.gin',
            skip_checker=self._skip_if_mario_unavailable(),
            extra_train_params=PPO_TRAIN_PARAMS)

    def test_ppo_icubwalk(self):
        self._test(
            conf_file='ppo_icubwalk.gin',
            skip_checker=self._skip_if_socialbot_unavailable,
            extra_train_params=PPO_TRAIN_PARAMS)

    def test_ppo_pr2(self):
        self._test(
            conf_file='ppo_pr2.gin',
            skip_checker=self._skip_if_socialbot_unavailable,
            extra_train_params=PPO_TRAIN_PARAMS)

    def test_ppo_rnd_mrevenge(self):
        self._test(
            conf_file='ppo_rnd_mrevenge_conf.py',
            extra_train_params=PPO_RND_ATARI_TRAIN_PARAMS)

    def test_taacq_fetch(self):
        self._test(
            conf_file='taacq_fetch_conf.py',
            skip_checker=self._skip_if_mujoco_unavailable,
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_rnd_super_mario(self):
        self._test(
            conf_file='rnd_super_mario.gin',
            skip_checker=self._skip_if_mario_unavailable,
            extra_train_params=ON_POLICY_TRAIN_PARAMS)

    def test_sac_bipedal_walker(self):
        self._test(
            conf_file='sac_bipedal_walker_conf.py',
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_sac_carla(self):
        self._test(
            conf_file="sac_carla_conf.py",
            skip_checker=self._skip_if_carla_unavailable,
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_sac_cart_pole(self):
        self._test(
            conf_file='sac_cart_pole_conf.py',
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_sac_fetchreach(self):
        self._test(
            conf_file="sac_fetchreach.gin",
            skip_checker=self._skip_if_mujoco_unavailable,
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_sac_fetchslide(self):
        self._test(
            conf_file="sac_fetchslide.gin",
            skip_checker=self._skip_if_mujoco_unavailable,
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_sac_humanoid(self):
        self._test(
            conf_file='sac_humanoid.gin',
            skip_checker=self._skip_if_mujoco_unavailable,
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_sac_lagrw_cargoal1(self):
        self._test(
            conf_file='sac_lagrw_cargoal1_conf.py',
            skip_checker=self._skip_if_safety_gym_unavailable,
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_sac_pendulum(self):
        def _test_func(returns, lengths):
            self.assertGreater(returns[-1], -200)

        self._test(conf_file='sac_pendulum_conf.py', test_perf_func=_test_func)

    def test_sac_pendulum_latent_actor(self):
        def _test_func(returns, lengths):
            self.assertGreater(returns[-1], -200)

        self._test(
            conf_file='sac_pendulum_latent_actor_conf.py',
            test_perf_func=_test_func)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_sac_fishswim(self):
        self._test(
            conf_file='sac_fishswim_conf.py',
            skip_checker=self._skip_if_dmc_unavailable,
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_sarsa_ddpg_pendulum(self):
        self._test(
            conf_file='sarsa_ddpg_pendulum.gin',
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_sarsa_pendulum(self):
        self._test(
            conf_file='sarsa_pendulum.gin',
            extra_train_params=ON_POLICY_TRAIN_PARAMS)

    def test_sarsa_sac_bipedal_walker(self):
        self._test(
            conf_file='sarsa_sac_bipedal_walker.gin',
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_param_reset_sac_bipedal_walker(self):
        self._test(
            conf_file='sac_bipedal_walker_param_reset_conf.py',
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_sarsa_sac_pendulum(self):
        self._test(
            conf_file='sarsa_sac_pendulum.gin',
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_bc_pendulum(self):
        self._test(
            conf_file='./hybrid_rl/bc_pendulum_conf.py',
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_causal_bc_pendulum(self):
        self._test(
            conf_file='./hybrid_rl/causal_bc_pendulum_conf.py',
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_iql_pendulum(self):
        self._test(
            conf_file='./hybrid_rl/iql_pendulum_conf.py',
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_hybrid_sac_pendulum(self):
        self._test(
            conf_file='./hybrid_rl/hybrid_sac_pendulum.gin',
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_hybrid_sac_pendulum_muzero_buffer(self):
        self._test(
            conf_file='./hybrid_rl/hybrid_sac_pendulum_muzero_buffer.gin',
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_taac_bipedal_walker(self):
        self._test(
            conf_file='taac_bipedal_walker_conf.py',
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_taac_fetch(self):
        self._test(
            conf_file='taac_fetch_conf.py',
            skip_checker=self._skip_if_mujoco_unavailable,
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_trac_breakout(self):
        self._test(
            conf_file='trac_breakout.gin',
            skip_checker=self._skip_if_atari_unavailable,
            extra_train_params=ON_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_trac_ddpg_pendulum(self):
        self._test(
            conf_file='trac_ddpg_pendulum.gin',
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_trac_ppo_pr2(self):
        self._test(
            conf_file='trac_ppo_pr2.gin',
            skip_checker=self._skip_if_socialbot_unavailable,
            extra_train_params=PPO_TRAIN_PARAMS)

    @unittest.skip(SKIP_TODO_MESSAGE)
    def test_trac_sac_pendulum(self):
        self._test(
            conf_file='trac_sac_pendulum.gin',
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_oac_halfcheetah(self):
        self._test(
            conf_file='oac_halfcheetah_conf.py',
            skip_checker=self._skip_if_mujoco_unavailable,
            extra_train_params=OFF_POLICY_TRAIN_PARAMS)

    def test_oac_humanoid(self):
        self._test(
            conf_file='oac_humanoid_conf.py',
            skip_checker=self._skip_if_mujoco_unavailable,
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
