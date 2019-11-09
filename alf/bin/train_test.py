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

import tempfile
import os
import subprocess
from pathlib import Path
import numpy as np
import sys
import time

from absl import logging
import tensorflow as tf

from tensorboard.backend.event_processing import event_file_loader
from tensorboard.util import tensor_util


def run_and_stream(cmd, cwd):
    """Run cmd in a new process and stream its stdout and stderr output

    Args:
        cmd (list[str]): command and args to run
        cwd (str): working directory for the process
    """
    logging.info("Running %s", " ".join(cmd))

    process = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=sys.stderr, cwd=cwd)

    while process.poll() is None:
        time.sleep(0.1)

    assert process.returncode == 0, ("cmd: {0} exit abnormally".format(
        " ".join(cmd)))


def get_metrics_from_eval_tfevents(eval_dir):
    """Get metrics from tfevents in eval dir

    Args:
        eval_dir (str):

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
                    tensor = tensor_util.make_ndarray(item.tensor)
                    episode_returns.append(float(tensor))
                elif item.tag == 'Metrics/AverageEpisodeLength':
                    tensor = tensor_util.make_ndarray(item.tensor)
                    episode_lengths.append(int(tensor))

    assert len(episode_returns) > 0
    logging.info("Episode returns, %s, episode lengths: %s", episode_returns,
                 episode_lengths)
    return episode_returns, episode_lengths


class TrainTest(tf.test.TestCase):
    def test_ac_cart_pole(self):
        def _test_func(returns, lengths):
            self.assertGreater(np.mean(returns[-2:]), 198)
            self.assertGreater(np.mean(lengths[-2:]), 198)

        self._test_train('ac_cart_pole.gin', _test_func)

    def test_ppo_cart_pole(self):
        def _test_func(returns, lengths):
            self.assertGreater(np.mean(returns[-2:]), 198)
            self.assertGreater(np.mean(lengths[-2:]), 198)

        self._test_train('ppo_cart_pole.gin', _test_func)

    def test_ddpg_pendulum(self):
        test_func = lambda returns, lengths: self.assertGreater(
            np.mean(returns[-5:]), -200)
        self._test_train('ddpg_pendulum.gin', test_func)

    def test_sac_pendulum(self):
        test_func = lambda returns, lengths: self.assertGreater(
            np.mean(returns[-5:]), -200)
        self._test_train('sac_pendulum.gin', test_func)

    def _test_train(self, conf_file, assert_func):
        bin_dir = Path(os.path.abspath(__file__)).parent
        examples_dir = os.path.join(Path(bin_dir).parent, 'examples')
        with tempfile.TemporaryDirectory() as root_dir:
            eval_dir = os.path.join(root_dir, 'eval')
            cmd = [
                'xvfb-run', 'python3', '-m', 'alf.bin.train',
                '--root_dir=%s' % root_dir,
                '--gin_file=%s' % conf_file,
                '--gin_param=TrainerConfig.random_seed=0'
            ]
            run_and_stream(cmd=cmd, cwd=examples_dir)
            episode_returns, episode_lengths = get_metrics_from_eval_tfevents(
                eval_dir)
            assert_func(episode_returns, episode_lengths)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    from alf.utils.common import set_per_process_memory_growth

    set_per_process_memory_growth()
    tf.test.main()
