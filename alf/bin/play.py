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
r"""Play a trained model.

You can visualize playing of the trained model by running:
```bash
cd ${PROJECT}/alf/examples;
python -m alf.bin.play \
  --root_dir=~/tmp/cart_pole \
  --alsologtostderr
```

"""

from absl import app
from absl import flags
from absl import logging
import copy
import gin
import os
import subprocess
import sys

import torch

from alf.algorithms.data_transformer import create_data_transformer
from alf.environments.utils import create_environment
from alf.trainers import policy_trainer
from alf.utils import common
import alf.utils.external_configurables

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer(
    'checkpoint_step', None, "the number of training steps which is used to "
    "specify the checkpoint to be loaded. If None, the latest checkpoint under "
    "train_dir will be used.")
flags.DEFINE_float('epsilon_greedy', 0., "probability of sampling action.")
flags.DEFINE_integer('random_seed', None, "random seed")
flags.DEFINE_integer('num_episodes', 10, "number of episodes to play")
flags.DEFINE_integer('max_episode_length', 0,
                     "If >0,  each episode is limited "
                     "to so many steps")
flags.DEFINE_integer(
    'future_steps', 0, "If >0, display information from so many "
    "number of future steps in addition to the current step "
    "on the current frame. Otherwise only information from the "
    "current step will be displayed.")
flags.DEFINE_integer(
    'append_blank_frames', 0,
    "If >0, wil append such number of blank frames at the "
    "end of each episode in the rendered video file.")
flags.DEFINE_float('sleep_time_per_step', 0.01,
                   "sleep so many seconds for each step")
flags.DEFINE_string(
    'record_file', None, "If provided, video will be recorded"
    "to a file instead of shown on the screen.")
# use '--norender' to disable frame rendering
flags.DEFINE_bool('render', True,
                  "Whether render ('human'|'rgb_array') the frames or not")
# use '--render_prediction' to enable pred info rendering
flags.DEFINE_bool('render_prediction', False,
                  "Whether render prediction info at every frame or not")
flags.DEFINE_string('gin_file', None, 'Path to the gin-config file.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')
flags.DEFINE_string('conf', None, 'Path to the alf config file.')
flags.DEFINE_multi_string('conf_param', None, 'Config binding parameters.')
flags.DEFINE_string(
    'ignored_parameter_prefixes', "",
    "Comma separated strings to ingore the parameters whose name has one of "
    "these prefixes in the checkpoint.")
flags.DEFINE_bool(
    'use_alf_snapshot', False,
    'Whether to use ALF snapshot stored in the model dir (if any). You can set '
    'this flag to play a model trained with legacy ALF code.')

FLAGS = flags.FLAGS


def play():
    if torch.cuda.is_available():
        alf.set_default_device("cuda")

    seed = common.set_random_seed(FLAGS.random_seed)
    alf.config('create_environment', nonparallel=True)
    alf.config('TrainerConfig', mutable=False, random_seed=seed)
    conf_file = common.get_conf_file()
    assert conf_file is not None, "Conf file not found! Check your root_dir"
    try:
        common.parse_conf_file(conf_file)
    except Exception as e:
        alf.close_env()
        raise e
    config = policy_trainer.TrainerConfig(root_dir="")

    env = alf.get_env()
    env.reset()
    data_transformer = create_data_transformer(config.data_transformer_ctor,
                                               env.observation_spec())
    config.data_transformer = data_transformer

    # keep compatibility with previous gin based config
    common.set_global_env(env)
    observation_spec = data_transformer.transformed_observation_spec
    common.set_transformed_observation_spec(observation_spec)

    algorithm_ctor = config.algorithm_ctor
    algorithm = algorithm_ctor(
        observation_spec=observation_spec,
        action_spec=env.action_spec(),
        reward_spec=env.reward_spec(),
        config=config)
    try:
        policy_trainer.play(
            FLAGS.root_dir,
            env,
            algorithm,
            checkpoint_step=FLAGS.checkpoint_step or "latest",
            epsilon_greedy=FLAGS.epsilon_greedy,
            num_episodes=FLAGS.num_episodes,
            max_episode_length=FLAGS.max_episode_length,
            sleep_time_per_step=FLAGS.sleep_time_per_step,
            record_file=FLAGS.record_file,
            future_steps=FLAGS.future_steps,
            append_blank_frames=FLAGS.append_blank_frames,
            render=FLAGS.render,
            render_prediction=FLAGS.render_prediction,
            ignored_parameter_prefixes=FLAGS.ignored_parameter_prefixes.split(
                ",") if FLAGS.ignored_parameter_prefixes else [])
    finally:
        alf.close_env()


def launch_snapshot_play():
    """This play function uses historical ALF snapshot for playing a trained
    model, consistent with the code snapshot that trains the model.

    In the newer version of ``train.py``, a ALF snapshot is saved to ``root_dir``
    right before the training begins. So this function prepends ``root_dir`` to
    ``PYTHONPATH`` to allow using the snapshot ALF repo in that place.

    Note that for any old training ``root_dir`` prior to snapshot being enabled,
    this function doesn't have any effect and the most up-to-date ALF will
    be used by play.
    """
    root_dir = os.path.expanduser(FLAGS.root_dir)
    alf_repo = os.path.join(root_dir, "alf")

    env_vars = common.get_alf_snapshot_env_vars(root_dir)

    flags = sys.argv[1:]
    flags.append('--nouse_alf_snapshot')

    args = ['python', '-m', 'alf.bin.play'] + flags
    try:
        if os.path.isdir(alf_repo):
            logging.info("=== Using an ALF snapshot at '%s' ===" % alf_repo)
        else:
            logging.info(
                "=== Didn't find a snapshot; using update-to-date ALF ===")
        subprocess.check_call(
            " ".join(args),
            env=env_vars,
            stdout=sys.stdout,
            stderr=sys.stdout,
            shell=True)
    except subprocess.CalledProcessError as e:
        # No need to output anything
        pass


def main(_):
    if not FLAGS.use_alf_snapshot:
        play()
    else:
        launch_snapshot_play()


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    logging.set_verbosity(logging.INFO)
    app.run(main)
