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

.. code-block:: bash

    cd ${PROJECT}/alf/examples;
    python -m alf.bin.play \
    --root_dir=~/tmp/cart_pole \
    --alsologtostderr

"""

from absl import app
from absl import flags
from absl import logging
import copy
import inspect
import os
import subprocess
import sys
import torch

from alf.algorithms.data_transformer import create_data_transformer
from alf.environments.utils import create_environment
from alf.trainers import policy_trainer
from alf.utils import common
import alf.summary.render as render
import alf.utils.external_configurables


def _define_flags():
    flags.DEFINE_string(
        'root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
        'Root directory for writing logs/summaries/checkpoints.')
    flags.DEFINE_string(
        'checkpoint_step', 'best',
        "the number of training steps which is used to "
        "specify the checkpoint to be loaded. If 'latest', the latest checkpoint "
        "under train_dir will be used. If 'best', the checkpoint with suffix "
        "'best' will be used if it can be found, otherwise the latest one will "
        "be used.")
    flags.DEFINE_integer('random_seed', None, "random seed")
    flags.DEFINE_bool(
        'force_torch_deterministic', True,
        'torch.use_deterministic_algorithms when random_seed is set. '
        'When it is False, deterministic behavior is not guaranteed, '
        'but could still be deterministic, e.g. for sac_breakout_conf.py. '
        'Setting a random seed without setting this to False, training '
        'could throw this error: _scatter_add kernel does not have a '
        'deterministic implementation.')
    flags.DEFINE_integer('num_episodes', 10, "number of episodes to play")
    flags.DEFINE_integer(
        'last_step_repeats', 0,
        "If >0, will repeat such number of times for the last "
        "frame of each episode in the rendered video file.")
    flags.DEFINE_integer(
        'append_blank_frames', 0,
        "If >0, will append such number of blank frames at the "
        "end of each episode in the rendered video file.")
    flags.DEFINE_float('sleep_time_per_step', 0.01,
                       "sleep so many seconds for each step")
    flags.DEFINE_string(
        'record_file', None, "If provided, video will be recorded"
        "to a file instead of shown on the screen.")
    # use '--norender' to disable frame rendering
    flags.DEFINE_bool(
        'render', True,
        "Whether render ('human'|'rgb_array') the frames or not")
    # use '--alg_render' to enable algorithm specific rendering
    flags.DEFINE_bool('alg_render', False,
                      "Whether enable algorithm specific rendering")
    flags.DEFINE_string('gin_file', None, 'Path to the gin-config file.')
    flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')
    flags.DEFINE_string('conf', None, 'Path to the alf config file.')
    flags.DEFINE_multi_string('conf_param', None, 'Config binding parameters.')
    flags.DEFINE_string(
        'ignored_parameter_prefixes', "",
        "Comma separated strings to ignore the parameters whose name has one of "
        "these prefixes in the checkpoint.")
    flags.DEFINE_bool(
        'use_alf_snapshot', False,
        'Whether to use ALF snapshot stored in the model dir (if any). You can set '
        'this flag to play a model trained with legacy ALF code.')
    flags.DEFINE_integer('parallel_play', 1,
                         'Play so many simulations simultaneously')

    flags.DEFINE_bool(
        'selective_mode', False, "Whether use the selective mode. "
        "If True, only save the discoverted selective cases within"
        "the `num_episodes` of test episode. This mode "
        "should be used together with the video recording mode.")


FLAGS = flags.FLAGS


def play():
    if torch.cuda.is_available():
        alf.set_default_device("cuda")

    render.enable_rendering(FLAGS.alg_render)

    seed = common.set_random_seed(FLAGS.random_seed)
    if FLAGS.parallel_play > 1:
        alf.config(
            'create_environment',
            for_evaluation=True,
            num_parallel_environments=FLAGS.parallel_play,
            mutable=False)
    else:
        alf.config(
            'create_environment',
            for_evaluation=True,
            nonparallel=True,
            num_parallel_environments=1,
            batch_size_per_env=1,
            mutable=False)
    alf.config('TrainerConfig', mutable=False, random_seed=seed)
    conf_file = common.get_conf_file()
    assert conf_file is not None, "Conf file not found! Check your root_dir"
    try:
        common.parse_conf_file(conf_file)
    except Exception as e:
        alf.close_env()
        raise e

    if FLAGS.selective_mode:
        assert FLAGS.record_file is not None, ("Should provide a valid value "
                                               "for `record_file`")

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
    signature = inspect.signature(algorithm_ctor)
    kwargs = {
        'observation_spec': observation_spec,
        'action_spec': env.action_spec(),
        'config': config
    }
    # only the algorithms derived from ``RLAlgorithm`` need reward_spec
    if 'reward_spec' in signature.parameters:
        kwargs['reward_spec'] = env.reward_spec()

    algorithm = algorithm_ctor(**kwargs)

    algorithm.set_path('')

    if FLAGS.checkpoint_step is not None:
        try:
            FLAGS.checkpoint_step = int(FLAGS.checkpoint_step)
        except ValueError:
            pass
    try:
        policy_trainer.play(
            common.abs_path(FLAGS.root_dir),
            env,
            algorithm,
            checkpoint_step=FLAGS.checkpoint_step,
            num_episodes=FLAGS.num_episodes,
            sleep_time_per_step=FLAGS.sleep_time_per_step,
            record_file=FLAGS.record_file,
            append_blank_frames=FLAGS.append_blank_frames,
            last_step_repeats=FLAGS.last_step_repeats,
            render=FLAGS.render,
            selective_mode=FLAGS.selective_mode,
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
    # assert the current path is not ALF_ROOT because sys.path will always prepend
    # the current path to the path list, which makes our snapshot ALF path shadowed
    root_dir = common.abs_path(FLAGS.root_dir)

    env_vars = common.get_alf_snapshot_env_vars(root_dir)

    flags = sys.argv[1:]
    flags.append('--nouse_alf_snapshot')

    args = ['python', '-m', 'alf.bin.play'] + flags
    try:
        subprocess.check_call(
            args,
            env=env_vars,
            stdout=sys.stdout,
            stderr=sys.stdout,
            shell=False)
    except subprocess.CalledProcessError:
        # No need to output anything
        pass


def main(_):
    if not FLAGS.use_alf_snapshot:
        play()
    else:
        launch_snapshot_play()


if __name__ == '__main__':
    _define_flags()
    flags.mark_flag_as_required('root_dir')
    logging.set_verbosity(logging.INFO)
    app.run(main)
