# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
r"""Utility to check whether checkpointed algorithm can be restored correctly.

It works as the following:

1. Save the config.
2. Train the algorithm for a few iterations.
3. Test the algorithm for a few steps and store the output of the algorithm and
   the environment time steps.
4. Save checkpoint.
5. Create the algorithm using the saved config.
6. Load checkpoint.
7. Run the algorithm using the stored time steps.
8. Compare the output from step 7 with the output from step 3. They should be
   exactly same.


The simplest way to use it is to invoke it in the following way:

.. code-block:: bash

    python -m alf.bin.verify_checkpoint --conf [CONF_FILE_NAME]

You may want to set a different value of ``--num_train_iterations`` if your training
does not start from beginning because of TrainerConfig.initial_collect_steps.
You may also want to set a different value of ``--num_test_steps`` to test more steps.
"""

from absl import app
from absl import flags
from absl import logging
import os
import tempfile
import torch

import alf
from alf.trainers import policy_trainer
from alf.algorithms.data_transformer import create_data_transformer
from alf.environments.utils import create_environment
from alf.trainers import policy_trainer
from alf.utils import common, dist_utils
import alf.utils.checkpoint_utils as ckpt_utils


def _define_flags():
    flags.DEFINE_string(
        'root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
        'Root directory for writing logs/summaries/checkpoints.')
    flags.DEFINE_float('epsilon_greedy', 1., "probability of sampling action.")
    flags.DEFINE_integer('random_seed', None, "random seed")
    flags.DEFINE_integer('num_train_iterations', 2,
                         "number of training iterations")
    flags.DEFINE_integer('num_test_steps', 10, "number of test steps")
    flags.DEFINE_string('gin_file', None, 'Path to the gin-config file.')
    flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')
    flags.DEFINE_string('conf', None, 'Path to the alf config file.')
    flags.DEFINE_multi_string('conf_param', None, 'Config binding parameters.')
    flags.DEFINE_float('tolerance', 0., "Allowed difference between two runs")


FLAGS = flags.FLAGS


def _step(algorithm, time_step, policy_state, trans_state, epsilon_greedy):
    """Run one step for the algorithm."""
    batch_size = time_step.step_type.shape[0]
    policy_state = common.reset_state_if_necessary(
        policy_state, algorithm.get_initial_predict_state(batch_size),
        time_step.is_first())
    transformed_time_step, trans_state = algorithm.transform_timestep(
        time_step, trans_state)
    policy_step = algorithm.predict_step(transformed_time_step, policy_state)
    return policy_step, trans_state


def _run_steps(algorithm, env, nsteps, time_steps=[]):
    """Run several steps of algorithm.

    If ``time_steps`` is provided, will use them as the input.
    Otherwise will obtain time steps through ``env``
    """
    run_env = not bool(time_steps)
    if run_env:
        time_step = common.get_initial_time_step(env)
        time_steps.append(time_step)
        batch_size = env.batch_size
    else:
        time_step = time_steps[0]
        batch_size = time_step.step_type.shape[0]

    policy_state = algorithm.get_initial_predict_state(batch_size)
    trans_state = algorithm.get_initial_transform_state(batch_size)
    policy_steps = []

    for i in range(1, nsteps + 1):
        policy_step, trans_state = _step(
            algorithm,
            time_step,
            policy_state,
            trans_state,
            epsilon_greedy=FLAGS.epsilon_greedy)
        if run_env:
            time_step = env.step(policy_step.output)
            time_steps.append(time_step)
        else:
            time_step = time_steps[i]

        policy_steps.append(policy_step)
        policy_state = policy_step.state

    return policy_steps, time_steps


def _create_algorithm_and_env(root_dir, old_configs=None):
    """Create algorithm and env from config file."""
    alf.reset_configs()
    conf_file = common.get_conf_file()
    try:
        common.parse_conf_file(conf_file)
    except Exception as e:
        alf.close_env()
        raise e
    new_configs = dict(alf.get_inoperative_configs())
    if old_configs is not None:
        ok = True
        for k, v in old_configs.items():
            if k not in new_configs:
                logging.error("config '%s' is set by the original config file "
                              "but is not set by root_dir/alf_config.py" % k)
                ok = False
        if not ok:
            logging.fatal(
                "Some config set by the original config file are not "
                "set by root_dir/alf_config.py. It may be because these configs "
                "are set through import. Currently verify_checkpoint.py does "
                "not support this. You should replace import with alf.import_config()."
            )
    config = policy_trainer.TrainerConfig(root_dir=root_dir)

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
        config=config,
        env=env)
    algorithm.set_path('')
    return algorithm, env, new_configs


def main(_):
    with tempfile.TemporaryDirectory() as root_dir:
        FLAGS.root_dir = root_dir
        conf_file = common.get_conf_file()
        step_num = FLAGS.num_train_iterations
        ckpt_dir = os.path.join(root_dir, 'ckpt')
        algorithm1, env1, configs = _create_algorithm_and_env(root_dir)
        # The behavior of some algorithms is based by scheduler using training
        # progress (e.g. VisitSoftmaxTemperatureByProgress for MCTSAlgorithm). So we
        # need to set a valid value for progress.
        # TODO: we may want to use a different progress value based on the actual
        # progress of the checkpoint or user provided progress value.
        policy_trainer.Trainer.get_trainer_progress().set_progress(0.0)
        for i in range(FLAGS.num_train_iterations):
            # The values of some checkpointed objects (e.g. Normalizer) are
            # changed by training (i.e. not through random initialization), we
            # need to run the training a few iterations to change those value.
            # Otherwise, they will remain zeros.
            logging.info("iter=%s" % i)
            algorithm1.train_iter()
        ckpt_mngr1 = ckpt_utils.Checkpointer(ckpt_dir, alg=algorithm1)
        ckpt_mngr1.save(step_num)
        common.write_config(root_dir)

        FLAGS.gin_file = None
        FLAGS.conf = None
        algorithm2, env2, _ = _create_algorithm_and_env(root_dir, configs)
        # need one iteration to load checkpoint correctly
        algorithm2.train_iter()
        ckpt_mngr2 = ckpt_utils.Checkpointer(ckpt_dir, alg=algorithm2)
        ckpt_mngr2.load(step_num)

        seed = common.set_random_seed(FLAGS.random_seed)
        policy_steps1, time_steps = _run_steps(algorithm1, env1,
                                               FLAGS.num_test_steps)
        # We calculate policy_steps1 again to make sure policy_steps1 and
        # policy_steps2 go through exactly same computation sequence so that
        # they can be compared with each other.
        common.set_random_seed(seed)
        policy_steps1, _ = _run_steps(algorithm1, None, FLAGS.num_test_steps,
                                      time_steps)
        common.set_random_seed(seed)
        policy_steps2, _ = _run_steps(algorithm2, None, FLAGS.num_test_steps,
                                      time_steps)

        def _compare(path, x1, x2):
            diff = (x1 - x2).abs().max().detach().cpu().numpy()
            if diff > FLAGS.tolerance:
                logging.info('*** %s: diff=%s' % (path, diff))
                return False
            else:
                logging.info('    %s: diff=%s' % (path, diff))
                return True

        policy_steps1 = dist_utils.distributions_to_params(policy_steps1)
        policy_steps2 = dist_utils.distributions_to_params(policy_steps2)
        oks = alf.nest.py_map_structure_with_path(_compare, policy_steps1,
                                                  policy_steps2)
        ok = all(alf.nest.flatten(oks))
        if ok:
            logging.info('%s passes the test' % conf_file)
        else:
            logging.info('%s does not pass the test' % conf_file)


if __name__ == '__main__':
    _define_flags()
    logging.set_verbosity(logging.INFO)
    if torch.cuda.is_available():
        alf.set_default_device("cuda")
    app.run(main)
