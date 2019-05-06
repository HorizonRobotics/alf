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
r"""Train using ActorCriticPolicy

To run actor_critic on gym CartPole:
```bash
pythond actor_critic.py \
  --root_dir=~/tmp/ac \
  --gin_param='train_eval.debug_summaries=1' \
  --alsologtostderr
```

To run on SocialBot CartPole:
```bash
pythond actor_critic.py \
  --root_dir=~/tmp/ac/SocialBot-CartPole \
  --env_name=SocialBot-CartPole-v0 \
  --num_parallel_environments=16 \
  --gin_param='train_eval.debug_summaries=1' \
  --gin_param='train_eval.env_load_fn=@suite_socialbot.load' \
  --alsologtostderr
```

"""

import os
import time

from absl import app
from absl import flags
from absl import logging
import random

import gin.tf
import tensorflow as tf

from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.drivers.py_driver import PyDriver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.encoding_network import EncodingNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.utils import common as tfa_common

from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.icm_algorithm import ICMAlgorithm
from alf.environments import suite_socialbot
from alf.policies.training_policy import TrainingPolicy

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('env_name', 'CartPole-v0', 'Name of an environment')
flags.DEFINE_integer('num_parallel_environments', 30,
                     'Number of environments to run in parallel')
flags.DEFINE_integer('num_iterations', 100000,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_boolean('use_rnns', False,
                     'If true, use RNN for policy and value function.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')

FLAGS = flags.FLAGS


@gin.configurable
def load_with_random_max_episode_steps(env_name,
                                       env_load_fn=suite_gym.load,
                                       min_steps=200,
                                       max_steps=250):
    return suite_gym.load(
        env_name, max_episode_steps=random.randint(min_steps, max_steps))


@gin.configurable
def train_eval(
        root_dir,
        env_name='CartPole-v0',
        env_load_fn=suite_gym.load,
        random_seed=0,
        actor_fc_layers=(200, 100),
        value_fc_layers=(200, 100),
        encoding_fc_layers=(),
        use_rnns=False,
        use_icm=False,
        num_parallel_environments=30,
        # Params for train
        train_interval=20,
        num_steps_per_iter=100,
        num_iterations=1000,
        learning_rate=5e-5,
        # Params for checkpoints
        checkpoint_interval=10000,
        # Params for summaries and logging
        log_interval=50,
        summary_interval=50,
        summaries_flush_secs=1,
        use_tf_functions=True,
        debug_summaries=False,
        summarize_grads_and_vars=False):
    """A simple train and eval for ActorCriticPolicy."""

    if root_dir is None:
        raise AttributeError('train_eval requires a root_dir.')

    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')

    train_summary_writer = tf.summary.create_file_writer(
        train_dir, flush_millis=summaries_flush_secs * 1000)
    train_summary_writer.set_as_default()

    global_step = tf.Variable(
        0, dtype=tf.int64, trainable=False, name="global_step")
    tf.summary.experimental.set_step(global_step)

    with tf.summary.record_if(
            lambda: tf.math.equal(global_step % summary_interval, 0)):
        tf.random.set_seed(random_seed)
        if num_parallel_environments == 1:
            py_env = env_load_fn(env_name)
        else:
            py_env = parallel_py_environment.ParallelPyEnvironment(
                [lambda: env_load_fn(env_name)] * num_parallel_environments)
        tf_env = tf_py_environment.TFPyEnvironment(py_env)
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        if use_rnns:
            actor_net = ActorDistributionRnnNetwork(
                tf_env.observation_spec(),
                tf_env.action_spec(),
                input_fc_layer_params=actor_fc_layers,
                output_fc_layer_params=None)
            value_net = ValueRnnNetwork(
                tf_env.observation_spec(),
                input_fc_layer_params=value_fc_layers,
                output_fc_layer_params=None)
        else:
            actor_net = ActorDistributionNetwork(
                tf_env.observation_spec(),
                tf_env.action_spec(),
                fc_layer_params=actor_fc_layers)
            value_net = ValueNetwork(
                tf_env.observation_spec(), fc_layer_params=value_fc_layers)

        encoding_net = None
        if encoding_fc_layers:
            encoding_net = EncodingNetwork(
                tf_env.observation_spec(), fc_layer_params=encoding_fc_layers)

        icm = None
        if use_icm:
            feature_spec = tf_env.observation_spec()
            if encoding_net:
                feature_spec = tf.TensorSpec((encoding_fc_layers[-1], ),
                                             dtype=tf.float32)
            icm = ICMAlgorithm(
                tf_env.action_spec(), feature_spec, encoding_net=encoding_net)

        algorithm = ActorCriticAlgorithm(
            action_spec=tf_env.action_spec(),
            actor_network=actor_net,
            value_network=value_net,
            intrinsic_curiosity_module=icm,
            optimizer=optimizer,
            train_step_counter=global_step,
            debug_summaries=debug_summaries)

        policy = TrainingPolicy(
            algorithm=algorithm,
            train_interval=train_interval,
            time_step_spec=tf_env.time_step_spec(),
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=global_step)

        checkpointer = tfa_common.Checkpointer(
            ckpt_dir=os.path.join(train_dir, 'policy'),
            policy=policy,
            global_step=global_step)
        checkpointer.initialize_or_restore()

        metric_buf_size = max(10, num_parallel_environments)
        train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(buffer_size=metric_buf_size),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=metric_buf_size),
        ]
        driver = PyDriver(
            tf_env,
            policy,
            observers=train_metrics,
            max_steps=num_steps_per_iter)

        start_time = time.time()
        start_iter = 0

        time_step = tf_env.reset()
        policy_state = policy.get_initial_state(tf_env.batch_size)
        for iter in range(1, num_iterations + 1):
            time_step, policy_state = driver.run(time_step, policy_state)

            if iter % log_interval == 0:
                time_acc = time.time() - start_time
                iters_per_sec = (iter - start_iter) / time_acc
                logging.info('%.3f iters/sec', iters_per_sec)
                tf.summary.scalar(
                    name='iters_per_sec', data=iters_per_sec, step=global_step)
                start_iter = iter
                start_time = time.time()

            for train_metric in train_metrics:
                train_metric.tf_summaries(
                    train_step=global_step, step_metrics=train_metrics[:2])

            if iter % checkpoint_interval == 0:
                checkpointer.save(global_step=global_step.numpy())


def main(_):
    logging.set_verbosity(logging.INFO)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    train_eval(
        FLAGS.root_dir,
        env_name=FLAGS.env_name,
        use_rnns=FLAGS.use_rnns,
        num_parallel_environments=FLAGS.num_parallel_environments,
        num_iterations=FLAGS.num_iterations)


if __name__ == '__main__':
    tf.config.gpu.set_per_process_memory_growth(True)
    flags.mark_flag_as_required('root_dir')
    app.run(main)
