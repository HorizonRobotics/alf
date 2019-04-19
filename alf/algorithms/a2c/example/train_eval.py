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


r"""Train using A2cAgent
To run a2c on gym CartPole:
```bash
python train_eval.py \
  --root_dir=~/tmp/a2c/CartPole
```
To run on gym Pendulum:
```bash
python train_eval.py \
  --root_dir=~/tmp/a2c/Pendulum \
  --gin_param='train_eval.env_name="Pendulum"'
```
"""

import os
from absl import app
from absl import flags

import tensorflow as tf
from tf_agents.metrics import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment, parallel_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.utils import common
from alf.algorithms.a2c import a2c_agent

import gin

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('num_iterations', 100000,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')
FLAGS = flags.FLAGS


@gin.configurable
def train_eval(root_dir,
               env_name='CartPole-v0',
               env_load_fn=suite_gym.load,
               num_iterations=1000,
               actor_fc_layers=(100,),
               value_fc_layers=(100,),
               use_tf_functions=True,
               num_parallel_environments=8,
               td_steps=8,
               replay_buffer_capacity=2000,
               learning_rate=1e-3,
               gamma=0.98,
               entropy_regularization=1e-3,
               gradient_clipping=None,
               num_eval_episodes=10,
               eval_interval=100,
               summary_interval=100,
               summaries_flush_secs=1,
               debug_summaries=True,
               summarize_grads_and_vars=False):
    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')

    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        train_dir, flush_millis=summaries_flush_secs * 1000)
    train_summary_writer.set_as_default()

    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000)
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
    ]

    with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(global_step % summary_interval, 0)):
        if num_parallel_environments > 1:
            tf_env = tf_py_environment.TFPyEnvironment(
                parallel_py_environment.ParallelPyEnvironment(
                    [lambda: env_load_fn(env_name)] * num_parallel_environments))
        else:
            tf_env = tf_py_environment.TFPyEnvironment(env_load_fn(env_name))
        eval_tf_env = tf_py_environment.TFPyEnvironment(env_load_fn(env_name))

        actor_net = actor_distribution_network.ActorDistributionNetwork(
            tf_env.time_step_spec().observation,
            tf_env.action_spec(),
            fc_layer_params=actor_fc_layers)

        value_net = value_network.ValueNetwork(
            tf_env.time_step_spec().observation,
            fc_layer_params=value_fc_layers)

        global_step = tf.compat.v1.train.get_or_create_global_step()

        tf_agent = a2c_agent.A2CAgent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            actor_net=actor_net,
            value_net=value_net,
            optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=learning_rate),
            gradient_clipping=gradient_clipping,
            gamma=gamma,
            entropy_regularization=entropy_regularization,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=global_step)
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            tf_agent.collect_data_spec,
            batch_size=tf_env.batch_size,
            max_length=replay_buffer_capacity)

        tf_agent.initialize()

        train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric()
        ]

        eval_policy = tf_agent.policy
        collect_policy = tf_agent.collect_policy
        collect_steps_per_iteration = num_parallel_environments * td_steps
        collect_driver = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            collect_policy,
            observers=[replay_buffer.add_batch] + train_metrics,
            num_steps=collect_steps_per_iteration)

        if use_tf_functions:
            collect_driver.run = common.function(collect_driver.run)
            tf_agent.train = common.function(tf_agent.train)

        metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics')

        time_step = None
        policy_state = collect_policy.get_initial_state(tf_env.batch_size)

        for _ in range(num_iterations):
            time_step, policy_state = collect_driver.run(
                time_step=time_step, policy_state=policy_state)
            experience = replay_buffer.gather_all()
            tf_agent.train(experience)
            replay_buffer.clear()
            global_step_val = global_step.numpy()
            if global_step_val % eval_interval == 0:
                metric_utils.eager_compute(
                    eval_metrics,
                    eval_tf_env,
                    eval_policy,
                    num_episodes=num_eval_episodes,
                    train_step=global_step,
                    summary_writer=eval_summary_writer,
                    summary_prefix='Metrics')


def main(_):
    tf.compat.v1.enable_eager_execution(
        config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
    tf.compat.v1.enable_v2_behavior()
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    train_eval(FLAGS.root_dir, num_iterations=FLAGS.num_iterations)


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
