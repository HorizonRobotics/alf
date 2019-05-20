
import os
import logging
import tensorflow as tf
from absl import flags
from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.environments import tf_py_environment
from alf.environments import suite_socialbot
from tf_agents.utils import common
from tf_agents.trajectories import time_step
from tf_agents.utils import tensor_normalizer
from tf_agents.agents.ppo import ppo_policy
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from absl import logging as absl_logging

import gin.tf
import gin
import gym
import sys

root_dir = sys.argv[1]
env_name='SocialBot-Pr2Gripper-v0'
env_load_fn=suite_socialbot.load

absl_logging.set_verbosity(absl_logging.INFO)

logging.basicConfig(level=logging.INFO)
logging.getLogger().addHandler(
    logging.FileHandler(filename=root_dir + '/demo.log'))


train_dir = os.path.join(root_dir, 'train')
actor_fc_layers=(100, 50, 25)
value_fc_layers=(100, 50, 25)


tf_env = tf_py_environment.TFPyEnvironment(env_load_fn(env_name))

actor_net = actor_distribution_network.ActorDistributionNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    fc_layer_params=actor_fc_layers,
    activation_fn=tf.keras.activations.softsign)
value_net = value_network.ValueNetwork(
    tf_env.observation_spec(), fc_layer_params=value_fc_layers,
    activation_fn=tf.keras.activations.softsign)

time_step_spec = tf_env.time_step_spec()
observation_normalizer = (
    tensor_normalizer.StreamingTensorNormalizer(
        time_step_spec.observation, scope='normalize_observations'))

tf_agent = ppo_agent.PPOAgent(
    time_step_spec,
    tf_env.action_spec(),
    actor_net=actor_net,
    value_net=value_net)

train_checkpointer = common.Checkpointer(
    ckpt_dir=train_dir,
    agent=tf_agent)

train_checkpointer.initialize_or_restore()

num_eval_episodes = 100

demo_metrics = [
    tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
    tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
]

policy = tf_agent.policy

metric_utils.eager_compute(
    demo_metrics,
    tf_env,
    policy,
    num_episodes=num_eval_episodes,
    train_step=100000000,
)
