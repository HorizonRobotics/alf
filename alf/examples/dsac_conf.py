# Copyright (c) 2023 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from functools import partial
import math

import alf
from alf.algorithms.dsac_algorithm import DSacAlgorithm
from alf.algorithms.one_step_loss import OneStepTDQRLoss
from alf.environments import suite_dmc
from alf.environments.gym_wrappers import FrameSkip
from alf.networks import NormalProjectionNetwork, ActorDistributionNetwork
from alf.networks import CriticQuantileNetwork, CriticNetwork
from alf.optimizers import AdamTF
from alf.tensor_specs import TensorSpec
from alf.utils.math_ops import clipped_exp

from alf.examples import sac_conf


@alf.configurable
def dsac_use_epistemic_alpha(epistemic_alpha=True):
    return epistemic_alpha


@alf.configurable
def dsac_dmc_tasks(dmc=True):
    return dmc


# algorithm config
use_epistemic_alpha = dsac_use_epistemic_alpha()
print(f"Use epistemic alpha: {use_epistemic_alpha}!")
dmc_tasks = dsac_dmc_tasks()
print(f"Evaluate on DM control: {dmc_tasks}!")

# environment config
if dmc_tasks:
    alf.config(
        "create_environment",
        env_name="cheetah:run",
        num_parallel_environments=1,
        env_load_fn=suite_dmc.load)

    alf.config(
        "suite_dmc.load",
        from_pixels=False,
        visualize_reward=True,
        gym_env_wrappers=(partial(FrameSkip, skip=1), ),
        max_episode_steps=1000)
    if use_epistemic_alpha:
        alpha_optimizer = None
        initial_log_alpha = -3.2
    else:
        alpha_optimizer = AdamTF(lr=1e-4)
        initial_log_alpha = math.log(0.1)
    layer_width = 1024
    target_update_period = 2
    num_iterations = 1000000
else:
    alf.config(
        'create_environment',
        env_name="HalfCheetah-v3",
        num_parallel_environments=1)
    if use_epistemic_alpha:
        alpha_optimizer = None
        initial_log_alpha = -3.2
    else:
        alpha_optimizer = AdamTF(lr=3e-4)
        initial_log_alpha = 0.0
    layer_width = 256
    target_update_period = 1
    num_iterations = 2500000

# algorithm config
fc_layer_params = (layer_width, layer_width)
actor_network_cls = partial(
    ActorDistributionNetwork,
    fc_layer_params=fc_layer_params,
    continuous_projection_net_ctor=partial(
        NormalProjectionNetwork,
        state_dependent_std=True,
        scale_distribution=True,
        std_transform=clipped_exp))

obs_act_tau_joint_fc_layer_params = (layer_width, )
num_quantiles = 32
tau_embedding_dim = 64

critic_network_cls = partial(
    CriticQuantileNetwork,
    tau_embedding_dim=tau_embedding_dim,
    obs_act_tau_joint_fc_layer_params=obs_act_tau_joint_fc_layer_params,
)

alf.config('calc_default_target_entropy', min_prob=0.184)
alf.config('OneStepTDQRLoss', num_quantiles=num_quantiles)

critic_loss_ctor = OneStepTDQRLoss
alf.config(
    'DSacAlgorithm',
    tau_type='iqn',
    num_quantiles=num_quantiles,
    actor_network_cls=actor_network_cls,
    critic_network_cls=critic_network_cls,
    critic_loss_ctor=critic_loss_ctor,
    num_critic_replicas=2,
    use_entropy_reward=True,
    target_update_tau=0.005,
    target_update_period=target_update_period,
    initial_log_alpha=initial_log_alpha,
    actor_optimizer=AdamTF(lr=3e-4),
    critic_optimizer=AdamTF(lr=3e-4),
    alpha_optimizer=alpha_optimizer)

# training config
alf.config('Agent', rl_algorithm_cls=DSacAlgorithm)

alf.config(
    'TrainerConfig',
    async_eval=True,
    initial_collect_steps=10000,
    mini_batch_length=2,
    unroll_length=1,
    mini_batch_size=256,
    num_updates_per_train_iter=1,
    num_iterations=num_iterations,
    num_checkpoints=1,
    evaluate=True,
    eval_interval=5000,
    num_eval_episodes=5,
    debug_summaries=True,
    random_seed=0,
    summarize_grads_and_vars=True,
    summary_interval=1000,
    replay_buffer_length=int(1e6))
