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

from functools import partial
import torch

import alf
from alf.algorithms.td_loss import TDLoss
from alf.algorithms.data_transformer import (ImageScaleTransformer,
                                             ObservationNormalizer,
                                             FrameStacker, RewardNormalizer)
from alf.algorithms.lagrangian_reward_weight_algorithm import LagrangianRewardWeightAlgorithm
from alf.environments import suite_carla
from alf.environments.alf_wrappers import ActionObservationWrapper, AlfEnvironmentBaseWrapper, ScalarRewardWrapper, CarlaActionWrapper
from alf.environments.carla_controller import VehicleController
from alf.networks import BetaProjectionNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec

from alf.examples import carla_conf
from alf.examples import sac_conf


@alf.configurable
class VectorRewardWrapper(AlfEnvironmentBaseWrapper):
    """A wrapper that returns a vector reward by extracting values from env_info.
    """

    def __init__(self, env, keys=[]):
        """
        Args:
            keys (list[str]): the list of info keys.
        """
        super().__init__(env)
        self._keys = keys

    def _add_rewards(self, time_step):
        rewards = []
        for key in self._keys:
            rewards.append(-time_step.env_info[key])
        return time_step._replace(
            reward=torch.stack([time_step.reward] + rewards, dim=-1))

    def _step(self, action):
        time_step = self._env._step(action)
        return self._add_rewards(time_step)

    def _reset(self):
        time_step = self._env._reset()
        return self._add_rewards(time_step)

    def reward_spec(self):
        return TensorSpec((len(self._keys) + 1, ))

    def time_step_spec(self):
        spec = self._env.time_step_spec()
        return spec._replace(reward=self.reward_spec())


class NormalizedActionWrapper(AlfEnvironmentBaseWrapper):
    """A wrapper that converts the last action dim ('reverse') from [0,1] to
    [-1,1].
    """

    def __init__(self, env):
        super().__init__(env)
        self._action_spec = BoundedTensorSpec(
            shape=env.action_spec().shape,
            dtype=env.action_spec().dtype,
            minimum=-1.,
            maximum=1.)
        self._time_step_spec = env.time_step_spec()._replace(
            prev_action=self._action_spec)

    def _step(self, action):
        action[..., -1] = (action[..., -1] + 1.) / 2.
        return self._env.step(action)

    def action_spec(self):
        return self._action_spec

    def time_step_spec(self):
        return self._time_step_spec


alf.config(
    'suite_carla.load',
    wrappers=[
        CarlaActionWrapper, ActionObservationWrapper, ScalarRewardWrapper,
        partial(VectorRewardWrapper, keys=['collision'])
    ])

alf.config(
    'suite_carla.Player',
    # Not yet able to successfully train with sparse reward.
    sparse_reward=False,
    # Cannot train well with allow_negative_distance_reward=False yet.
    allow_negative_distance_reward=True,
    # don't add these penalties to the overall reward; instead we'll use the
    # Lagrangian weights to combine them.
    max_collision_penalty=0.,
    max_red_light_penalty=0.,
    overspeed_penalty_weight=0.,

    # uncomment the following line to use PID controller
    # controller_ctor=VehicleController,
    with_gnss_sensor=False,
)

alf.config(
    'CarlaEnvironment',
    step_time=0.1,
    vehicle_filter='vehicle.tesla.model3',
    num_other_vehicles=20,
    num_walkers=20,
    # 1000 second day length means 4.5 days in replay buffer of 90000 length
    day_length=1000,
    max_weather_length=500,
)

encoding_dim = 256
fc_layers_params = (256, )
activation = torch.relu_
use_batch_normalization = False
learning_rate = 1e-4

#proj_net = partial(BetaProjectionNetwork, min_concentration=1.)
proj_net = partial(
    alf.networks.StableNormalProjectionNetwork,
    state_dependent_std=True,
    scale_distribution=True,
    min_std=1e-3,
    max_std=10)

actor_network_cls = partial(
    alf.networks.ActorDistributionNetwork,
    input_preprocessors=alf.layers.Detach(),
    fc_layer_params=fc_layers_params,
    activation=activation,
    use_fc_bn=use_batch_normalization,
    continuous_projection_net_ctor=proj_net)

alf.config(
    "TrainerConfig",
    data_transformer_ctor=[
        partial(ImageScaleTransformer, min=0.0, fields=['observation.camera']),
        partial(ObservationNormalizer, clipping=5.0),
        partial(RewardNormalizer, clip_value=5.0)
    ])

env = alf.get_env()

critic_network_cls = partial(
    alf.networks.CriticNetwork,
    joint_fc_layer_params=fc_layers_params,
    activation=activation,
    use_fc_bn=use_batch_normalization)

from alf.utils.dist_utils import calc_default_target_entropy

# config EncodingAlgorithm
encoder_cls = partial(
    alf.networks.EncodingNetwork,
    input_preprocessors=carla_conf.create_input_preprocessors(
        encoding_dim, use_batch_normalization),
    preprocessing_combiner=alf.layers.NestSum(
        activation=activation, average=True),
    activation=activation,
    fc_layer_params=fc_layers_params,
)
from alf.algorithms.encoding_algorithm import EncodingAlgorithm
alf.config('EncodingAlgorithm', encoder_cls=encoder_cls)

alf.config(
    'LagrangianRewardWeightAlgorithm',
    reward_thresholds=[None, -1e-3],
    optimizer=alf.optimizers.AdamTF(lr=1e-3))

alf.config(
    'Agent',
    representation_learner_cls=EncodingAlgorithm,
    optimizer=alf.optimizers.AdamTF(lr=learning_rate),
    reward_weight_algorithm_cls=LagrangianRewardWeightAlgorithm)

alf.config('calc_default_target_entropy', min_prob=0.1)

alf.config(
    'SacAlgorithm',
    actor_network_cls=actor_network_cls,
    critic_network_cls=critic_network_cls,
    target_update_tau=0.005,
    critic_loss_ctor=TDLoss)

# training config
alf.config(
    'TrainerConfig',
    temporally_independent_train_step=True,
    initial_collect_steps=10000,
    mini_batch_length=4,
    unroll_length=50,
    mini_batch_size=64,
    num_updates_per_train_iter=5,
    num_iterations=0,
    num_env_steps=int(5e6),
    num_checkpoints=1,
    evaluate=False,
    debug_summaries=True,
    num_evals=100,
    num_summaries=100,
    replay_buffer_length=70000)
