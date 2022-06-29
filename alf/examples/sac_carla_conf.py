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
from alf.algorithms.data_transformer import (
    ImageScaleTransformer, ObservationNormalizer, RewardNormalizer)
from alf.environments import suite_carla
from alf.environments.alf_wrappers import (ActionObservationWrapper,
                                           ScalarRewardWrapper)
from alf.environments.carla_env.carla_utils import CarlaActionWrapper
from alf.environments.carla_controller import VehicleController
from alf.tensor_specs import BoundedTensorSpec

from alf.examples import carla_conf
from alf.examples import sac_conf

alf.config('ImageScaleTransformer', min=0.0, fields=['observation.camera'])
alf.config('ObservationNormalizer', clipping=5.0)

# training config
alf.config(
    'TrainerConfig',
    initial_collect_steps=3000,
    unroll_length=10,
    mini_batch_size=64,
    num_updates_per_train_iter=1,
    num_iterations=1000000,
    num_checkpoints=20,
    evaluate=False,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    summary_interval=0,
    num_summaries=1000,
    replay_buffer_length=70000,
    summarize_action_distributions=True,
)

alf.config(
    'suite_carla.Player',
    # Not yet able to successfully train with sparse reward.
    sparse_reward=False,
    # Cannot train well with allow_negative_distance_reward=False yet.
    allow_negative_distance_reward=True,
    max_collision_penalty=20.,
    max_red_light_penalty=20.,

    # uncomment the following line to use PID controller
    # controller_ctor=VehicleController,
    with_gnss_sensor=False,

    # uncomment the following two lines to use the light-weight CARLA setting
    # with_red_light_sensor=True,
    # with_camera_sensor=False
)

alf.config(
    'CarlaEnvironment',
    vehicle_filter='vehicle.*',
    num_other_vehicles=20,
    num_walkers=20,
    # 1000 second day length means 4.5 days in replay buffer of 90000 length
    day_length=1000,
    max_weather_length=500,
)

alf.define_config('taac', True)
taac = alf.get_config_value('taac')

wrappers = [CarlaActionWrapper, ActionObservationWrapper, ScalarRewardWrapper]
alf.config('suite_carla.load', wrappers=wrappers)

if 'camera' in alf.get_raw_observation_spec()['observation']:
    data_transformer_ctor = [ImageScaleTransformer, ObservationNormalizer]
else:
    data_transformer_ctor = [ObservationNormalizer]

if taac:
    alf.config('RewardNormalizer', clip_value=1.0)
    data_transformer_ctor.append(RewardNormalizer)
    mini_batch_length = 8
else:
    mini_batch_length = 4

alf.config(
    'TrainerConfig',
    mini_batch_length=mini_batch_length,
    data_transformer_ctor=data_transformer_ctor,
)

encoding_dim = 256
fc_layers_params = (256, )
activation = torch.relu_
use_batch_normalization = False
learning_rate = 1e-4

proj_net = partial(
    alf.networks.StableNormalProjectionNetwork,
    state_dependent_std=True,
    squash_mean=False,
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

# config PredictiveRepresentationLearner
from alf.algorithms.predictive_representation_learner import PredictiveRepresentationLearner, SimpleDecoder

decoder_net_ctor = partial(
    alf.networks.EncodingNetwork,
    fc_layer_params=fc_layers_params,
    last_layer_size=suite_carla.Player.REWARD_DIMENSION,
    last_activation=alf.math.identity,
    last_kernel_initializer=torch.nn.init.zeros_)
decoder_ctor = partial(
    SimpleDecoder,
    decoder_net_ctor=decoder_net_ctor,
    target_field='reward',
    summarize_each_dimension=True)
dynamics_net_ctor = partial(
    alf.networks.LSTMEncodingNetwork,
    preprocessing_combiner=alf.layers.NestSum(
        activation=activation, average=True),
    hidden_size=(encoding_dim, encoding_dim))
alf.config(
    'PredictiveRepresentationLearner',
    num_unroll_steps=10,
    decoder_ctor=decoder_ctor,
    encoding_net_ctor=encoder_cls,
    dynamics_net_ctor=dynamics_net_ctor)

alf.config('ReplayBuffer', keep_episodic_info=True)

# representation learning options:
# a) set `ReprLearner=EncodingAlgorithm` to use EncodingAlgorithm
# b) set `ReprLearner=PredictiveRepresentationLearner` to use PredictiveRepresentationLearner
ReprLearner = EncodingAlgorithm

alf.config(
    'Agent',
    representation_learner_cls=ReprLearner,
    optimizer=alf.optimizers.Adam(lr=learning_rate),
)

if not taac:
    from alf.algorithms.sac_algorithm import SacAlgorithm
    alf.config('Agent', rl_algorithm_cls=SacAlgorithm)

    alf.config(
        'SacAlgorithm',
        actor_network_cls=actor_network_cls,
        critic_network_cls=critic_network_cls,
        target_entropy=partial(calc_default_target_entropy, min_prob=0.1),
        target_update_tau=0.005,
        critic_loss_ctor=TDLoss,
        use_entropy_reward=False)
else:
    from alf.algorithms.taac_algorithm import TaacLAlgorithm

    alf.config('Agent', rl_algorithm_cls=TaacLAlgorithm)
    alf.config(
        'TrainerConfig',
        use_rollout_state=True,
        temporally_independent_train_step=True)

    alf.config(
        'TaacAlgorithmBase',
        actor_network_cls=actor_network_cls,
        critic_network_cls=critic_network_cls,
        target_update_tau=0.005,
        target_entropy=(partial(calc_default_target_entropy, min_prob=0.1),
                        partial(calc_default_target_entropy, min_prob=0.1)))
    # In this particular task, ``inverse_mode=False`` will be much better
    alf.config('TaacLAlgorithm', inverse_mode=False)
