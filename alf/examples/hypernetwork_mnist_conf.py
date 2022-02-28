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

import functools
import alf
from alf.utils import datagen
from alf.algorithms.hypernetwork_algorithm import HyperNetwork
from alf.trainers import policy_trainer

CONV_LAYER_PARAMS = ((6, 5, 1, 2, 2), (16, 5, 1, 0, 2), (120, 5, 1))
FC_LAYER_PARAMS = (84, )
HIDDEN_LAYERS = (512, 1024)

noise_dim = 256

dcreator = functools.partial(datagen.load_mnist, train_bs=128, test_bs=100)

alf.config(
    'HyperNetwork',
    data_creator=dcreator,
    # data_creator_outlier= dcreator_outlier,
    conv_layer_params=CONV_LAYER_PARAMS,
    fc_layer_params=FC_LAYER_PARAMS,
    use_fc_bias=True,
    hidden_layers=HIDDEN_LAYERS,
    noise_dim=noise_dim,
    num_particles=10,
    par_vi='svgd3',
    functional_gradient=True,
    loss_type='classification',
    init_lambda=.1,
    lambda_trainable=True,
    direct_jac_inverse=False,
    block_inverse_mvp=True,
    entropy_regularization=None,
    inverse_mvp_hidden_size=512,
    inverse_mvp_hidden_layers=3,
    optimizer=alf.optimizers.Adam(lr=1e-4, weight_decay=1e-4),
    inverse_mvp_optimizer=alf.optimizers.Adam(lr=1e-4),
    lambda_optimizer=alf.optimizers.Adam(lr=1e-3),
    logging_training=True,
    logging_evaluate=True)

alf.config('ParamConvNet', use_bias=True)

alf.config(
    'TrainerConfig',
    algorithm_ctor=HyperNetwork,
    ml_type='sl',
    num_iterations=200,
    num_checkpoints=1,
    evaluate=True,
    # eval_uncertainty=True,
    eval_interval=1,
    summary_interval=1,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    random_seed=0)
