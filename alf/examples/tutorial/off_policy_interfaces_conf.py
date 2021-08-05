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
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.data_structures import namedtuple
from alf.examples import sac_conf

MySacInfo = namedtuple("MySacInfo", ["sac", "zeros"])


class MySacAlgorithm(SacAlgorithm):
    def rollout_step(self, inputs, state):
        alg_step = super().rollout_step(inputs, state)
        action = alg_step.output
        zeros = torch.zeros_like(action)
        print("rollout_step: ", zeros.shape)
        alg_step = alg_step._replace(
            info=MySacInfo(sac=alg_step.info, zeros=zeros))
        return alg_step

    def train_step(self, inputs, state, rollout_info: MySacInfo):
        alg_step = super().train_step(inputs, state, rollout_info.sac)
        print("train_step rollout zeros:  ", rollout_info.zeros.shape)
        train_zeros = torch.zeros_like(alg_step.output, dtype=torch.uint8)
        print("train_step train zeros:", train_zeros.shape)
        alg_step = alg_step._replace(
            info=MySacInfo(sac=alg_step.info, zeros=train_zeros))
        return alg_step

    def calc_loss(self, info: MySacInfo):
        zeros = info.zeros
        print("calc_loss: ", zeros.shape, zeros.dtype)
        return super().calc_loss(info.sac)

    def after_update(self, root_inputs, info: MySacInfo):
        zeros = info.zeros
        print("after_update: ", zeros.shape, zeros.dtype)
        super().after_update(root_inputs, info.sac)

    def after_train_iter(self, root_inputs, rollout_info: MySacInfo):
        zeros = rollout_info.zeros
        print("after_train_iter: ", zeros.shape, zeros.dtype)
        super().after_train_iter(root_inputs, rollout_info.sac)


alf.config(
    'Agent',
    rl_algorithm_cls=MySacAlgorithm,
    optimizer=alf.optimizers.Adam(lr=1e-3))

alf.config('create_environment', num_parallel_environments=10)

alf.config(
    'TrainerConfig',
    temporally_independent_train_step=False,
    mini_batch_length=2,
    unroll_length=3,
    mini_batch_size=4,
    num_updates_per_train_iter=1,
    num_iterations=1)
