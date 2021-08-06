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
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.data_structures import namedtuple

MyACInfo = namedtuple("MyACInfo", ["ac", "zeros"])


class MyACAlgorithm(ActorCriticAlgorithm):
    def rollout_step(self, inputs, state):
        alg_step = super().rollout_step(inputs, state)
        action = alg_step.output
        zeros = torch.zeros_like(action)
        print("rollout_step: ", zeros.shape)
        alg_step = alg_step._replace(
            info=MyACInfo(ac=alg_step.info, zeros=zeros))
        return alg_step

    def calc_loss(self, info: MyACInfo):
        zeros = info.zeros
        print("calc_loss: ", zeros.shape)
        return super().calc_loss(info.ac)

    def after_update(self, root_inputs, info: MyACInfo):
        zeros = info.zeros
        print("after_update: ", zeros.shape)
        super().after_update(root_inputs, info.ac)

    def after_train_iter(self, root_inputs, rollout_info: MyACInfo):
        zeros = rollout_info.zeros
        print("after_train_iter: ", zeros.shape)
        super().after_train_iter(root_inputs, rollout_info.ac)


# configure which RL algorithm to use
alf.config(
    'TrainerConfig',
    algorithm_ctor=partial(
        MyACAlgorithm, optimizer=alf.optimizers.Adam(lr=1e-3)),
    num_iterations=1)
