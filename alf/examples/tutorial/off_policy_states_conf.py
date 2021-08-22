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
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.data_structures import AlgStep, LossInfo
from alf.tensor_specs import TensorSpec


class MyOffPolicyAlgorithm(OffPolicyAlgorithm):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 reward_spec=None,
                 env=None,
                 config=None,
                 debug_summaries=False):
        rollout_state_spec = TensorSpec(shape=(), dtype=torch.int32)
        train_state_spec = TensorSpec(shape=(2, ))
        super().__init__(
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            observation_spec=observation_spec,
            action_spec=action_spec,
            train_state_spec=train_state_spec,
            rollout_state_spec=rollout_state_spec)

    def rollout_step(self, inputs, state):
        print("rollout_step: ", state)
        is_first_steps = inputs.is_first()
        is_zero_state = (state == 0)
        assert torch.all(is_zero_state[is_first_steps])
        return AlgStep(output=inputs.prev_action, state=state - 1)

    def train_step(self, inputs, state, rollout_info):
        print("train_step: ", state)
        return AlgStep(output=inputs.prev_action, state=state + 1)

    def calc_loss(self, info):
        return LossInfo()


alf.config('create_environment', num_parallel_environments=10)

alf.config(
    'TrainerConfig',
    algorithm_ctor=MyOffPolicyAlgorithm,
    whole_replay_buffer_training=False,
    use_rollout_state=False,
    mini_batch_length=2,
    unroll_length=3,
    mini_batch_size=4,
    num_updates_per_train_iter=1,
    num_iterations=1)
