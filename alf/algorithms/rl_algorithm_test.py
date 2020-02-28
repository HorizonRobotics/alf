# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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

import torch
import torch.distributions as td
import unittest

import alf
from alf.utils import common, dist_utils, tensor_utils
from alf.data_structures import AlgStep, LossInfo, StepType, TimeStep, TrainingInfo
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm


class MyAlg(OnPolicyAlgorithm):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 env,
                 config,
                 debug_summaries=False):
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            train_state_spec=(),
            env=env,
            config=config,
            optimizer=torch.optim.Adam(lr=1e-1),
            debug_summaries=debug_summaries,
            name="MyAlg")

        self._proj_net = alf.networks.CategoricalProjectionNetwork(
            input_size=2, num_actions=3)

    def is_on_policy(self):
        return True

    def predict_step(self, time_step: TimeStep, state, epsilon_greedy):
        dist = self._proj_net(time_step.observation)
        return AlgStep(output=dist.sample(), state=(), info=())

    def rollout_step(self, time_step: TimeStep, state):
        dist = self._proj_net(time_step.observation)
        return AlgStep(output=dist.sample(), state=(), info=dist)

    def calc_loss(self, training_info: TrainingInfo):
        dist: td.Distribution = training_info.info
        log_prob = dist.log_prob(training_info.action)
        loss = -log_prob[:-1] * training_info.reward[1:]
        loss = tensor_utils.tensor_extend_zero(loss)
        return LossInfo(loss=loss)


#TODO: move this to environments.suite_unittest
class MyEnv(object):
    def __init__(self, batch_size):
        super().__init__()
        self._batch_size = batch_size
        self._rewards = torch.tensor([0.5, 1.0, -1.])
        self._observation_spec = alf.TensorSpec((2, ), dtype='float32'),
        self._action_spec = alf.BoundedTensorSpec(
            shape=(), dtype='int32', minimum=0, maximum=2),
        self.reset()

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def reset(self):
        self._prev_action = torch.zeros(self._batch_size, dtype=torch.int32),
        self._current_time_step = TimeStep(
            observation=torch.randn(self._batch_size, 2),
            step_type=torch.full([self._batch_size], StepType.FIRST),
            reward=torch.zeros(self._batch_size),
            discount=torch.zeros(self._batch_size),
            prev_action=self._prev_action,
            env_id=torch.arange(self._batch_size))
        return self._current_time_step

    def close(self):
        pass

    @property
    def batch_size(self):
        return self._batch_size

    def step(self, action):
        prev_step_type = self._current_time_step.step_type
        is_first = prev_step_type == StepType.FIRST
        is_mid = prev_step_type == StepType.MID
        is_last = prev_step_type == StepType.LAST

        step_type = torch.where(is_mid & (torch.rand(self._batch_size) < 0.2),
                                torch.full([self._batch_size], StepType.LAST),
                                torch.full([self._batch_size], StepType.MID))
        step_type = torch.where(is_last,
                                torch.full([self._batch_size], StepType.FIRST),
                                step_type)
        step_type = torch.where(is_first,
                                torch.full([self._batch_size], StepType.MID),
                                step_type)

        self._current_time_step = TimeStep(
            observation=torch.randn(self._batch_size, 2),
            step_type=step_type,
            reward=self._rewards[action],
            discount=torch.zeros(self._batch_size),
            prev_action=self._prev_action,
            env_id=torch.arange(self._batch_size))

        self._prev_action = action
        return self._current_time_step

    def current_time_step(self):
        return self._current_time_step


class Config(object):
    def __init__(self):
        self.unroll_length = 5


class RLAlgorithmTest(unittest.TestCase):
    def test_rl_algorithm(self):
        config = Config()
        env = MyEnv(batch_size=3)
        alg = MyAlg(
            observation_spec=env.observation_spec(),
            action_spec=env.action_spec(),
            env=env,
            config=config)
        for _ in range(100):
            alg.train_iter()

        time_step = common.get_initial_time_step(env)
        state = alg.get_initial_predict_state(env.batch_size)
        policy_step = alg.rollout_step(time_step, state)
        logits = policy_step.info.logits
        print("logits: ", logits)
        self.assertTrue(torch.all(logits[:, 1] > logits[:, 0]))
        self.assertTrue(torch.all(logits[:, 1] > logits[:, 2]))


if __name__ == '__main__':
    RLAlgorithmTest().test_rl_algorithm()
    #unittest.main()
