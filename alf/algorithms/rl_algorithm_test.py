# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import tempfile
import torch
import torch.distributions as td
import unittest

import alf
from alf.utils import common, dist_utils, tensor_utils
from alf.data_structures import AlgStep, Experience, LossInfo, StepType, TimeStep
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.algorithms.config import TrainerConfig
from alf.tensor_specs import TensorSpec


class MyAlg(RLAlgorithm):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 reward_spec=TensorSpec(()),
                 env=None,
                 config=None,
                 on_policy=True,
                 debug_summaries=False):
        self._on_policy = on_policy
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=observation_spec,
            env=env,
            is_on_policy=on_policy,
            config=config,
            optimizer=alf.optimizers.Adam(lr=1e-1),
            debug_summaries=debug_summaries,
            name="MyAlg")

        self._proj_net = alf.networks.CategoricalProjectionNetwork(
            input_size=2, action_spec=action_spec)

    def predict_step(self, time_step: TimeStep, state):
        dist, _ = self._proj_net(time_step.observation)
        return AlgStep(output=dist.sample(), state=(), info=())

    def rollout_step(self, time_step: TimeStep, state):
        dist, _ = self._proj_net(time_step.observation)
        action = dist.sample()
        return AlgStep(
            output=action,
            state=time_step.observation,
            info=dict(dist=dist, action=action, reward=time_step.reward))

    def train_step(self, time_step: TimeStep, state, rollout_info):
        dist, _ = self._proj_net(time_step.observation)
        return AlgStep(
            output=dist.sample(),
            state=time_step.observation,
            info=dict(
                dist=dist,
                action=rollout_info['action'],
                reward=time_step.reward))

    def calc_loss(self, info):
        dist: td.Distribution = info['dist']
        log_prob = dist.log_prob(info['action'])
        loss = -log_prob[:-1] * info['reward'][1:]
        loss = tensor_utils.tensor_extend_zero(loss)
        return LossInfo(loss=loss)


#TODO: move this to environments.suite_unittest
class MyEnv(object):
    """A simple environment for unittesting algorithms.

    At each step, each episode ends with probability 0.2 (independently among
    the batch). Reward depends only on the action. Action 0 gets reward 0.5,
    action 1 gets 1.0, action 2 gets reward -1.
    """

    def __init__(self, batch_size, obs_shape=(2, ), reward_dim=1):
        super().__init__()
        self._batch_size = batch_size
        self._reward_dim = reward_dim
        self._rewards = torch.tensor([0.5, 1.0, -1.])
        if reward_dim != 1:
            self._rewards = self._rewards.unsqueeze(-1).expand(
                (-1, self._reward_dim))
        self._observation_spec = alf.TensorSpec(obs_shape, dtype='float32')
        self._action_spec = alf.BoundedTensorSpec(
            shape=(), dtype='int64', minimum=0, maximum=2)
        self.reset()

    @property
    def is_tensor_based(self):
        return True

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def reward_spec(self):
        if self._reward_dim == 1:
            return TensorSpec(())
        else:
            return TensorSpec((self._reward_dim, ))

    def reset(self):
        self._prev_action = torch.zeros(self._batch_size, dtype=torch.int64)
        self._current_time_step = TimeStep(
            observation=self._observation_spec.randn([self._batch_size]),
            step_type=torch.full([self._batch_size],
                                 StepType.FIRST,
                                 dtype=torch.int32),
            reward=self.reward_spec().zeros(outer_dims=(self._batch_size, )),
            discount=torch.zeros(self._batch_size),
            prev_action=self._prev_action,
            env_id=torch.arange(self._batch_size, dtype=torch.int32))
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

        step_type = torch.where(
            is_mid & (torch.rand(self._batch_size) < 0.2),
            torch.full([self._batch_size], StepType.LAST, dtype=torch.int32),
            torch.full([self._batch_size], StepType.MID, dtype=torch.int32))
        step_type = torch.where(
            is_last,
            torch.full([self._batch_size], StepType.FIRST, dtype=torch.int32),
            step_type)
        step_type = torch.where(
            is_first,
            torch.full([self._batch_size], StepType.MID, dtype=torch.int32),
            step_type)

        self._current_time_step = TimeStep(
            observation=self._observation_spec.randn([self._batch_size]),
            step_type=step_type,
            reward=self._rewards[action],
            discount=torch.zeros(self._batch_size),
            prev_action=self._prev_action,
            env_id=torch.arange(self._batch_size, dtype=torch.int32))
        self._prev_action = action
        return self._current_time_step

    def current_time_step(self):
        return self._current_time_step


class RLAlgorithmTest(unittest.TestCase):
    def test_on_policy_algorithm(self):
        # root_dir is not used. We have to give it a value because
        # it is a required argument of TrainerConfig.
        config = TrainerConfig(
            root_dir='/tmp/rl_algorithm_test', unroll_length=5)
        env = MyEnv(batch_size=3)
        alg = MyAlg(
            observation_spec=env.observation_spec(),
            action_spec=env.action_spec(),
            env=env,
            config=config,
            on_policy=True,
            debug_summaries=True)
        for _ in range(100):
            alg.train_iter()

        time_step = common.get_initial_time_step(env)
        state = alg.get_initial_predict_state(env.batch_size)
        policy_step = alg.rollout_step(time_step, state)
        logits = policy_step.info['dist'].log_prob(
            torch.arange(3).reshape(3, 1))
        print("logits: ", logits)
        self.assertTrue(torch.all(logits[1, :] > logits[0, :]))
        self.assertTrue(torch.all(logits[1, :] > logits[2, :]))

    def test_off_policy_algorithm(self):
        with tempfile.TemporaryDirectory() as root_dir:
            common.run_under_record_context(
                lambda: self._test_off_policy_algorithm(root_dir),
                summary_dir=root_dir,
                summary_interval=1,
                flush_secs=1)

    def _test_off_policy_algorithm(self, root_dir):
        alf.summary.enable_summary()
        config = TrainerConfig(
            root_dir=root_dir,
            unroll_length=5,
            num_updates_per_train_iter=1,
            mini_batch_length=5,
            mini_batch_size=3,
            use_rollout_state=True,
            summarize_grads_and_vars=True,
            summarize_action_distributions=True,
            whole_replay_buffer_training=True)
        env = MyEnv(batch_size=3)
        alg = MyAlg(
            observation_spec=env.observation_spec(),
            action_spec=env.action_spec(),
            env=env,
            on_policy=False,
            config=config)
        for _ in range(100):
            alg.train_iter()

        time_step = common.get_initial_time_step(env)
        state = alg.get_initial_predict_state(env.batch_size)
        policy_step = alg.rollout_step(time_step, state)
        logits = policy_step.info['dist'].log_prob(
            torch.arange(3).reshape(3, 1))
        print("logits: ", logits)
        self.assertTrue(torch.all(logits[1, :] > logits[0, :]))
        self.assertTrue(torch.all(logits[1, :] > logits[2, :]))


if __name__ == '__main__':
    unittest.main()
