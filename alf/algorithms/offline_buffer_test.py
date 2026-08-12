# Copyright (c) 2026 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
import torch.nn as nn

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import (AlgStep, Experience, LossInfo, StepType,
                                 TimeStep)
from alf.tensor_specs import TensorSpec
from alf.utils import checkpoint_utils
from alf.utils.schedulers import update_progress


class _TinyMlpAlgorithm(RLAlgorithm):
    """Minimal off-policy learner used to exercise replay mixing end to end."""

    def __init__(self, config):
        observation_spec = TensorSpec((1, ))
        action_spec = TensorSpec((1, ))
        super().__init__(observation_spec=observation_spec,
                         action_spec=action_spec,
                         train_state_spec=(),
                         env=None,
                         is_on_policy=False,
                         config=config,
                         optimizer=alf.optimizers.Adam(lr=1e-2),
                         name="TinyMlpAlgorithm")
        self._mlp = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 1))
        self.seen_sources = []
        self.seen_grad_steps = []
        self.set_replay_buffer(num_envs=1,
                               max_length=128,
                               prioritized_sampling=False)

    def predict_step(self, time_step, state):
        return AlgStep(output=self._mlp(time_step.observation), state=state)

    def rollout_step(self, time_step, state):
        return AlgStep(output=self._mlp(time_step.observation),
                       state=state,
                       info=())

    def train_step(self, time_step, state, rollout_info):
        prediction = self._mlp(time_step.observation).squeeze(-1)
        target = rollout_info["target"].reshape(-1)
        source = rollout_info["source"].reshape(-1)
        self.seen_sources.extend(source.detach().cpu().tolist())
        self.seen_grad_steps.append(alf.summary.get_grad_step_counter())
        return AlgStep(state=state,
                       info=LossInfo(loss=(prediction - target)**2))


class OfflineBufferTest(alf.test.TestCase):

    _BATCH_SIZE = 6
    _UTD = 2
    _BATCH_LENGTH = 2

    def tearDown(self):
        update_progress("iterations", 0)
        super().tearDown()

    def _config(self, use_offline_buffer):
        return TrainerConfig(
            root_dir="/tmp/offline_buffer_test",
            num_iterations=20,
            unroll_length=1,
            initial_collect_steps=0,
            num_updates_per_train_iter=self._UTD,
            mini_batch_length=self._BATCH_LENGTH,
            mini_batch_size=self._BATCH_SIZE,
            whole_replay_buffer_training=False,
            clear_replay_buffer=False,
            mask_out_loss_for_last_step=False,
            use_offline_buffer=use_offline_buffer,
            offline_training_iters=10 if use_offline_buffer else 0)

    def _experience(self, source, value):
        time_step = TimeStep(step_type=torch.tensor([StepType.MID],
                                                    dtype=torch.int32),
                             reward=torch.zeros(1),
                             discount=torch.ones(1),
                             observation=torch.tensor([[float(value)]],
                                                      dtype=torch.float32),
                             prev_action=torch.zeros(1, 1),
                             env_id=torch.zeros(1, dtype=torch.int32))
        return Experience(time_step=time_step,
                          action=torch.zeros(1, 1),
                          rollout_info={
                              "source": torch.tensor([source]),
                              "target": torch.tensor([float(value + 1)])
                          })

    def _add_experiences(self, algorithm, source, count):
        for i in range(count):
            algorithm.observe_for_replay(
                self._experience(source=source, value=i % 3 + 1))

    def _parameters(self, algorithm):
        return [
            parameter.detach().clone()
            for parameter in algorithm._mlp.parameters()
        ]

    def test_disabled_preserves_single_buffer_training(self):
        algorithm = _TinyMlpAlgorithm(self._config(use_offline_buffer=False))
        self._add_experiences(algorithm, source=0, count=16)
        parameters_before = self._parameters(algorithm)

        trained_steps = algorithm.train_from_replay_buffer()

        expected_steps = (self._BATCH_SIZE * self._UTD * self._BATCH_LENGTH)
        self.assertEqual(trained_steps, expected_steps)
        self.assertEqual(len(algorithm.seen_sources), expected_steps)
        self.assertEqual(set(algorithm.seen_sources), {0})
        self.assertIsNone(algorithm._offline_replay_buffer)
        self.assertTrue(
            any(not torch.equal(before, after) for before, after in zip(
                parameters_before, algorithm._mlp.parameters())))

    def test_boundary_freezes_offline_and_keeps_online_separate(self):
        algorithm = _TinyMlpAlgorithm(self._config(use_offline_buffer=True))
        self._add_experiences(algorithm, source=0, count=16)
        collected_buffer = algorithm._replay_buffer
        parameters_before = self._parameters(algorithm)

        update_progress("iterations", 9)
        algorithm._maybe_activate_collected_offline_buffer()
        self.assertIs(algorithm._replay_buffer, collected_buffer)
        self.assertFalse(algorithm.has_offline)
        self.assertEqual(algorithm.train_from_replay_buffer(), 0)
        self.assertEqual(algorithm.seen_sources, [])
        self.assertTrue(
            all(
                torch.equal(before, after) for before, after in zip(
                    parameters_before, algorithm._mlp.parameters())))

        update_progress("iterations", 10)
        algorithm._maybe_activate_collected_offline_buffer()
        self.assertIs(algorithm._offline_replay_buffer, collected_buffer)
        self.assertIsNot(algorithm._replay_buffer, collected_buffer)
        self.assertTrue(algorithm.has_offline)
        self.assertEqual(int(algorithm._offline_replay_buffer.total_size), 16)
        self.assertEqual(int(algorithm._replay_buffer.total_size), 0)

        self._add_experiences(algorithm, source=1, count=8)
        self.assertEqual(int(algorithm._offline_replay_buffer.total_size), 16)
        self.assertEqual(int(algorithm._replay_buffer.total_size), 8)

        offline_buffer = algorithm._offline_replay_buffer
        online_buffer = algorithm._replay_buffer
        algorithm._has_offline = False  # Simulate the non-checkpointed flag.
        update_progress("iterations", 11)
        algorithm._maybe_activate_collected_offline_buffer()
        self.assertIs(algorithm._offline_replay_buffer, offline_buffer)
        self.assertIs(algorithm._replay_buffer, online_buffer)
        self.assertTrue(algorithm.has_offline)

        replay_paths = dict(
            checkpoint_utils.Checkpointer._iter_replay_buffers(algorithm))
        self.assertIn("_replay_buffer", replay_paths)
        self.assertIn("_offline_replay_buffer", replay_paths)

    def test_enabled_trains_mlp_with_exact_50_50_mix(self):
        algorithm = _TinyMlpAlgorithm(self._config(use_offline_buffer=True))
        self._add_experiences(algorithm, source=0, count=16)
        update_progress("iterations", 10)
        algorithm._maybe_activate_collected_offline_buffer()
        self._add_experiences(algorithm, source=1, count=16)
        parameters_before = self._parameters(algorithm)

        trained_steps = algorithm.train_from_replay_buffer()

        expected_per_source = (self._BATCH_SIZE // 2 * self._UTD *
                               self._BATCH_LENGTH)
        self.assertEqual(trained_steps, 2 * expected_per_source)
        self.assertEqual(algorithm.seen_sources.count(0), expected_per_source)
        self.assertEqual(algorithm.seen_sources.count(1), expected_per_source)
        # Online and offline halves of each combined update share a gradient
        # step, allowing algorithms to reuse one sampled target-critic subset.
        self.assertEqual(algorithm.seen_grad_steps, [0, 0, 1, 1])
        self.assertTrue(
            any(not torch.equal(before, after) for before, after in zip(
                parameters_before, algorithm._mlp.parameters())))


if __name__ == "__main__":
    alf.test.main()
