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

from functools import partial
import torch
from torch import nn

import alf
from alf.algorithms.data_transformer import FrameStacker
from alf.algorithms.muzero_algorithm import MuzeroAlgorithm, MuzeroInfo, OffPolicyAlgorithm
from alf.algorithms.mcts_algorithm import MCTSInfo, MCTSState
from alf.algorithms.mcts_models import get_unique_num_actions, MCTSModel, ModelOutput, ModelTarget
import alf.data_structures as ds
from alf.experience_replayers.replay_buffer import ReplayBuffer, BatchInfo
from alf.utils import common, dist_utils


class MockMCTSModel(nn.Module):
    def __init__(self, observation_spec, action_spec, scale):
        super().__init__()
        self._observation_spec = observation_spec
        self._action_spec = action_spec
        self._num_actions = get_unique_num_actions(action_spec)
        self._scale = scale

    def initial_inference(self, observation):
        return self._predict(observation)

    def recurrent_inference(self, state, action):
        return self._predict(state + 1.)

    def _predict(self, state):
        action_probs = state[:, 3:5] * self._scale
        return ModelOutput(
            value=state[:, -1] * 0.5 * self._scale,
            actions=(),
            action_probs=action_probs,
            state=state)


_mcts_model_id = 0


def _create_mcts_model(observation_spec, action_spec, debug_summaries):
    global _mcts_model_id
    scale = 1 + _mcts_model_id % 2
    _mcts_model_id += 1
    return MockMCTSModel(observation_spec, action_spec, scale)


class MockMCTSAlgorithm(OffPolicyAlgorithm):
    def __init__(self, observation_spec, action_spec, debug_summaries):
        super().__init__(
            observation_spec,
            action_spec,
            train_state_spec=MCTSState(
                steps=alf.TensorSpec((), dtype=torch.int64)),
            debug_summaries=debug_summaries)
        self._model = None

    @property
    def discount(self):
        return 0.5

    def set_model(self, model: MCTSModel):
        self._model = model

    def predict_step(self, time_step, state):
        model_output = self._model.initial_inference(time_step.observation)
        action_id = model_output.action_probs.argmax(dim=1)
        if isinstance(model_output.actions, torch.Tensor):
            action = model_output.actions[torch.arange(model_output.actions.
                                                       shape[0]), action_id]
        else:
            action = action_id
        return ds.AlgStep(
            output=action,
            state=MCTSState(steps=state.steps + 1),
            info=MCTSInfo(
                candidate_actions=model_output.actions,
                candidate_action_policy=model_output.action_probs,
                value=model_output.value,
            ))


class MuzeroAlgorithmTest(alf.test.TestCase):
    def _test_preprocess_experience(self, train_reward_function, td_steps,
                                    reanalyze_ratio, expected):
        """
        The following summarizes how the data is generated:

        .. code-block:: python

            # position:   01234567890123
            step_type0 = 'FMMMLFMMLFMMMM'
            step_type1 = 'FMMMMMLFMMMMLF'
            scale = 1. for current model
                    2. for target model
            observation = [position] * 3
            reward = position if train_reward_function and td_steps!=-1
                     else position * (step_type == LAST)
            value = 0.5 * position * scale
            action_probs = scale * [position, position+1, position] for env 0
                           scale * [position+1, position, position] for env 1
            action = 1 for env 0
                     0 for env 1

        """
        reanalyze_td_steps = 2

        num_unroll_steps = 4
        batch_size = 2
        obs_dim = 3

        observation_spec = alf.TensorSpec([obs_dim])
        action_spec = alf.BoundedTensorSpec((),
                                            minimum=0,
                                            maximum=1,
                                            dtype=torch.int32)
        reward_spec = alf.TensorSpec(())
        time_step_spec = ds.time_step_spec(observation_spec, action_spec,
                                           reward_spec)

        global _mcts_model_id
        _mcts_model_id = 0
        muzero = MuzeroAlgorithm(
            observation_spec,
            action_spec,
            model_ctor=_create_mcts_model,
            mcts_algorithm_ctor=MockMCTSAlgorithm,
            num_unroll_steps=num_unroll_steps,
            td_steps=td_steps,
            train_game_over_function=True,
            train_reward_function=train_reward_function,
            reanalyze_ratio=reanalyze_ratio,
            reanalyze_td_steps=reanalyze_td_steps,
            data_transformer_ctor=partial(FrameStacker, stack_size=2))

        data_transformer = FrameStacker(observation_spec, stack_size=2)
        time_step = common.zero_tensor_from_nested_spec(
            time_step_spec, batch_size)
        dt_state = common.zero_tensor_from_nested_spec(
            data_transformer.state_spec, batch_size)
        state = muzero.get_initial_predict_state(batch_size)
        transformed_time_step, dt_state = data_transformer.transform_timestep(
            time_step, dt_state)
        alg_step = muzero.rollout_step(transformed_time_step, state)
        alg_step_spec = dist_utils.extract_spec(alg_step)

        experience = ds.make_experience(time_step, alg_step, state)
        experience_spec = ds.make_experience(time_step_spec, alg_step_spec,
                                             muzero.train_state_spec)
        replay_buffer = ReplayBuffer(
            data_spec=experience_spec,
            num_environments=batch_size,
            max_length=16,
            keep_episodic_info=True)

        #             01234567890123
        step_type0 = 'FMMMLFMMLFMMMM'
        step_type1 = 'FMMMMMLFMMMMLF'

        dt_state = common.zero_tensor_from_nested_spec(
            data_transformer.state_spec, batch_size)
        for i in range(len(step_type0)):
            step_type = [step_type0[i], step_type1[i]]
            step_type = [
                ds.StepType.MID if c == 'M' else
                (ds.StepType.FIRST if c == 'F' else ds.StepType.LAST)
                for c in step_type
            ]
            step_type = torch.tensor(step_type, dtype=torch.int32)
            reward = reward = torch.full([batch_size], float(i))
            if not train_reward_function or td_steps == -1:
                reward = reward * (step_type == ds.StepType.LAST).to(
                    torch.float32)
            time_step = time_step._replace(
                discount=(step_type != ds.StepType.LAST).to(torch.float32),
                step_type=step_type,
                observation=torch.tensor([[i, i + 1, i], [i + 1, i, i]],
                                         dtype=torch.float32),
                reward=reward,
                env_id=torch.arange(batch_size, dtype=torch.int32))
            transformed_time_step, dt_state = data_transformer.transform_timestep(
                time_step, dt_state)
            alg_step = muzero.rollout_step(transformed_time_step, state)
            experience = ds.make_experience(time_step, alg_step, state)
            replay_buffer.add_batch(experience)
            state = alg_step.state

        env_ids = torch.tensor([0] * 14 + [1] * 14, dtype=torch.int64)
        positions = torch.tensor(
            list(range(14)) + list(range(14)), dtype=torch.int64)
        experience = replay_buffer.get_field(None,
                                             env_ids.unsqueeze(-1).cpu(),
                                             positions.unsqueeze(-1).cpu())
        experience = experience._replace(
            replay_buffer=replay_buffer,
            batch_info=BatchInfo(env_ids=env_ids, positions=positions),
            rollout_info_field='rollout_info')
        processed_experience = muzero.preprocess_experience(experience)
        import pprint
        pprint.pprint(processed_experience.rollout_info)
        alf.nest.map_structure(lambda x, y: self.assertEqual(x, y),
                               processed_experience.rollout_info, expected)

    def get_exptected_info(self, sparse_reward):
        # yapf: disable
        expected = MuzeroInfo(
            action=torch.tensor([
                [[1, 1, 1, 1]],
                [[1, 1, 1, 1]],
                [[1, 1, 1, 1]],
                [[1, 1, 1, 1]],
                [[1, 1, 1, 1]],
                [[1, 1, 1, 1]],
                [[1, 1, 1, 1]],
                [[1, 1, 1, 1]],
                [[1, 1, 1, 1]],
                [[1, 1, 1, 1]],
                [[1, 1, 1, 1]],
                [[1, 1, 1, 1]],
                [[1, 1, 1, 1]],
                [[1, 1, 1, 1]],
                [[0, 0, 0, 0]],
                [[0, 0, 0, 0]],
                [[0, 0, 0, 0]],
                [[0, 0, 0, 0]],
                [[0, 0, 0, 0]],
                [[0, 0, 0, 0]],
                [[0, 0, 0, 0]],
                [[0, 0, 0, 0]],
                [[0, 0, 0, 0]],
                [[0, 0, 0, 0]],
                [[0, 0, 0, 0]],
                [[0, 0, 0, 0]],
                [[0, 0, 0, 0]],
                [[0, 0, 0, 0]]]),
            value=torch.tensor([
                [0.0000],
                [0.5000],
                [1.0000],
                [1.5000],
                [2.0000],
                [2.5000],
                [3.0000],
                [3.5000],
                [4.0000],
                [4.5000],
                [5.0000],
                [5.5000],
                [6.0000],
                [6.5000],
                [0.0000],
                [0.5000],
                [1.0000],
                [1.5000],
                [2.0000],
                [2.5000],
                [3.0000],
                [3.5000],
                [4.0000],
                [4.5000],
                [5.0000],
                [5.5000],
                [6.0000],
                [6.5000]]),
            target=ModelTarget(
                reward=torch.tensor([
                    [[ 0.,  1.,  2.,  3.,  4.]],
                    [[ 1.,  2.,  3.,  4.,  0.]],
                    [[ 2.,  3.,  4.,  0.,  0.]],
                    [[ 3.,  4.,  0.,  0.,  0.]],
                    [[ 4.,  0.,  0.,  0.,  0.]],
                    [[ 5.,  6.,  7.,  8.,  0.]],
                    [[ 6.,  7.,  8.,  0.,  0.]],
                    [[ 7.,  8.,  0.,  0.,  0.]],
                    [[ 8.,  0.,  0.,  0.,  0.]],
                    [[ 9., 10., 11., 12., 13.]],
                    [[10., 11., 12., 13.,  0.]],
                    [[11., 12., 13.,  0.,  0.]],
                    [[12., 13.,  0.,  0.,  0.]],
                    [[13.,  0.,  0.,  0.,  0.]],
                    [[ 0.,  1.,  2.,  3.,  4.]],
                    [[ 1.,  2.,  3.,  4.,  5.]],
                    [[ 2.,  3.,  4.,  5.,  6.]],
                    [[ 3.,  4.,  5.,  6.,  0.]],
                    [[ 4.,  5.,  6.,  0.,  0.]],
                    [[ 5.,  6.,  0.,  0.,  0.]],
                    [[ 6.,  0.,  0.,  0.,  0.]],
                    [[ 7.,  8.,  9., 10., 11.]],
                    [[ 8.,  9., 10., 11., 12.]],
                    [[ 9., 10., 11., 12.,  0.]],
                    [[10., 11., 12.,  0.,  0.]],
                    [[11., 12.,  0.,  0.,  0.]],
                    [[12.,  0.,  0.,  0.,  0.]],
                    [[13.,  0.,  0.,  0.,  0.]]]),
                action=(),
                action_policy=torch.tensor([
                  [[[ 0.,  1.],
                    [ 1.,  2.],
                    [ 2.,  3.],
                    [ 3.,  4.],
                    [ 4.,  5.]]],
                  [[[ 1.,  2.],
                    [ 2.,  3.],
                    [ 3.,  4.],
                    [ 4.,  5.],
                    [ 4.,  5.]]],
                  [[[ 2.,  3.],
                    [ 3.,  4.],
                    [ 4.,  5.],
                    [ 4.,  5.],
                    [ 4.,  5.]]],
                  [[[ 3.,  4.],
                    [ 4.,  5.],
                    [ 4.,  5.],
                    [ 4.,  5.],
                    [ 4.,  5.]]],
                  [[[ 4.,  5.],
                    [ 4.,  5.],
                    [ 4.,  5.],
                    [ 4.,  5.],
                    [ 4.,  5.]]],
                  [[[ 5.,  6.],
                    [ 6.,  7.],
                    [ 7.,  8.],
                    [ 8.,  9.],
                    [ 8.,  9.]]],
                  [[[ 6.,  7.],
                    [ 7.,  8.],
                    [ 8.,  9.],
                    [ 8.,  9.],
                    [ 8.,  9.]]],
                  [[[ 7.,  8.],
                    [ 8.,  9.],
                    [ 8.,  9.],
                    [ 8.,  9.],
                    [ 8.,  9.]]],
                  [[[ 8.,  9.],
                    [ 8.,  9.],
                    [ 8.,  9.],
                    [ 8.,  9.],
                    [ 8.,  9.]]],
                  [[[ 9., 10.],
                    [10., 11.],
                    [11., 12.],
                    [12., 13.],
                    [13., 14.]]],
                  [[[10., 11.],
                    [11., 12.],
                    [12., 13.],
                    [13., 14.],
                    [13., 14.]]],
                  [[[11., 12.],
                    [12., 13.],
                    [13., 14.],
                    [13., 14.],
                    [13., 14.]]],
                  [[[12., 13.],
                    [13., 14.],
                    [13., 14.],
                    [13., 14.],
                    [13., 14.]]],
                  [[[13., 14.],
                    [13., 14.],
                    [13., 14.],
                    [13., 14.],
                    [13., 14.]]],
                  [[[ 1.,  0.],
                    [ 2.,  1.],
                    [ 3.,  2.],
                    [ 4.,  3.],
                    [ 5.,  4.]]],
                  [[[ 2.,  1.],
                    [ 3.,  2.],
                    [ 4.,  3.],
                    [ 5.,  4.],
                    [ 6.,  5.]]],
                  [[[ 3.,  2.],
                    [ 4.,  3.],
                    [ 5.,  4.],
                    [ 6.,  5.],
                    [ 7.,  6.]]],
                  [[[ 4.,  3.],
                    [ 5.,  4.],
                    [ 6.,  5.],
                    [ 7.,  6.],
                    [ 7.,  6.]]],
                  [[[ 5.,  4.],
                    [ 6.,  5.],
                    [ 7.,  6.],
                    [ 7.,  6.],
                    [ 7.,  6.]]],
                  [[[ 6.,  5.],
                    [ 7.,  6.],
                    [ 7.,  6.],
                    [ 7.,  6.],
                    [ 7.,  6.]]],
                  [[[ 7.,  6.],
                    [ 7.,  6.],
                    [ 7.,  6.],
                    [ 7.,  6.],
                    [ 7.,  6.]]],
                  [[[ 8.,  7.],
                    [ 9.,  8.],
                    [10.,  9.],
                    [11., 10.],
                    [12., 11.]]],
                  [[[ 9.,  8.],
                    [10.,  9.],
                    [11., 10.],
                    [12., 11.],
                    [13., 12.]]],
                  [[[10.,  9.],
                    [11., 10.],
                    [12., 11.],
                    [13., 12.],
                    [13., 12.]]],
                  [[[11., 10.],
                    [12., 11.],
                    [13., 12.],
                    [13., 12.],
                    [13., 12.]]],
                  [[[12., 11.],
                    [13., 12.],
                    [13., 12.],
                    [13., 12.],
                    [13., 12.]]],
                  [[[13., 12.],
                    [13., 12.],
                    [13., 12.],
                    [13., 12.],
                    [13., 12.]]],
                  [[[14., 13.],
                    [14., 13.],
                    [14., 13.],
                    [14., 13.],
                    [14., 13.]]]]),
                game_over=torch.tensor([
                    [[False, False, False, False,  True]],
                    [[False, False, False,  True,  True]],
                    [[False, False,  True,  True,  True]],
                    [[False,  True,  True,  True,  True]],
                    [[ True,  True,  True,  True,  True]],
                    [[False, False, False,  True,  True]],
                    [[False, False,  True,  True,  True]],
                    [[False,  True,  True,  True,  True]],
                    [[ True,  True,  True,  True,  True]],
                    [[False, False, False, False, False]],
                    [[False, False, False, False, False]],
                    [[False, False, False, False, False]],
                    [[False, False, False, False, False]],
                    [[False, False, False, False, False]],
                    [[False, False, False, False, False]],
                    [[False, False, False, False, False]],
                    [[False, False, False, False,  True]],
                    [[False, False, False,  True,  True]],
                    [[False, False,  True,  True,  True]],
                    [[False,  True,  True,  True,  True]],
                    [[ True,  True,  True,  True,  True]],
                    [[False, False, False, False, False]],
                    [[False, False, False, False,  True]],
                    [[False, False, False,  True,  True]],
                    [[False, False,  True,  True,  True]],
                    [[False,  True,  True,  True,  True]],
                    [[ True,  True,  True,  True,  True]],
                    [[False, False, False, False, False]]])))
        if sparse_reward:
            expected = expected._replace(target=expected.target._replace(
                reward=torch.tensor([
                    [[ 0.,  0.,  0.,  0.,  4.]],
                    [[ 0.,  0.,  0.,  4.,  0.]],
                    [[ 0.,  0.,  4.,  0.,  0.]],
                    [[ 0.,  4.,  0.,  0.,  0.]],
                    [[ 4.,  0.,  0.,  0.,  0.]],
                    [[ 0.,  0.,  0.,  8.,  0.]],
                    [[ 0.,  0.,  8.,  0.,  0.]],
                    [[ 0.,  8.,  0.,  0.,  0.]],
                    [[ 8.,  0.,  0.,  0.,  0.]],
                    [[ 0.,  0.,  0.,  0.,  0.]],
                    [[ 0.,  0.,  0.,  0.,  0.]],
                    [[ 0.,  0.,  0.,  0.,  0.]],
                    [[ 0.,  0.,  0.,  0.,  0.]],
                    [[ 0.,  0.,  0.,  0.,  0.]],
                    [[ 0.,  0.,  0.,  0.,  0.]],
                    [[ 0.,  0.,  0.,  0.,  0.]],
                    [[ 0.,  0.,  0.,  0.,  6.]],
                    [[ 0.,  0.,  0.,  6.,  0.]],
                    [[ 0.,  0.,  6.,  0.,  0.]],
                    [[ 0.,  6.,  0.,  0.,  0.]],
                    [[ 6.,  0.,  0.,  0.,  0.]],
                    [[ 0.,  0.,  0.,  0.,  0.]],
                    [[ 0.,  0.,  0.,  0., 12.]],
                    [[ 0.,  0.,  0., 12.,  0.]],
                    [[ 0.,  0., 12.,  0.,  0.]],
                    [[ 0., 12.,  0.,  0.,  0.]],
                    [[12.,  0.,  0.,  0.,  0.]],
                    [[ 0.,  0.,  0.,  0.,  0.]]])))
        return expected

    def test_bootstrap_return_with_reward_function(self):
        expected = self.get_exptected_info(False)
        expected = expected._replace(target=expected.target._replace(
            value=torch.tensor([
                [[ 2.2500,  3.8750,  5.0000,  4.0000,  0.0000]],
                [[ 3.8750,  5.0000,  4.0000,  0.0000,  0.0000]],
                [[ 5.0000,  4.0000,  0.0000,  0.0000,  0.0000]],
                [[ 4.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                [[10.3750, 11.0000,  8.0000,  0.0000,  0.0000]],
                [[11.0000,  8.0000,  0.0000,  0.0000,  0.0000]],
                [[ 8.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                [[16.8750, 18.5000, 20.1250, 16.2500,  6.5000]],
                [[18.5000, 20.1250, 16.2500,  6.5000,  6.5000]],
                [[20.1250, 16.2500,  6.5000,  6.5000,  6.5000]],
                [[16.2500,  6.5000,  6.5000,  6.5000,  6.5000]],
                [[ 6.5000,  6.5000,  6.5000,  6.5000,  6.5000]],
                [[ 2.2500,  3.8750,  5.5000,  7.1250,  8.0000]],
                [[ 3.8750,  5.5000,  7.1250,  8.0000,  6.0000]],
                [[ 5.5000,  7.1250,  8.0000,  6.0000,  0.0000]],
                [[ 7.1250,  8.0000,  6.0000,  0.0000,  0.0000]],
                [[ 8.0000,  6.0000,  0.0000,  0.0000,  0.0000]],
                [[ 6.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                [[13.6250, 15.2500, 16.8750, 17.0000, 12.0000]],
                [[15.2500, 16.8750, 17.0000, 12.0000,  0.0000]],
                [[16.8750, 17.0000, 12.0000,  0.0000,  0.0000]],
                [[17.0000, 12.0000,  0.0000,  0.0000,  0.0000]],
                [[12.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                [[ 6.5000,  6.5000,  6.5000,  6.5000,  6.5000]]])))
        self._test_preprocess_experience(
            train_reward_function=True,
            td_steps=2,
            reanalyze_ratio=0.,
            expected=expected)

    def test_bootstrap_return_without_reward_function(self):
        expected = self.get_exptected_info(False)
        expected = expected._replace(target=expected.target._replace(
            reward=(),
            value=torch.tensor([
                [[ 0.2500,  0.3750,  2.0000,  4.0000,  4.0000]],
                [[ 0.3750,  2.0000,  4.0000,  4.0000,  4.0000]],
                [[ 2.0000,  4.0000,  4.0000,  4.0000,  4.0000]],
                [[ 4.0000,  4.0000,  4.0000,  4.0000,  4.0000]],
                [[ 4.0000,  4.0000,  4.0000,  4.0000,  4.0000]],
                [[ 0.8750,  4.0000,  8.0000,  8.0000,  8.0000]],
                [[ 4.0000,  8.0000,  8.0000,  8.0000,  8.0000]],
                [[ 8.0000,  8.0000,  8.0000,  8.0000,  8.0000]],
                [[ 8.0000,  8.0000,  8.0000,  8.0000,  8.0000]],
                [[ 1.3750,  1.5000,  1.6250,  3.2500,  6.5000]],
                [[ 1.5000,  1.6250,  3.2500,  6.5000,  6.5000]],
                [[ 1.6250,  3.2500,  6.5000,  6.5000,  6.5000]],
                [[ 3.2500,  6.5000,  6.5000,  6.5000,  6.5000]],
                [[ 6.5000,  6.5000,  6.5000,  6.5000,  6.5000]],
                [[ 0.2500,  0.3750,  0.5000,  0.6250,  3.0000]],
                [[ 0.3750,  0.5000,  0.6250,  3.0000,  6.0000]],
                [[ 0.5000,  0.6250,  3.0000,  6.0000,  6.0000]],
                [[ 0.6250,  3.0000,  6.0000,  6.0000,  6.0000]],
                [[ 3.0000,  6.0000,  6.0000,  6.0000,  6.0000]],
                [[ 6.0000,  6.0000,  6.0000,  6.0000,  6.0000]],
                [[ 6.0000,  6.0000,  6.0000,  6.0000,  6.0000]],
                [[ 1.1250,  1.2500,  1.3750,  6.0000, 12.0000]],
                [[ 1.2500,  1.3750,  6.0000, 12.0000, 12.0000]],
                [[ 1.3750,  6.0000, 12.0000, 12.0000, 12.0000]],
                [[ 6.0000, 12.0000, 12.0000, 12.0000, 12.0000]],
                [[12.0000, 12.0000, 12.0000, 12.0000, 12.0000]],
                [[12.0000, 12.0000, 12.0000, 12.0000, 12.0000]],
                [[ 6.5000,  6.5000,  6.5000,  6.5000,  6.5000]]])))
        self._test_preprocess_experience(
            train_reward_function=False,
            td_steps=2,
            reanalyze_ratio=0.,
            expected=expected)

    def test_monte_carla_return_with_reward_function(self):
        expected = self.get_exptected_info(True)
        expected = expected._replace(target=expected.target._replace(
            value=torch.tensor([
                [[ 0.5000,  1.0000,  2.0000,  4.0000,  0.0000]],
                [[ 1.0000,  2.0000,  4.0000,  0.0000,  0.0000]],
                [[ 2.0000,  4.0000,  0.0000,  0.0000,  0.0000]],
                [[ 4.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                [[ 2.0000,  4.0000,  8.0000,  0.0000,  0.0000]],
                [[ 4.0000,  8.0000,  0.0000,  0.0000,  0.0000]],
                [[ 8.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                [[ 0.40625, 0.8125,  1.6250,  3.2500,  3.2500]],
                [[ 0.8125,  1.6250,  3.2500,  3.2500,  3.2500]],
                [[ 1.6250,  3.2500,  3.2500,  3.2500,  3.2500]],
                [[ 3.2500,  3.2500,  3.2500,  3.2500,  3.2500]],
                [[ 3.2500,  3.2500,  3.2500,  3.2500,  3.2500]],
                [[ 0.1875,  0.3750,  0.7500,  1.5000,  3.0000]],
                [[ 0.3750,  0.7500,  1.5000,  3.0000,  6.0000]],
                [[ 0.7500,  1.5000,  3.0000,  6.0000,  0.0000]],
                [[ 1.5000,  3.0000,  6.0000,  0.0000,  0.0000]],
                [[ 3.0000,  6.0000,  0.0000,  0.0000,  0.0000]],
                [[ 6.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                [[ 0.7500,  1.5000,  3.0000,  6.0000, 12.0000]],
                [[ 1.5000,  3.0000,  6.0000, 12.0000,  0.0000]],
                [[ 3.0000,  6.0000, 12.0000,  0.0000,  0.0000]],
                [[ 6.0000, 12.0000,  0.0000,  0.0000,  0.0000]],
                [[12.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                [[ 3.2500,  3.2500,  3.2500,  3.2500,  3.2500]]])))
        self._test_preprocess_experience(train_reward_function=True, td_steps=-1, reanalyze_ratio=0., expected=expected)

    def test_monte_carla_return_without_reward_function(self):
        expected = self.get_exptected_info(True)
        expected = expected._replace(target=expected.target._replace(
            reward=(),
            value=torch.tensor([
                [[ 0.5000,  1.0000,  2.0000,  4.0000,  4.0000]],
                [[ 1.0000,  2.0000,  4.0000,  4.0000,  4.0000]],
                [[ 2.0000,  4.0000,  4.0000,  4.0000,  4.0000]],
                [[ 4.0000,  4.0000,  4.0000,  4.0000,  4.0000]],
                [[ 4.0000,  4.0000,  4.0000,  4.0000,  4.0000]],
                [[ 2.0000,  4.0000,  8.0000,  8.0000,  8.0000]],
                [[ 4.0000,  8.0000,  8.0000,  8.0000,  8.0000]],
                [[ 8.0000,  8.0000,  8.0000,  8.0000,  8.0000]],
                [[ 8.0000,  8.0000,  8.0000,  8.0000,  8.0000]],
                [[ 0.40625, 0.8125,  1.6250,  3.2500,  3.2500]],
                [[ 0.8125,  1.6250,  3.2500,  3.2500,  3.2500]],
                [[ 1.6250,  3.2500,  3.2500,  3.2500,  3.2500]],
                [[ 3.2500,  3.2500,  3.2500,  3.2500,  3.2500]],
                [[ 3.2500,  3.2500,  3.2500,  3.2500,  3.2500]],
                [[ 0.1875,  0.3750,  0.7500,  1.5000,  3.0000]],
                [[ 0.3750,  0.7500,  1.5000,  3.0000,  6.0000]],
                [[ 0.7500,  1.5000,  3.0000,  6.0000,  6.0000]],
                [[ 1.5000,  3.0000,  6.0000,  6.0000,  6.0000]],
                [[ 3.0000,  6.0000,  6.0000,  6.0000,  6.0000]],
                [[ 6.0000,  6.0000,  6.0000,  6.0000,  6.0000]],
                [[ 6.0000,  6.0000,  6.0000,  6.0000,  6.0000]],
                [[ 0.7500,  1.5000,  3.0000,  6.0000, 12.0000]],
                [[ 1.5000,  3.0000,  6.0000, 12.0000, 12.0000]],
                [[ 3.0000,  6.0000, 12.0000, 12.0000, 12.0000]],
                [[ 6.0000, 12.0000, 12.0000, 12.0000, 12.0000]],
                [[12.0000, 12.0000, 12.0000, 12.0000, 12.0000]],
                [[12.0000, 12.0000, 12.0000, 12.0000, 12.0000]],
                [[ 3.2500,  3.2500,  3.2500,  3.2500,  3.2500]]])))
        self._test_preprocess_experience(
            train_reward_function=False,
            td_steps=-1,
            reanalyze_ratio=0.,
            expected=expected)

    def test_reanalyze_with_reward_function(self):
        expected = self.get_exptected_info(True)
        expected = expected._replace(
            value=expected.value,
            target=expected.target._replace(
                action_policy=expected.target.action_policy * 2,
                value=torch.tensor([
                    [[ 0.5000,  0.7500,  2.0000,  4.0000,  0.0000]],
                    [[ 0.7500,  2.0000,  4.0000,  0.0000,  0.0000]],
                    [[ 2.0000,  4.0000,  0.0000,  0.0000,  0.0000]],
                    [[ 4.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                    [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                    [[ 1.7500,  4.0000,  8.0000,  0.0000,  0.0000]],
                    [[ 4.0000,  8.0000,  0.0000,  0.0000,  0.0000]],
                    [[ 8.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                    [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                    [[ 2.7500,  3.0000,  3.2500,  6.5000, 13.0000]],
                    [[ 3.0000,  3.2500,  6.5000, 13.0000, 13.0000]],
                    [[ 3.2500,  6.5000, 13.0000, 13.0000, 13.0000]],
                    [[ 6.5000, 13.0000, 13.0000, 13.0000, 13.0000]],
                    [[13.0000, 13.0000, 13.0000, 13.0000, 13.0000]],
                    [[ 0.5000,  0.7500,  1.0000,  1.2500,  3.0000]],
                    [[ 0.7500,  1.0000,  1.2500,  3.0000,  6.0000]],
                    [[ 1.0000,  1.2500,  3.0000,  6.0000,  0.0000]],
                    [[ 1.2500,  3.0000,  6.0000,  0.0000,  0.0000]],
                    [[ 3.0000,  6.0000,  0.0000,  0.0000,  0.0000]],
                    [[ 6.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                    [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                    [[ 2.2500,  2.5000,  2.7500,  6.0000, 12.0000]],
                    [[ 2.5000,  2.7500,  6.0000, 12.0000,  0.0000]],
                    [[ 2.7500,  6.0000, 12.0000,  0.0000,  0.0000]],
                    [[ 6.0000, 12.0000,  0.0000,  0.0000,  0.0000]],
                    [[12.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                    [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                    [[13.0000, 13.0000, 13.0000, 13.0000, 13.0000]]])))
        self._test_preprocess_experience(
            train_reward_function=True,
            td_steps=-1,
            reanalyze_ratio=1.,
            expected=expected)

    def test_reanalyze_without_reward_function(self):
        expected = self.get_exptected_info(True)
        expected = expected._replace(
            value=expected.value,
            target=expected.target._replace(
                reward=(),
                action_policy=expected.target.action_policy * 2,
                value=torch.tensor([
                    [[ 0.5000,  0.7500,  2.0000,  4.0000,  4.0000]],
                    [[ 0.7500,  2.0000,  4.0000,  4.0000,  4.0000]],
                    [[ 2.0000,  4.0000,  4.0000,  4.0000,  4.0000]],
                    [[ 4.0000,  4.0000,  4.0000,  4.0000,  4.0000]],
                    [[ 4.0000,  4.0000,  4.0000,  4.0000,  4.0000]],
                    [[ 1.7500,  4.0000,  8.0000,  8.0000,  8.0000]],
                    [[ 4.0000,  8.0000,  8.0000,  8.0000,  8.0000]],
                    [[ 8.0000,  8.0000,  8.0000,  8.0000,  8.0000]],
                    [[ 8.0000,  8.0000,  8.0000,  8.0000,  8.0000]],
                    [[ 2.7500,  3.0000,  3.2500,  6.5000, 13.0000]],
                    [[ 3.0000,  3.2500,  6.5000, 13.0000, 13.0000]],
                    [[ 3.2500,  6.5000, 13.0000, 13.0000, 13.0000]],
                    [[ 6.5000, 13.0000, 13.0000, 13.0000, 13.0000]],
                    [[13.0000, 13.0000, 13.0000, 13.0000, 13.0000]],
                    [[ 0.5000,  0.7500,  1.0000,  1.2500,  3.0000]],
                    [[ 0.7500,  1.0000,  1.2500,  3.0000,  6.0000]],
                    [[ 1.0000,  1.2500,  3.0000,  6.0000,  6.0000]],
                    [[ 1.2500,  3.0000,  6.0000,  6.0000,  6.0000]],
                    [[ 3.0000,  6.0000,  6.0000,  6.0000,  6.0000]],
                    [[ 6.0000,  6.0000,  6.0000,  6.0000,  6.0000]],
                    [[ 6.0000,  6.0000,  6.0000,  6.0000,  6.0000]],
                    [[ 2.2500,  2.5000,  2.7500,  6.0000, 12.0000]],
                    [[ 2.5000,  2.7500,  6.0000, 12.0000, 12.0000]],
                    [[ 2.7500,  6.0000, 12.0000, 12.0000, 12.0000]],
                    [[ 6.0000, 12.0000, 12.0000, 12.0000, 12.0000]],
                    [[12.0000, 12.0000, 12.0000, 12.0000, 12.0000]],
                    [[12.0000, 12.0000, 12.0000, 12.0000, 12.0000]],
                    [[13.0000, 13.0000, 13.0000, 13.0000, 13.0000]]])))
        self._test_preprocess_experience(
            train_reward_function=False,
            td_steps=-1,
            reanalyze_ratio=1.,
            expected=expected)
        # yapf: enable


if __name__ == '__main__':
    alf.test.main()
