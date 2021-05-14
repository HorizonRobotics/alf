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
import pprint
import torch

import alf
import alf.data_structures as ds
from alf.algorithms.predictive_representation_learner import (
    PredictiveRepresentationLearner, PredictiveRepresentationLearnerInfo,
    SimpleDecoder)
from alf.experience_replayers.replay_buffer import ReplayBuffer, BatchInfo
from alf.networks import EncodingNetwork, LSTMEncodingNetwork
from alf.utils import common, dist_utils


class PredictiveRepresentationLearnerTest(alf.test.TestCase):
    def test_preprocess_experience(self):
        """
        The following summarizes how the data is generated:

        .. code-block:: python

            # position:   01234567890123
            step_type0 = 'FMMMLFMMLFMMMM'
            step_type1 = 'FMMMMMLFMMMMLF'
            reward = position if train_reward_function and td_steps!=-1
                     else position * (step_type == LAST)
            action = t + 1 for env 0
                     t for env 1

        """
        num_unroll_steps = 4
        batch_size = 2
        obs_dim = 3
        observation_spec = alf.TensorSpec([obs_dim])
        action_spec = alf.BoundedTensorSpec((1, ),
                                            minimum=0,
                                            maximum=1,
                                            dtype=torch.float32)
        reward_spec = alf.TensorSpec(())
        time_step_spec = ds.time_step_spec(observation_spec, action_spec,
                                           reward_spec)

        repr_learner = PredictiveRepresentationLearner(
            observation_spec,
            action_spec,
            num_unroll_steps=num_unroll_steps,
            decoder_ctor=partial(
                SimpleDecoder,
                target_field='reward',
                decoder_net_ctor=partial(
                    EncodingNetwork, fc_layer_params=(4, ))),
            encoding_net_ctor=LSTMEncodingNetwork,
            dynamics_net_ctor=LSTMEncodingNetwork)

        time_step = common.zero_tensor_from_nested_spec(
            time_step_spec, batch_size)
        state = repr_learner.get_initial_predict_state(batch_size)
        alg_step = repr_learner.rollout_step(time_step, state)
        alg_step = alg_step._replace(output=torch.tensor([[1.], [0.]]))
        alg_step_spec = dist_utils.extract_spec(alg_step)

        experience = ds.make_experience(time_step, alg_step, state)
        experience_spec = ds.make_experience(time_step_spec, alg_step_spec,
                                             repr_learner.train_state_spec)
        replay_buffer = ReplayBuffer(
            data_spec=experience_spec,
            num_environments=batch_size,
            max_length=16,
            keep_episodic_info=True)

        #             01234567890123
        step_type0 = 'FMMMLFMMLFMMMM'
        step_type1 = 'FMMMMMLFMMMMLF'

        prev_action = time_step.prev_action

        for i in range(len(step_type0)):
            step_type = [step_type0[i], step_type1[i]]
            step_type = [
                ds.StepType.MID if c == 'M' else
                (ds.StepType.FIRST if c == 'F' else ds.StepType.LAST)
                for c in step_type
            ]
            step_type = torch.tensor(step_type, dtype=torch.int32)
            reward = reward = torch.full([batch_size], float(i))
            time_step = time_step._replace(
                discount=(step_type != ds.StepType.LAST).to(torch.float32),
                step_type=step_type,
                observation=torch.tensor([[i, i + 1, i], [i + 1, i, i]],
                                         dtype=torch.float32),
                reward=reward,
                prev_action=prev_action,
                env_id=torch.arange(batch_size, dtype=torch.int32))
            alg_step = repr_learner.rollout_step(time_step, state)
            alg_step = alg_step._replace(output=i + torch.tensor([[1.], [0.]]))
            prev_action = alg_step.output
            experience = ds.make_experience(time_step, alg_step, state)
            replay_buffer.add_batch(experience)
            state = alg_step.state

        env_ids = torch.tensor([0] * 14 + [1] * 14, dtype=torch.int64)
        positions = torch.tensor(
            list(range(14)) + list(range(14)), dtype=torch.int64)
        experience = replay_buffer.get_field(None,
                                             env_ids.unsqueeze(-1).cpu(),
                                             positions.unsqueeze(-1).cpu())
        processed_experience, processed_rollout_info = repr_learner.preprocess_experience(
            experience, experience.rollout_info,
            BatchInfo(
                env_ids=env_ids,
                positions=positions,
                replay_buffer=replay_buffer))
        pprint.pprint(processed_rollout_info)

        # yapf: disable
        expected = PredictiveRepresentationLearnerInfo(
            action=torch.tensor(
               [[[ 1.,  2.,  3.,  4.]],
                [[ 2.,  3.,  4.,  4.]],
                [[ 3.,  4.,  4.,  4.]],
                [[ 4.,  4.,  4.,  4.]],
                [[ 4.,  4.,  4.,  4.]],
                [[ 6.,  7.,  8.,  8.]],
                [[ 7.,  8.,  8.,  8.]],
                [[ 8.,  8.,  8.,  8.]],
                [[ 8.,  8.,  8.,  8.]],
                [[10., 11., 12., 13.]],
                [[11., 12., 13., 13.]],
                [[12., 13., 13., 13.]],
                [[13., 13., 13., 13.]],
                [[13., 13., 13., 13.]],
                [[ 0.,  1.,  2.,  3.]],
                [[ 1.,  2.,  3.,  4.]],
                [[ 2.,  3.,  4.,  5.]],
                [[ 3.,  4.,  5.,  5.]],
                [[ 4.,  5.,  5.,  5.]],
                [[ 5.,  5.,  5.,  5.]],
                [[ 5.,  5.,  5.,  5.]],
                [[ 7.,  8.,  9., 10.]],
                [[ 8.,  9., 10., 11.]],
                [[ 9., 10., 11., 11.]],
                [[10., 11., 11., 11.]],
                [[11., 11., 11., 11.]],
                [[11., 11., 11., 11.]],
                [[12., 12., 12., 12.]]]).unsqueeze(-1),
            mask=torch.tensor(
               [[[ True,  True,  True,  True,  True]],
                [[ True,  True,  True,  True, False]],
                [[ True,  True,  True, False, False]],
                [[ True,  True, False, False, False]],
                [[ True, False, False, False, False]],
                [[ True,  True,  True,  True, False]],
                [[ True,  True,  True, False, False]],
                [[ True,  True, False, False, False]],
                [[ True, False, False, False, False]],
                [[ True,  True,  True,  True,  True]],
                [[ True,  True,  True,  True, False]],
                [[ True,  True,  True, False, False]],
                [[ True,  True, False, False, False]],
                [[ True, False, False, False, False]],
                [[ True,  True,  True,  True,  True]],
                [[ True,  True,  True,  True,  True]],
                [[ True,  True,  True,  True,  True]],
                [[ True,  True,  True,  True, False]],
                [[ True,  True,  True, False, False]],
                [[ True,  True, False, False, False]],
                [[ True, False, False, False, False]],
                [[ True,  True,  True,  True,  True]],
                [[ True,  True,  True,  True,  True]],
                [[ True,  True,  True,  True, False]],
                [[ True,  True,  True, False, False]],
                [[ True,  True, False, False, False]],
                [[ True, False, False, False, False]],
                [[ True, False, False, False, False]]]),
            target=torch.tensor(
               [[[ 0.,  1.,  2.,  3.,  4.]],
                [[ 1.,  2.,  3.,  4.,  4.]],
                [[ 2.,  3.,  4.,  4.,  4.]],
                [[ 3.,  4.,  4.,  4.,  4.]],
                [[ 4.,  4.,  4.,  4.,  4.]],
                [[ 5.,  6.,  7.,  8.,  8.]],
                [[ 6.,  7.,  8.,  8.,  8.]],
                [[ 7.,  8.,  8.,  8.,  8.]],
                [[ 8.,  8.,  8.,  8.,  8.]],
                [[ 9., 10., 11., 12., 13.]],
                [[10., 11., 12., 13., 13.]],
                [[11., 12., 13., 13., 13.]],
                [[12., 13., 13., 13., 13.]],
                [[13., 13., 13., 13., 13.]],
                [[ 0.,  1.,  2.,  3.,  4.]],
                [[ 1.,  2.,  3.,  4.,  5.]],
                [[ 2.,  3.,  4.,  5.,  6.]],
                [[ 3.,  4.,  5.,  6.,  6.]],
                [[ 4.,  5.,  6.,  6.,  6.]],
                [[ 5.,  6.,  6.,  6.,  6.]],
                [[ 6.,  6.,  6.,  6.,  6.]],
                [[ 7.,  8.,  9., 10., 11.]],
                [[ 8.,  9., 10., 11., 12.]],
                [[ 9., 10., 11., 12., 12.]],
                [[10., 11., 12., 12., 12.]],
                [[11., 12., 12., 12., 12.]],
                [[12., 12., 12., 12., 12.]],
                [[13., 13., 13., 13., 13.]]]))
        # yapf: enable

        alf.nest.map_structure(lambda x, y: self.assertEqual(x, y),
                               processed_rollout_info, expected)


if __name__ == '__main__':
    alf.test.main()
