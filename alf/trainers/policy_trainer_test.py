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

import tempfile
import torch

import alf
from alf.algorithms.rl_algorithm_test import MyEnv, MyAlg
from alf.trainers.policy_trainer import Trainer, TrainerConfig, play
from alf.utils import common


class MyTrainer(Trainer):
    def _create_environment(self,
                            nonparallel=False,
                            random_seed=None,
                            register=True):
        env = MyEnv(3)
        if register:
            self._register_env(env)
        return env


class TrainerTest(alf.test.TestCase):
    def test_trainer(self):
        with tempfile.TemporaryDirectory() as root_dir:
            conf = TrainerConfig(
                algorithm_ctor=MyAlg,
                root_dir=root_dir,
                unroll_length=5,
                num_iterations=100)

            # test train
            trainer = MyTrainer(conf)
            self.assertEqual(Trainer.progress(), 0)
            trainer.train()
            self.assertEqual(Trainer.progress(), 1)

            alg = trainer._algorithm
            env = common.get_env()
            time_step = common.get_initial_time_step(env)
            state = alg.get_initial_predict_state(env.batch_size)
            policy_step = alg.rollout_step(time_step, state)
            logits = policy_step.info.logits
            print("logits: ", logits)
            self.assertTrue(torch.all(logits[:, 1] > logits[:, 0]))
            self.assertTrue(torch.all(logits[:, 1] > logits[:, 2]))

            # test checkpoint
            conf.num_iterations = 200
            new_trainer = MyTrainer(conf)
            new_trainer._restore_checkpoint()
            self.assertEqual(Trainer.progress(), 0.5)
            time_step = common.get_initial_time_step(env)
            state = alg.get_initial_predict_state(env.batch_size)
            policy_step = alg.rollout_step(time_step, state)
            logits = policy_step.info.logits
            self.assertTrue(torch.all(logits[:, 1] > logits[:, 0]))
            self.assertTrue(torch.all(logits[:, 1] > logits[:, 2]))

            new_trainer.train()
            self.assertEqual(Trainer.progress(), 1)

            # TODO: test play. Need real env to test.


if __name__ == "__main__":
    alf.test.main()
