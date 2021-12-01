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

import functools
import tempfile
import torch

import alf
from alf.algorithms.hypernetwork_algorithm import HyperNetwork
from alf.algorithms.rl_algorithm_test import MyEnv, MyAlg
from alf.trainers.policy_trainer import RLTrainer, TrainerConfig, play
from alf.trainers.policy_trainer import SLTrainer
from alf.utils import common, datagen


def env_load(env_name, batch_size):
    return MyEnv(3)


env_load.batched = True


class TrainerTest(alf.test.TestCase):
    def test_rl_trainer(self):
        with tempfile.TemporaryDirectory() as root_dir:
            alf.config("create_environment", env_load_fn=env_load)
            conf = TrainerConfig(
                algorithm_ctor=MyAlg,
                root_dir=root_dir,
                unroll_length=5,
                num_iterations=100)

            # test train
            trainer = RLTrainer(conf)
            self.assertEqual(RLTrainer.progress(), 0)
            trainer.train()
            self.assertEqual(RLTrainer.progress(), 1)

            alg = trainer._algorithm
            env = common.get_env()
            time_step = common.get_initial_time_step(env)
            state = alg.get_initial_predict_state(env.batch_size)
            policy_step = alg.rollout_step(time_step, state)
            logits = policy_step.info['dist'].logits
            print("logits: ", logits)
            self.assertTrue(torch.all(logits[:, 1] > logits[:, 0]))
            self.assertTrue(torch.all(logits[:, 1] > logits[:, 2]))

            # test checkpoint
            conf.num_iterations = 200
            new_trainer = RLTrainer(conf)
            new_trainer._restore_checkpoint()
            self.assertEqual(RLTrainer.progress(), 0.5)
            time_step = common.get_initial_time_step(env)
            state = alg.get_initial_predict_state(env.batch_size)
            policy_step = alg.rollout_step(time_step, state)
            logits = policy_step.info['dist'].logits
            self.assertTrue(torch.all(logits[:, 1] > logits[:, 0]))
            self.assertTrue(torch.all(logits[:, 1] > logits[:, 2]))

            new_trainer.train()
            self.assertEqual(RLTrainer.progress(), 1)

            # TODO: test play. Need real env to test.

    def test_sl_trainer(self):
        with tempfile.TemporaryDirectory() as root_dir:
            conf = TrainerConfig(
                algorithm_ctor=functools.partial(
                    HyperNetwork,
                    data_creator=datagen.load_test,
                    hidden_layers=None,
                    loss_type='regression',
                    num_train_classes=1,
                    optimizer=alf.optimizers.Adam(lr=1e-4, weight_decay=1e-4)),
                root_dir=root_dir,
                num_checkpoints=1,
                evaluate=True,
                eval_interval=1,
                num_iterations=1)

            # test train
            trainer = SLTrainer(conf)
            self.assertEqual(SLTrainer.progress(), 0)
            trainer.train()
            self.assertEqual(SLTrainer.progress(), 1)

            # test checkpoint
            conf2 = TrainerConfig(
                algorithm_ctor=functools.partial(
                    HyperNetwork,
                    data_creator=datagen.load_test,
                    hidden_layers=None,
                    loss_type='regression',
                    num_train_classes=1,
                    optimizer=alf.optimizers.Adam(lr=1e-4, weight_decay=1e-4)),
                root_dir=root_dir,
                num_checkpoints=1,
                evaluate=True,
                eval_interval=1,
                num_iterations=2)

            new_trainer = SLTrainer(conf2)
            new_trainer._restore_checkpoint()
            self.assertEqual(SLTrainer.progress(), 0.5)
            new_trainer.train()
            self.assertEqual(SLTrainer.progress(), 1)


if __name__ == "__main__":
    alf.test.main()
