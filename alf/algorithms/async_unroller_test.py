# Copyright (c) 2022 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from absl import flags
import os
import tempfile
import time

import alf
from alf.utils import common
from alf.trainers.policy_trainer import TrainerConfig
from alf.algorithms.data_transformer import create_data_transformer


class AsyncUnrollerTest(alf.test.TestCase):
    def test_async_unroller(self):
        with tempfile.TemporaryDirectory() as root_dir:
            alf.pre_config({
                "TrainerConfig.unroll_length": 100,
                "TrainerConfig.async_unroll": True,
                "TrainerConfig.max_unroll_length": 100,
                "TrainerConfig.unroll_queue_size": 200,
                #"TrainerConfig.unroll_step_interval": 0.25,
                "create_environment.num_parallel_environments": 1,
            })
            conf_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', 'examples',
                'sac_cart_pole_conf.py')
            common.parse_conf_file(conf_file)
            config = TrainerConfig(root_dir=root_dir, conf_file=conf_file)
            env = alf.get_env()
            env.reset()
            data_transformer = create_data_transformer(
                config.data_transformer_ctor, env.observation_spec())
            config.data_transformer = data_transformer
            observation_spec = data_transformer.transformed_observation_spec
            algorithm = config.algorithm_ctor(
                observation_spec=observation_spec,
                action_spec=env.action_spec(),
                reward_spec=env.reward_spec(),
                env=env,
                config=config)
            algorithm.set_path('')
            t0 = time.time()
            for i in range(5):
                exp = algorithm.unroll(config.unroll_length)
                t = time.time()
                print("time: ", t - t0)
                t0 = t
                if exp is None:
                    print("length: 0")
                else:
                    print("length:", exp.step_type.shape[0])
                self.assertEqual(exp.step_type.shape[0], config.unroll_length)
            step_metrics = algorithm.get_step_metrics()
            step_metrics = dict(
                (m.name, int(m.result())) for m in step_metrics)
            t = time.time()
            algorithm._async_unroller.update_parameter(algorithm)
            print("time: ", t - t0)
            algorithm.finish_train()


if __name__ == "__main__":
    alf.test.main()
