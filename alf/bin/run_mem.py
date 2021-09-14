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

import torch
import gym3
import time
from gym3 import types_np
from procgen import ProcgenGym3Env


def run_raw_gym3():
    env = ProcgenGym3Env(num=64, env_name='bossfight', render_mode='rgb_array')
    env = gym3.ExtractDictObWrapper(env, "rgb")
    step = 0
    while True:
        env.act(types_np.sample(env.ac_space, bshape=(env.num, )))
        reward, obs, first = env.observe()
        print(f"step {step} reward {reward[0]} first {first[0]}")
        step += 1
        time.sleep(1)


if __name__ == '__main__':
    run_raw_gym3()
