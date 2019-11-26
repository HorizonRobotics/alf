# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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

import gin
import tensorflow as tf


@gin.configurable
def split_observation_fn(o):

    dimo = o.get_shape().as_list()[-1]
    assert dimo == 23, ("The dimension does not match.")

    task_specific_ob, agent_pose, agent_vel, internal_states, action = tf.split(
        o, [3, 6, 6, 6, 2], axis=-1)

    agent_pose_1, agent_pose_2 = tf.split(agent_pose, [3, 3], axis=-1)

    return (agent_pose_1, task_specific_ob)
