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
    assert dimo == 23, (
        "Please specify the state of interests and the context state from the obsevation."
    )

    task_specific_ob, agent_pose, agent_vel, internal_states, action = tf.split(
        o, [3, 6, 6, 6, 2], axis=-1)

    agent_pose_1, agent_pose_2 = tf.split(agent_pose, [3, 3], axis=-1)
    joint_pose_sin_1, joint_pose_sin_2, joint_pose_cos_1, joint_pose_cos_2, joint_vel_1, joint_vel_2 = tf.split(
        internal_states, [1, 1, 1, 1, 1, 1], axis=-1)
    joint_1 = tf.concat([joint_pose_sin_1, joint_pose_cos_1, joint_vel_1],
                        axis=-1)
    joint_2 = tf.concat([joint_pose_sin_2, joint_pose_cos_2, joint_vel_2],
                        axis=-1)

    obs_achieved_goal = joint_1
    obs_excludes_goal = joint_2

    return (obs_excludes_goal, obs_achieved_goal)
