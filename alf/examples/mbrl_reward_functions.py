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

import gin
import torch

from alf.utils import common
# implement the respective reward functions for desired environments here


@gin.configurable
def reward_function_for_pendulum(obs, action):
    """Function for computing reward for gym Pendulum environment. It takes
        as input:
        (1) observation (Tensor of shape [batch_size, observation_dim])
        (2) action (Tensor of shape [batch_size, num_actions])
        and returns a reward Tensor of shape [batch_size].
    """

    def _observation_cost(obs):
        c_theta, s_theta, d_theta = obs[..., :1], obs[..., 1:2], obs[..., 2:3]
        theta = torch.atan2(s_theta, c_theta)
        cost = theta**2 + 0.1 * d_theta**2
        cost = torch.sum(cost, dim=1)
        cost = torch.where(
            torch.isnan(cost), 1e6 * torch.ones_like(cost), cost)
        return cost

    def _action_cost(action):
        return 0.001 * torch.sum(action**2, dim=-1)

    cost = _observation_cost(obs) + _action_cost(action)
    # negative cost as reward
    reward = -cost
    return reward


@gin.configurable
def reward_function_for_cartpole(obs, action):
    """Function for computing reward for gym CartPole environment. It takes
        as input:
        (1) observation (Tensor of shape [batch_size, observation_dim])
        (2) action (Tensor of shape [batch_size, num_actions])
        and returns a reward Tensor of shape [batch_size].
    """

    def _observation_cost(obs):
        x0, theta = obs[..., :1], obs[..., 1:2]
        ee_pos = torch.cat(
            (x0 - 0.6 * torch.sin(theta), -0.6 * torch.cos(theta)), dim=-1)
        cost = (ee_pos - torch.as_tensor([.0, .6]))**2
        cost = -torch.exp(torch.sum(cost, dim=-1) / (0.6**2))
        cost = torch.where(
            torch.isnan(cost), 1e6 * torch.ones_like(cost), cost)

        return cost

    def _action_cost(action):
        cost = 0.01 * torch.sum(action**2, dim=-1)
        return cost

    cost = _observation_cost(obs) + _action_cost(action)
    reward = -cost
    return reward


@gin.configurable
def reward_function_for_halfcheetah(obs, action):
    """Function for computing reward for gym CartPole environment. It takes
        as input:
        (1) observation (Tensor of shape [batch_size, observation_dim])
        (2) action (Tensor of shape [batch_size, num_actions])
        and returns a reward Tensor of shape [batch_size].
    """

    def _observation_cost(obs):
        cost = -obs[..., 0]
        return cost

    def _action_cost(action):
        cost = 0.1 * torch.sum(action**2, dim=-1)
        return cost

    cost = _observation_cost(obs) + _action_cost(action)
    reward = -cost
    return reward


@gin.configurable
def reward_function_for_pusher(obs, action):
    """Function for computing reward for gym CartPole environment. It takes
        as input:
        (1) observation (Tensor of shape [batch_size, observation_dim])
        (3) action (Tensor of shape [batch_size, num_actions])
        and returns a reward Tensor of shape [batch_size].
    """

    def _observation_cost(obs):
        to_w, og_w = 0.5, 1.25
        tip_pos, obj_pos = obs[..., 14:17], obs[..., 17:20]
        tip_obj_dist = torch.sum(torch.abs(tip_pos - obj_pos), dim=-1)
        obj_goal_dist = torch.sum(
            torch.abs(common.get_gym_env_attr('ac_goal_pos') - obj_pos),
            dim=-1)
        cost = to_w * tip_obj_dist + og_w * obj_goal_dist
        cost = torch.where(
            torch.isnan(cost), 1e6 * torch.ones_like(cost), cost)

        return cost

    def _action_cost(action):
        cost = 0.1 * torch.sum(action**2, dim=-1)
        return cost

    cost = _observation_cost(obs) + _action_cost(action)
    reward = -cost
    return reward


@gin.configurable
def reward_function_for_reacher(obs, action):
    """Function for computing reward for gym CartPole environment. It takes
        as input:
        (1) observation (Tensor of shape [batch_size, observation_dim])
        (2) action (Tensor of shape [batch_size, num_actions])
        and returns a reward Tensor of shape [batch_size].
    """

    def _observation_cost(obs):
        theta1, theta2, theta3, theta4, theta5, theta6, theta7 = \
            obs[..., :1], obs[..., 1:2], obs[..., 2:3], obs[..., 3:4], \
            obs[..., 4:5], obs[..., 5:6], obs[..., 6:]
        rot_axis = torch.cat(
            (torch.cos(theta2) * torch.cos(theta1),
             torch.cos(theta2) * torch.sin(theta1), -torch.sin(theta2)),
            dim=-1)
        rot_perp_axis = torch.cat(
            (-torch.sin(theta1), torch.cos(theta1), torch.zeros_like(theta1)),
            dim=-1)
        cur_end = torch.cat((
            0.1 * torch.cos(theta1) + 0.4 * torch.cos(theta1) * torch.cos(theta2),
            0.1 * torch.sin(theta1) + 0.4 * torch.sin(theta1) * torch.cos(theta2) \
                - 0.188,
            -0.4 * torch.sin(theta2)), dim=-1)

        for length, hinge, roll in [(0.321, theta4, theta3), \
                (0.16828, theta6, theta5)]:
            perp_all_axis = torch.cross(rot_axis, rot_perp_axis)
            x = torch.cos(hinge) * rot_axis
            y = torch.sin(hinge) * torch.sin(roll) * rot_perp_axis
            z = -torch.sin(hinge) * torch.cos(roll) * perp_all_axis
            new_rot_axis = x + y + z
            new_rot_perp_axis = torch.cross(new_rot_axis, rot_axis)
            tmp_rot_perp_axis = torch.where(
                torch.lt(torch.norm(new_rot_perp_axis, dim=-1), 1e-30),
                rot_perp_axis.permute(-1,
                                      *list(range(rot_perp_axis.ndim - 1))),
                new_rot_perp_axis.permute(
                    -1, *list(range(new_rot_perp_axis.ndim - 1))))
            new_rot_perp_axis = tmp_rot_perp_axis.permute(
                *list(range(1, tmp_rot_perp_axis.ndim)), 0)
            new_rot_perp_axis /= torch.norm(
                new_rot_perp_axis, dim=-1, keepdim=True)
            rot_axis, rot_perp_axis, cur_end = \
                new_rot_axis, new_rot_perp_axis, cur_end + length * new_rot_axis

        cost = torch.sum(
            torch.square(cur_end - common.get_gym_env_attr('goal')), dim=-1)
        return cost

    def _action_cost(action):
        cost = 0.01 * torch.sum(action**2, dim=-1)
        return cost

    cost = _observation_cost(obs) + _action_cost(action)
    reward = -cost
    return reward
