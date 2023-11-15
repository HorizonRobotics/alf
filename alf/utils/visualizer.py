# Copyright (c) 2021 Horizon Robotics. All Rights Reserved.
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
"""Various functions related to visualizations of networks etc."""

import torch
import alf.nest as nest


def critic_network_visualizer(net,
                              observation,
                              action_upper_left,
                              action_upper_right,
                              action_lower_left,
                              H=20,
                              W=20,
                              batch_size=None):
    """Generate a batched network response image within the rectangular range
    of actions (referred to as probing region) specified by ``action_top_left``,
    ``action_top_right``, ``action_bottom_left`` as shown below:

    ``action_upper_left`` -----> ``action_upper_right``
            |                           |
            |                           |
            v                           |
    ``action_lower_left``---- ``action_lower_right``

    where ``action_lower_right`` is computed from the three provided points
    as the following because of the rectangular assumption:

    ``action_lower_right = (action_upper_right + action_lower_left - action_upper_left)``

    Example usage:

    .. code-block:: python

        # assume a case where the dimensionality of action is 4
        # the action for the upper-left point of the probing region
        action_upper_left = torch.Tensor([1, -1, 0, 0])
        # the action for the upper-right point of the probing region
        action_upper_right = torch.Tensor([1, 1, 0, 0])
        # the action for the lower-left point of the probing region
        action_lower_left = torch.Tensor([-1, -1, 0, 0])

        # define a network function
        def net_func(net_input):
            critics, _ = self._critic_networks(
                net_input)  # [B, replicas * reward_dim]
            critics = critics.reshape(  # [B, replicas, reward_dim]
                -1, self._num_critic_replicas, *self._reward_spec.shape)
            critics = critics.min(dim=1)[0]
            return critics

        img = critic_network_visualizer(net_func, inputs.observation,
                                action_upper_left, action_upper_right,
                                action_lower_left,
                                20, 20)

        # visualize the first response image in the batch
        data = img[0, ...].squeeze(0)
        data = data.cpu().numpy()

        import alf.summary.render as render
        val_img = render.render_heatmap(name="val_img", data=data)


    Args:
        net (Callable): a callable that is called as``net((obsevation, actions))``
        observation (Tensor): [B, ...]
        action_upper_left (tensor): tensor representing the upper-left point of
            the probing region, with the shape of [action_dim]
        action_upper_right (tensor): a tensor representing the upper-right point
            of the probing region, with the shape of [action_dim]
        action_lower_left (tensor): a tensor representing the lower-left point
            of the probing region, with the shape of [action_dim]
        H (int): number of samples to be used for creating visualization along
            the direction of ``action_lower_left - action_upper_left``.
        W (int): number of samples to be used for creating visualization along
            the direction of ``action_upper_right - action_upper_left``.
            The total number of samples is H * W.
        batch_size (int): the batch size of the input ``observation``. If None,
            will be inferred from the input ``observation``.

    Returns:
        The network response image of the shape [B, K, H, W], where K denotes
        the dimensionality of the network output for the non-batch dimension.
    """
    if batch_size is None:
        batch_size = nest.get_nest_size(observation, dim=0)

    # total number of anchors for probing the network
    num_anchors = H * W
    # expand observation from [B, ...] to [B * num_anchors, ...]
    ext_obs = nest.map_structure(
        lambda obs: torch.repeat_interleave(obs, num_anchors, dim=0),
        observation)

    assert action_upper_right.ndim == 1, "Only support 1D action"
    action_lower_right = action_upper_right + action_lower_left - action_upper_left

    # create the mini-image: [action_dim, 2, 2]
    # [action_upper_left,  action_upper_right]
    # [action_lower_left,  action_lower_right]
    mini_image = torch.stack(
        (torch.stack((action_upper_left, action_lower_left), dim=1),
         torch.stack((action_upper_right, action_lower_right), dim=1)),
        dim=2)

    # [action_dim, 2, 2] -> [1, action_dim, 2, 2] -> [1, action_dim, H, W]
    mini_image = torch.nn.functional.interpolate(
        mini_image.unsqueeze(0),
        size=(H, W),
        mode='bilinear',
        align_corners=True)

    # [action_dim, H, W]
    mini_image = mini_image.squeeze(0)

    # [action_dim, H, W] -> [action_dim, H * W] -> [H * W, action_dim]
    anchors = mini_image.reshape(-1, num_anchors).transpose(0, 1)

    # expand action from [H * W, action_dim] to [B * H * W, action_dim]
    other_dims = [1] * (anchors.dim() - 1)
    ext_anchors = anchors.repeat(batch_size, *other_dims)

    network_inputs = (ext_obs, ext_anchors)

    out = net(network_inputs)

    # q_values: [B * H * W, K] -> [B, H, W, K], where K denotes the
    # dimensionality of the network output (non-batch dim)
    value_image = out.reshape(batch_size, H, W, -1)
    # [B, H, W, K] -> [B, K, H, W]
    value_image = value_image.permute(0, 3, 1, 2)

    return value_image
