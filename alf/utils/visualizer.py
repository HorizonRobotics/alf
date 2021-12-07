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


def critic_network_visualizer(net, observation, origin, dx, dy, H=20, W=20):
    """Generate a batched network response image within the range specified by
    ``origin``, ``dx``, ``dy`` as shown below:

    ``origin`` ---> ``dx``
        |
        v
      ``dy``

    Example usage:

    .. code-block:: python

        # assume a case where the dimensionality of action is 4
        # the upper-left point
        origin = torch.Tensor([1, -1, 0, 0])
        # the upper right point
        dx = torch.Tensor([1, 1, 0, 0])
        # the lower-left point
        dy = torch.Tensor([-1, -1, 0, 0])

        # define a network function
        def net_func(net_input):
            critics, _ = self._critic_networks(
                net_input)  # [B, replicas * reward_dim]
            critics = critics.reshape(  # [B, replicas, reward_dim]
                -1, self._num_critic_replicas, *self._reward_spec.shape)
            critics = critics.min(dim=1)[0]
            return critics

        img = critic_network_visualizer(net_func, inputs.observation,
                                origin, dx, dy,
                                20, 20)

        # visualize the first response image in the batch
        data = img[0, ...].squeeze(0)
        data = data.cpu().numpy()

        val_img = alf.summary.render.render_heatmap(name="val_img", data=data)


    Args:
        net (Callable): a callable that is called as``net((obsevation, actions))``
        observation (Tensor): [B, ...]
        origin (tensor): tensor representing the start of the probing region,
            with the shape of [d]
        dx (tensor): a tensor representing the end position of the probing
            region along ``dx`` direction, with the shape of [d]
        dy (tensor): a tensor representing the end position of the probing
            region along ``dy`` direction, with the shape of [d]
        H (int): number of samples to be used for creating
            visualization along ``dx`` direction.
        W (int): number of samples to be used for creating
            visualization along ``dx`` direction. The
            total number of samples is H * W.

    Returns:
        The network response image of the shape [B, K, H, W], where K denotes
        the dimensionality of the network output for the non-batch dimension.
    """

    batch_size = observation.shape[0]
    # total number of anchors for probing the network
    num_anchors = H * W
    # expand observation from [B, ...] to [B * num_anchors, ...]
    ext_obs = torch.repeat_interleave(observation, num_anchors, dim=0)
    dxy = dx + dy - origin

    # create the mini-image: [B, d, 2, 2], where d denotes the action dimensions
    # [origin, dx]
    # [dy,     dxy]
    mini_image = torch.stack((torch.stack(
        (origin, dy), dim=1), torch.stack((dx, dxy), dim=1)),
                             dim=2)

    # [B, d, 2, 2] -> [B, d, H, W]
    mini_image = torch.nn.functional.interpolate(
        mini_image.unsqueeze(0),
        size=(H, W),
        mode='bilinear',
        align_corners=True)

    # [d, H, W]
    mini_image = mini_image.squeeze(0)

    # [d, H, W] -> [d, H * W] -> [H * W, d]
    anchors = mini_image.reshape(-1, num_anchors).transpose(0, 1)

    # expand action from [H * W, d] to [B * H * W, d]
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
