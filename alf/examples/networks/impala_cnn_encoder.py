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

from typing import List, Callable, Optional

import torch
from torch import Tensor
from torch.nn.init import xavier_uniform_

import alf
from alf.tensor_specs import TensorSpec
from alf.utils.math_ops import identity


def _create_residual_cnn_block(
        input_tensor_spec,
        kernel_initializer: Callable[[Tensor], None] = xavier_uniform_):
    """Create a CNN block with 2 Conv2D plus a residual connection

    It essentially performs

    .. math::

        f(x) = x + \textrm{Conv2D}(\textrm{Conv2D}(\textrm{ReLU}(x)))

    where the inner ``Conv2D`` does have an activation while the outer
    ``Conv2D`` does not. Both ``Conv2D`` here does not change the number of
    channels (i.e. the dimension of :math:`x` remains the same).

    Args:

        input_tensor_spec (nested TesnorSpec): The spec of the input tensor,
            denotated as ``x`` in the above formula. The shape of the input
            tensor omitting batch size should be (C, W, H).
        kernel_initializer: initializer for the Conv2D and FC layers in the
            module.

    Returns:

        A module that perofrms the described operations.

    """

    input_channels = input_tensor_spec.shape[0]

    return alf.nn.Sequential(
        # Explicitly set inplace to False because the input will need
        # to be used for the residual skip connection.
        torch.nn.ReLU(inplace=False),
        # TODO(breakds): Normalized initialization in openai's
        # original implementation
        alf.layers.Conv2D(
            input_channels, input_channels, kernel_size=3, padding=1),
        residual=alf.layers.Conv2D(
            input_channels,
            input_channels,
            kernel_size=3,
            padding=1,
            activation=identity,
            kernel_initializer=kernel_initializer),
        final=(('input', 'residual'), lambda x: x[0] + x[1]),
        input_tensor_spec=input_tensor_spec)


def _create_downsampling_cnn_stack(
        input_tensor_spec,
        output_channels: int,
        num_residual_blocks: int = 2,
        kernel_initializer: Callable[[Tensor], None] = xavier_uniform_):
    """Create a stack of Convolutional layers that also does downsampling

    It essentially performs one ``Conv2D`` that changes the number of channels,
    a max pooling filter that downsamples the image by half, followed by a
    series of residual CNN blocks. The number of CNN blocks in the series is
    specified by ``num_residual_blocks``.

    Args:

        input_tensor_spec (nested TensorSpec): The spec of the input tensor,
            denotated as ``x`` in the above formula. The shape of the input
            tensor omitting batch size should be (C, W, H), where C is the input
            number of channels.
        output_channels: specifies the number of output channels of the
            first ``Conv2D``, which is the only layer that changes the
            number of channels in the resulting module.
        num_residual_blocks: specifies how many residual blocks to apply
            after the ``Conv2D`` and the downsampling max pooling filter.
        kernel_initializer: initializer for the Conv2D and FC layers in the
            module.

    Returns:

        A module that perofrms the described operations.

    """
    input_channels = input_tensor_spec.shape[0]
    # The output tensor spec is identical to the input tesnor spec except for
    # that the channel dimension is replaced by the output number of channels.
    output_tensor_spec = TensorSpec(
        (output_channels, *input_tensor_spec.shape[1:]),
        input_tensor_spec.dtype)
    return alf.nn.Sequential(
        # NOTE: this Conv2D layer does not have activation
        alf.layers.Conv2D(
            input_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            activation=identity,
            kernel_initializer=kernel_initializer),
        # Downsample via a max pooling filter
        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        *[
            _create_residual_cnn_block(
                output_tensor_spec, kernel_initializer=kernel_initializer)
            for _ in range(num_residual_blocks)
        ],
        input_tensor_spec=input_tensor_spec)


def create(input_tensor_spec,
           cnn_channel_list: List[int] = [16, 32, 32],
           num_blocks_per_stack: int = 2,
           flatten_output_size: Optional[int] = 256,
           output_activation: Callable[[Tensor], Tensor] = torch.relu_,
           kernel_initializer: Callable[[Tensor], None] = xavier_uniform_):
    """Create the Impala CNN Encoder

    Here the so called Impala CNN Encoder is essentially a series of CNN
    downsampling stacks.

    When explicitly specified, add an FC layer at the end to flatten the output
    to an 1D tensor (if not counting the batch dimension) of the specified
    output size.

    Note that the default values for the parameters are the ones that are used
    for the procgen environments. Credits to the OpenAI's official PPG
    implementation.

    If your use case is different, it is very likely that you will need to find
    your own parameters.

    Args:

        input_tensor_spec (nested TensorSpec): The spec of the input tensor,
            denotated as ``x`` in the above formula. The shape of the input
            tensor omitting batch size should be (C, W, H), where C is the input
            number of channels.
        cnn_channel_list (List[int]): a list that specifies the number of output
            channels of each CNN downsampling stack in the application order.
        num_blocks_per_stack (int): specifies how many residual blocks to apply
            in each of the CNN downsampling stack. This parameter is uniform to
            all the CNN downsampling stacks.
        flatten_output_size: When not None, the 2D output tensors will be
            flattened into 1D vectors of this size at the end via FC.
        output_activation: the activation applied to the output vector/tensor.
        kernel_initializer: initializer for the Conv2D and FC layers in the
            network.

    Returns:

        A module representing the Impala CNN encoding network.

    """
    stacks = []
    # The CNN encoder expects float32 pixels between [0.0, 1.0]. If
    # the dtype of the input is uint8, we assume that the pixels are
    # within range [0, 255]. In this case we will apply "image scaling
    # transformation" logic to convert it to float32 [0.0, 1.0].
    if input_tensor_spec.dtype is torch.uint8:
        scale_factor = 1.0 / 255.0
        stacks.append(alf.layers.Cast(dtype=torch.float32))
        stacks.append(lambda x: x * scale_factor)
        last_output_spec = alf.BoundedTensorSpec(
            input_tensor_spec.shape,
            dtype=torch.float32,
            minimum=0.0,
            maximum=1.0)
    else:
        last_output_spec = input_tensor_spec

    for output_channels in cnn_channel_list:
        stacks.append(
            _create_downsampling_cnn_stack(
                last_output_spec,
                output_channels,
                num_blocks_per_stack,
                kernel_initializer=kernel_initializer))
        last_output_spec = stacks[-1].output_spec
    stack_output_spec = stacks[-1].output_spec

    if flatten_output_size is not None:
        return alf.nn.Sequential(
            *stacks,
            # Flatten the images
            alf.layers.Reshape((-1, )),
            torch.nn.ReLU(inplace=True),
            alf.layers.FC(
                input_size=stack_output_spec.numel,
                output_size=flatten_output_size,
                activation=output_activation,
                kernel_initializer=kernel_initializer),
            input_tensor_spec=input_tensor_spec)
    else:
        return alf.nn.Sequential(
            *stacks, output_activation, input_tensor_spec=input_tensor_spec)
