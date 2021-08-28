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

from typing import List

import torch
import alf
from alf.tensor_specs import TensorSpec
from alf.utils.math_ops import identity


def _create_residual_cnn_block(
        input_tensor_spec, kernel_initializer=torch.nn.init.xavier_uniform_):
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
        kernel_initializer (Callable[[Tensor], None]): initializer for
            the Conv2D and FC layers in the module.
    
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
        kernel_initializer=torch.nn.init.xavier_uniform_):
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
        output_channels (int): specifies the number of output channels of the
            first ``Conv2D``, which is the only layer that changes the
            number of channels in the resulting module.
        num_residual_blocks (int): specifies how many residual blocks to apply
            after the ``Conv2D`` and the downsampling max pooling filter.
        kernel_initializer (Callable[[Tensor], None]): initializer for
            the Conv2D and FC layers in the module.

    Returns:
        
        A module that perofrms the described operations.

    """
    input_channels = input_tensor_spec.shape[0]
    # The output tensor spec is indentical to the input tesnor spec except for
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
           cnn_channel_list: List[int] = (16, 32, 32),
           num_blocks_per_stack: int = 2,
           output_size: int = 256,
           kernel_initializer=torch.nn.init.xavier_uniform_):
    """Create the Impala CNN Encoder

    Here the so called Impala CNN Encoder is essentially a series of CNN
    downsampling stacks with an FC layer at the end to output an 1D tensor (if
    not counting the batch dimension) of the specified output size.

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
        output_size (int): the size of the output vector for each of the input
            image in the mini batch.
        kernel_initializer (Callable[[Tensor], None]): initializer for
            the Conv2D and FC layers in the network.
    
    Returns:
        
        A module representing the Impala CNN encoding network.

    """
    stacks = []
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
    return alf.nn.Sequential(
        *stacks,
        # Flatten the images
        alf.layers.Reshape((-1, )),
        torch.nn.ReLU(inplace=True),
        alf.layers.FC(
            input_size=stack_output_spec.numel,
            output_size=output_size,
            activation=torch.relu_,
            kernel_initializer=kernel_initializer))
