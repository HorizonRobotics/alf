# Copyright (c) 2025 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import math
import torch
import alf


class Concat(torch.nn.Module):
    """Module that concatenates a sequence of tensors along the last axis."""

    def forward(self, x):
        return torch.cat(x, dim=-1)


# Timestep embedding used in the DDPM++ and ADM architectures.
class PositionalEmbedding(torch.nn.Module):
    """Positional time embedding using deterministic sinusoidal features."""

    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        """Create a positional embedding module.

        Args:
            num_channels: Total number of output channels for the embedding.
            max_positions: Maximum number of positions used to scale the
                frequencies of the sinusoidal embedding.
            endpoint: Whether the highest frequency should reach the endpoint
                ``1 / max_positions`` exactly.
        """
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0,
                             end=self.num_channels // 2,
                             dtype=torch.float32,
                             device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions)**freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# Timestep embedding used in the NCSN++ architecture.
class FourierEmbedding(torch.nn.Module):
    """Random Fourier feature based time embedding."""

    def __init__(self, num_channels, scale=16):
        """Create a Fourier embedding module with random frequencies.

        Args:
            num_channels: Total number of output channels for the embedding.
            scale: Standard deviation used when sampling the random base
                frequencies.
        """
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * math.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class MLPNet(alf.networks.Network):
    """Multi-layer perceptron used to predict scores or velocities for SDEs."""

    def __init__(self,
                 input_spec,
                 output_spec,
                 mean_flow=False,
                 hidden_dim=256,
                 time_embedding_type='positional'):
        """Construct the MLP used for score or velocity prediction.

        Args:
            input_spec: Specification of the conditional input tensor.
            output_spec: Specification describing the generated tensor.
            mean_flow: Whether the network receives an additional mean-flow
                horizon input.
            hidden_dim: Feature dimension used throughout the hidden layers and
                embeddings.
            time_embedding_type: Chooses between ``'positional'`` and
                ``'fourier'`` time embeddings.
        """
        input_tensor_spec = (output_spec, input_spec, alf.TensorSpec(()))
        if mean_flow:
            input_tensor_spec = (output_spec, input_spec, alf.TensorSpec(
                ())) + (alf.TensorSpec(()), )
        super().__init__(input_tensor_spec)
        self._mean_flow = mean_flow
        k = 4 if mean_flow else 3
        self._model = torch.nn.Sequential(
            Concat(),
            torch.nn.Linear(k * hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, output_spec.numel),
        )
        self._time_embedding = (PositionalEmbedding(
            num_channels=hidden_dim, endpoint=True) if time_embedding_type
                                == 'positional' else FourierEmbedding(
                                    num_channels=hidden_dim))
        self._cond_embedding = torch.nn.Linear(in_features=input_spec.numel,
                                               out_features=hidden_dim,
                                               bias=False)
        self._x_embedding = torch.nn.Linear(in_features=output_spec.numel,
                                            out_features=hidden_dim,
                                            bias=False)

    def forward(self, inputs, state=()):
        x, cond, t = inputs[:3]
        h = inputs[3] if self._mean_flow else None
        embeddings = [self._x_embedding(x), self._time_embedding(t)]
        if h is not None:
            embeddings.append(self._time_embedding(h))
        embeddings.append(self._cond_embedding(cond))
        x = self._model(embeddings)
        return x, state


class DiTBlock(torch.nn.Module):
    """Transformer block with adaptive layer normalization conditioning."""

    def __init__(self, d_model, d_ff, cond_dim, num_heads):
        """Initialize the DiT block.

        Args:
            d_model: Transformer hidden size.
            d_ff: Hidden size of the feed-forward network inside the block.
            cond_dim: Dimensionality of the conditioning vector.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self._norm1 = torch.nn.LayerNorm(d_model, elementwise_affine=False)
        self._attn = torch.nn.MultiheadAttention(d_model,
                                                 num_heads=num_heads,
                                                 batch_first=True)
        self._norm2 = torch.nn.LayerNorm(d_model, elementwise_affine=False)
        self._fc1 = alf.layers.FC(d_model,
                                  d_ff,
                                  activation=torch.nn.functional.silu)
        self._fc2 = alf.layers.FC(d_ff, d_model)
        self._cond_mlp = torch.nn.Sequential(
            # torch.nn.SiLU(),
            alf.layers.FC(cond_dim, 6 * d_model, use_bias=True))

    def forward(self, inputs):
        x, cond = inputs
        scale1, shift1, gate1, scale2, shift2, gate2 = self._cond_mlp(
            cond).unsqueeze(1).chunk(6, dim=-1)
        h = self._norm1(x)
        h = torch.addcmul(shift1, h, 1 + scale1)
        attn_output, _ = self._attn(h, h, h)
        x.addcmul(attn_output, gate1)
        h = self._norm2(x)
        h = torch.addcmul(shift2, h, 1 + scale2)
        h = self._fc1(h)
        h = self._fc2(h)
        return x.addcmul(h, gate2)


class DiT(alf.networks.Network):
    """Diffusion Transformer architecture for sequence-shaped outputs."""

    def __init__(self,
                 input_spec,
                 output_spec,
                 d_model=128,
                 num_heads=4,
                 num_blocks=2,
                 time_embedding_dim=32,
                 mean_flow=False):
        """Create a DiT network tailored for ALF tensor specs.

        Args:
            input_spec: Specification of conditioning inputs.
            output_spec: Specification of the generated tensor shaped as a
                sequence.
            d_model: Transformer hidden size.
            num_heads: Number of attention heads in each block.
            num_blocks: Number of stacked transformer blocks.
            time_embedding_dim: Dimensionality of sinusoidal time embeddings.
            mean_flow: Whether the network receives the extra mean-flow
                horizon input.
        """
        input_tensor_spec = (output_spec, input_spec, alf.TensorSpec(()))
        if mean_flow:
            input_tensor_spec = (output_spec, input_spec, alf.TensorSpec(
                ())) + (alf.TensorSpec(()), )
        super().__init__(input_tensor_spec)

        self._blocks = torch.nn.ModuleList()
        assert output_spec.ndim == 2, "DiT only supports 2D output"
        self._in_proj = alf.layers.FC(output_spec.shape[1], d_model)
        length = output_spec.shape[0]
        self._pe = torch.nn.Parameter(torch.zeros(1, length, d_model))
        self._out_proj = alf.layers.FC(d_model, output_spec.shape[1])
        cond_dim = sum(spec.numel for spec in alf.nest.flatten(input_spec))
        cond_dim += time_embedding_dim * (2 if mean_flow else 1)
        self._time_embedding = PositionalEmbedding(
            num_channels=time_embedding_dim, endpoint=True)
        self._cond_mlp = torch.nn.Sequential(
            alf.layers.FC(cond_dim,
                          d_model,
                          activation=torch.nn.functional.silu),
            alf.layers.FC(d_model,
                          d_model,
                          activation=torch.nn.functional.silu),
        )
        for _ in range(num_blocks):
            self._blocks.append(
                DiTBlock(d_model, 4 * d_model, d_model, num_heads))

    def forward(self, inputs, state=()):
        x, cond, t = inputs[:3]
        assert x.ndim == 3, "DiT only supports 3D input"
        h = inputs[3] if len(inputs) == 4 else None
        embeddings = alf.nest.flatten(cond)
        embeddings.append(self._time_embedding(t))
        if h is not None:
            embeddings.append(self._time_embedding(h))
        cond = torch.cat(embeddings, dim=-1)
        cond = self._cond_mlp(cond)

        x = self._in_proj(x) + self._pe
        for block in self._blocks:
            x = block((x, cond))
        x = self._out_proj(x)
        return x, state


class AdaLnBlock(torch.nn.Module):
    """Residual block with adaptive layer normalization conditioning."""

    def __init__(self, in_dim, out_dim, hidden_dim, cond_dim):
        """Configure the adaptive layer normalization block.

        Args:
            in_dim: Size of the input feature dimension.
            out_dim: Size of the output feature dimension.
            hidden_dim: Hidden dimension for the internal MLP.
            cond_dim: Dimensionality of the conditioning vector applied to AdaLN.
        """
        super().__init__()
        self._norm = torch.nn.LayerNorm(in_dim, elementwise_affine=False)
        self._fc1 = alf.layers.FC(in_dim,
                                  hidden_dim,
                                  activation=torch.nn.functional.silu)
        self._fc2 = alf.layers.FC(hidden_dim, out_dim)
        self._ada = torch.nn.Sequential(
            # torch.nn.SiLU(),
            alf.layers.FC(cond_dim, 3 * in_dim, use_bias=True))

    def forward(self, inputs):
        x, cond = inputs
        h = self._norm(x)
        scale, shift, gate = self._ada(cond).chunk(3, dim=-1)
        h = h * (1 + scale) + shift
        h = self._fc1(h)
        h = self._fc2(h)
        return x + h * gate


class AdaNet(alf.networks.Network):
    """Fully-connected network with AdaLN blocks for diffusion modeling."""

    def __init__(self,
                 input_spec,
                 output_spec,
                 d_model=256,
                 num_blocks=2,
                 time_embedding_dim=32,
                 mean_flow=False):
        """Create an AdaNet model for diffusion-based generation.

        Args:
            input_spec: Specification of conditioning inputs.
            output_spec: Specification of the generated tensor.
            d_model: Hidden size of the AdaLN blocks.
            num_blocks: Number of stacked AdaLN residual blocks.
            time_embedding_dim: Dimensionality of sinusoidal embeddings.
            mean_flow: Whether to include the additional mean-flow horizon.
        """
        input_tensor_spec = (output_spec, input_spec, alf.TensorSpec(()))
        if mean_flow:
            input_tensor_spec = (output_spec, input_spec, alf.TensorSpec(
                ())) + (alf.TensorSpec(()), )
        super().__init__(input_tensor_spec)

        self._blocks = torch.nn.ModuleList()
        self._in_proj = alf.layers.FC(output_spec.numel, d_model)
        self._out_proj = alf.layers.FC(d_model, output_spec.numel)
        cond_dim = sum(spec.numel for spec in alf.nest.flatten(input_spec))
        cond_dim += time_embedding_dim * (2 if mean_flow else 1)
        self._time_embedding = PositionalEmbedding(
            num_channels=time_embedding_dim, endpoint=True)
        self._cond_mlp = torch.nn.Sequential(
            alf.layers.FC(cond_dim,
                          d_model,
                          activation=torch.nn.functional.silu),
            alf.layers.FC(d_model,
                          d_model,
                          activation=torch.nn.functional.silu),
        )
        for _ in range(num_blocks):
            self._blocks.append(
                AdaLnBlock(d_model, d_model, 4 * d_model, d_model))

    def forward(self, inputs, state=()):
        x, cond, t = inputs[:3]
        x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        h = inputs[3] if len(inputs) == 4 else None
        embeddings = alf.nest.flatten(cond)
        embeddings.append(self._time_embedding(t))
        if h is not None:
            embeddings.append(self._time_embedding(h))
        cond = torch.cat(embeddings, dim=-1)
        cond = self._cond_mlp(cond)

        x = self._in_proj(x)
        for block in self._blocks:
            x = block((x, cond))
        x = self._out_proj(x)
        x = x.reshape(*x_shape)
        return x, state


class SimpleMLPNet(alf.networks.Network):
    """Compact MLP baseline for score or velocity prediction."""

    def __init__(self, input_spec, output_spec, mean_flow=False):
        """Construct a simple baseline MLP network.

        Args:
            input_spec: Specification of conditioning inputs.
            output_spec: Specification of the generated tensor.
            mean_flow: Whether to include a mean-flow horizon input.
        """
        input_tensor_spec = (output_spec, input_spec, alf.TensorSpec(()))
        if mean_flow:
            input_tensor_spec = (output_spec, input_spec, alf.TensorSpec(
                ())) + (alf.TensorSpec(()), )
        super().__init__(input_tensor_spec)
        activation = torch.nn.GELU

        self._model = torch.nn.Sequential(
            Concat(),
            torch.nn.Linear(
                output_spec.numel + input_spec.numel + (2 if mean_flow else 1),
                256),
            activation(),
            torch.nn.Linear(256, 256),
            activation(),
            torch.nn.Linear(256, 256),
            activation(),
            torch.nn.Linear(256, output_spec.numel),
        )

    def forward(self, inputs, state=()):
        x, cond, t = inputs[:3]
        h = inputs[3] if len(inputs) == 4 else None
        embeddings = [x, cond, t.unsqueeze(-1)]
        if h is not None:
            embeddings.append(h.unsqueeze(-1))
        x = self._model(embeddings)
        return x, state
