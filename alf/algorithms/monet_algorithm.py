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

from typing import Callable, Tuple, List
import numpy as np

import torch
import torch.distributions as td

import alf
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.vae import VAEOutput
from alf.data_structures import namedtuple, AlgStep, LossInfo
from alf.utils import tensor_utils


class MoNetUNet(alf.networks.Network):
    """Implement the UNet architecture used by MoNet. See Appendix B.2 of the
    MoNet paper `<https://arxiv.org/abs/1901.11390>`_ for details.

    The architecture is slightly different from the one in the paper, where for
    the downsampling path, we don't downsample for the first block but always
    downsample for the other blocks. For an illustration,

    ::

                         (img) 16       16 (output)
                    (3x3 conv) |  skip  | (3x3 conv + 1x1 conv)
                               16 ----> 16
        (3x3 conv + maxpool 2) |  skip  | (3x3 conv + upsample 2)
                               8 -----> 8
        (3x3 conv + maxpool 2) |  skip  | (3x3 conv + upsample 2)
                               4 -----> 4
                                \      /
                                  MLP
    """

    def __init__(self,
                 input_tensor_spec: alf.NestedTensorSpec,
                 filters: Tuple[int],
                 nonskip_fc_layers: Tuple[int],
                 output_channels: int,
                 use_instance_norm: bool = True,
                 name: str = "MoNetUNet"):
        """
        Args:
            input_tensor_spec: spec of the input image
            filters: a tuple of output channels along the downsampling path, each for
                a conv layer. The upsampling path uses a reversed tuple.
            nonskip_fc_layers: a tuple of fc layer sizes for the bottleneck connection
                (nonskip MLP in the illustration) of the UNet. Note that there
                will be implicitly an additional FC layer after this MLP to project
                its output to a proper dimension (the output size of the downsampling
                path) before being reshaped for the upsampling path.
            output_channels: final output channels. The output features are non-activated.
            use_instance_norm: whether to insert an instance normalization layer
                after each conv/deconv layer.
        """
        super().__init__(input_tensor_spec=input_tensor_spec, name=name)

        conv_blocks = []
        channels = input_tensor_spec.shape[0]
        for i in range(len(filters)):
            block = [
                alf.layers.Conv2D(
                    channels,
                    filters[i],
                    3,
                    strides=1,
                    padding=1,
                    use_bias=not use_instance_norm,
                    activation=alf.math.identity),
                *([torch.nn.InstanceNorm2d(filters[i], affine=True)]
                  if use_instance_norm else []),
                torch.nn.ReLU()
            ]
            if i > 0:
                block.append(torch.nn.MaxPool2d(2))
            conv_blocks.append(torch.nn.Sequential(*block))

            channels = filters[i]

        self._downsampling_path = torch.nn.ModuleList(conv_blocks)
        last_skip_tensor_spec = alf.nn.Sequential(
            *self._downsampling_path,
            input_tensor_spec=input_tensor_spec).output_spec

        self._nonskip_mlp = alf.networks.EncodingNetwork(
            input_tensor_spec=alf.TensorSpec((last_skip_tensor_spec.numel, )),
            fc_layer_params=nonskip_fc_layers)

        self._reshape = torch.nn.Sequential(
            alf.layers.FC(nonskip_fc_layers[-1], last_skip_tensor_spec.numel,
                          torch.relu_),
            alf.layers.Reshape(*last_skip_tensor_spec.shape))

        self._encoding_dim = self._nonskip_mlp.output_spec.numel

        deconv_blocks = []
        filters = filters[::-1]
        for i in range(len(filters)):
            out_channels = filters[i +
                                   1] if i < len(filters) - 1 else filters[-1]
            block = [
                alf.layers.Conv2D(
                    channels * 2,
                    out_channels,
                    3,
                    strides=1,
                    padding=1,
                    use_bias=not use_instance_norm,
                    activation=alf.math.identity),
                *([torch.nn.InstanceNorm2d(out_channels, affine=True)]
                  if use_instance_norm else []),
                torch.nn.ReLU()
            ]
            if i < len(filters) - 1:
                block.append(torch.nn.UpsamplingNearest2d(scale_factor=2))
            else:
                block.append(
                    alf.layers.Conv2D(
                        out_channels,
                        output_channels,
                        1,
                        strides=1,
                        activation=alf.math.identity))
            deconv_blocks.append(torch.nn.Sequential(*block))
            channels = out_channels

        self._upsampling_path = torch.nn.ModuleList(deconv_blocks)

    @property
    def encoding_dim(self):
        """Return the output dim of the non-skip MLP.
        """
        return self._encoding_dim

    def forward(self, inputs: torch.Tensor, state=()):
        """Do a forward step of the UNet.

        Args:
            inputs: the input image of shape ``[B,C,H,W]`` where ``C`` can be
                any value.
        Returns:
            tuple:
            - output: an output image of the shape ``[B,K,H,W]``, where ``K`` is
                ``output_channels``. The output image is non-activated.
            - state: empty
        """
        return self.decode(*self.encode(inputs)), state

    def encode(self, inputs: torch.Tensor):
        """Do an encoding step of the UNet.

        Args:
            inputs: the input image of shape ``[B,C,H,W]`` where ``C`` can be
                any value.
        Returns:
            tuple:
            - output: a latent image embedding of shape ``[B,D]``.
            - encodings: the intermediate encodings in the downsampling path.
        """
        output = inputs
        outputs = []
        for l in self._downsampling_path:
            output = l(output)
            outputs.append(output)
        output = output.reshape((output.shape[0], -1))
        return self._nonskip_mlp(output)[0], outputs

    def decode(self, inputs: torch.Tensor, encodings: List[torch.Tensor]):
        """The decoding step of the UNet.

        Args:
            inputs: the latent image embedding of shape ``[B,D]``.
            encodings: the intermediate encodings in the downsampling path.
        Returns:
            torch.Tensor: an output image of the shape ``[B,K,H,W]``, where ``K`` is
                ``output_channels``. The output image is non-activated.
        """
        output = self._reshape(inputs)
        encodings = encodings[::-1]
        for i, l in enumerate(self._upsampling_path):
            output = torch.cat([output, encodings[i]], dim=1)
            output = l(output)
        return output


MoNetInfo = namedtuple(
    "MoNetInfo",
    ['kld', 'rec_loss', 'mask_rec_loss', 'full_rec', 'mask', 'z_dist'],
    default_value=())


@alf.configurable
class MoNetAlgorithm(Algorithm):
    r"""Implement the MoNet algorithm in the paper:

    `Burgess et al. 2019, MONet: Unsupervised Scene Decomposition and Representation
    <https://arxiv.org/abs/1901.11390>`_

    The algorithm can be thought of as one kind of VAEs except that it's expected
    to produce object-centric posterior latent embeddings.

    1. We follow the exact form of image reconstruction loss in the paper. For
       each pixel, the mask values are the component weights of a GMM, and the
       predicted pixel values are the means of the GMM (log of weighted probs).
       Another implementation `<https://github.com/stelzner/monet>`_ uses an
       upper bound of this loss, where the mask values are weights of the mean
       square errors between a pixel and its predicted values (weighted log probs).
    2. We also support generating attention masks all at once, which could speed
       up the attention process if the number of slots is large. However, we do
       observe that the recurrent process usually gives better performance than
       this one-time process.
    3. Each slot has a different pre-assigned fixed sigma for its Gaussian model.
       The sigmas are automatically generated. The unequal sigmas are crucial for
       breaking symmetry when generating attention masks for the slots.
    """

    def __init__(self,
                 n_slots: int,
                 slot_size: int,
                 input_tensor_spec: alf.NestedTensorSpec,
                 attention_unet_cls: Callable = MoNetUNet,
                 encoder_cls: Callable = alf.networks.EncodingNetwork,
                 decoder_cls: Callable = alf.networks.
                 SpatialBroadcastDecodingNetwork,
                 recurrent_attention: bool = True,
                 beta: float = 0.,
                 gamma: float = 0.,
                 name: str = "MoNetAlgorithm"):
        """
        Args:
            n_slots: number of slots (or objects) pre-defined. Note that background
                is also counted as an "object".
            slot_size: the dimension of each slot embedding.
            input_tensor_spec: the spec of input images
            attention_unet_cls: creates the attention UNet that generates masks for
                the slots. Depending on the value of ``recurrent_attention``, this unet
                input and output channels might change. The user doesn't need to specify
                the input and output specs for this UNet, as it is automatically handled
                by the algorithm.

                - If ``recurrent_attention==True``, this UNet receives RGB+attention_scope
                  and outputs attention logits for the current iteration. Input shape:
                  ``[B,C+1,H,W]``; output shape: ``[B,2,H,W]``.
                - Otherwise it receives RGB and outputs ``n_slots`` channels
                  (all attention logits). Input shape: ``[B,C,H,W]``; output shape:
                  ``[B,n_slots,H,W]``.

                In either case, the UNet's output should be *non-activated*.
            encoder_cls: creates the posterior encoder of MoNet. Note that this encoder
                operates on each individual slot independently, and thus it's invariant
                to the slot order. For each slot, the encoder accepts a concatenation
                of the image and an attention mask for the slot, in a shape of
                ``[B,C+1,H,W]``. The encoder outputs a *non-activated* vector of shape
                ``[B,2*slot_size]``, representing the mean and log variance of the
                slot Gaussian posterior.
            decoder_cls: creates the decoder of MoNet. The decoder also operates on
                each individual slot independently, and it should reconstruct both
                the image (the part masked by the attention; 3 channels) and the
                attention mask input to the encoder (1 channel). The output should
                be *non-activated*. Input shape: ``[B,slot_size]``; output shape:
                ``[B,C+1,H,W]``.
            recurrent_attention: if True, recurrently generates attention masks where
                each iteration conditions on the scope as the remaining attention;
                otherwise all attention masks are generated once.
            beta: weight for the VAE KLD term, sometimes this KLD can be ignored.
            gamma: weight for the KLD between generated attention masks and the
                reconstructed masks. A positive value might help make the masks more
                regular and compact.
        """

        # Notation convention in the code comments:
        # B - batch size
        # G - number of slots (n_slots)
        # D - vector size per slot (slot_size)
        # C - image channels
        # H - image height
        # W - image width

        super(MoNetAlgorithm, self).__init__(name=name)

        assert input_tensor_spec.ndim == 3, "Expect an RGB input!"
        C, H, W = input_tensor_spec.shape

        self._recurrent_attention = recurrent_attention

        # In the case of recurrent mask, the trick is to set output channels as 2,
        # because we can use ``log_softmax`` to get ``log(1-a)`` without actually
        # doing minus in the log space.
        in_channels = (C + 1 if recurrent_attention else C)
        self._attention_net = attention_unet_cls(
            input_tensor_spec=alf.TensorSpec((in_channels, H, W)),
            output_channels=(2 if recurrent_attention else n_slots))

        self._encoder = alf.networks.BatchSquashNetwork(
            encoder_cls(
                input_tensor_spec=alf.TensorSpec((C + 1, H, W)),
                last_layer_size=slot_size * 2,  # mean and var
                last_activation=alf.math.identity))

        self._decoder = alf.networks.BatchSquashNetwork(
            decoder_cls(
                input_size=slot_size,
                output_height=H,
                output_width=W,
                output_activation=alf.math.identity))
        assert self._decoder.output_spec.shape[0] == C + 1, (
            "The decoder's output channels should be RGBA")

        self._n_slots = n_slots
        self._beta = beta
        self._gamma = gamma
        # Inverse variance of slot reconstruction Gaussians in [1,1.5]
        inv_var = torch.arange(n_slots) * (0.5 / n_slots) + 1.
        self._inv_var = inv_var.reshape(1, -1, 1, 1, 1)

    @staticmethod
    def make_gaussian(z_mean_and_log_var):
        D = z_mean_and_log_var.shape[-1] // 2
        z_mean = z_mean_and_log_var[..., :D]
        z_log_var = z_mean_and_log_var[..., D:]
        # [B,G,D]
        return td.Independent(
            td.Normal(loc=z_mean, scale=z_log_var.exp()),
            reinterpreted_batch_ndims=2)

    def _compute_mask_logprobs(self, img):
        if self._recurrent_attention:
            # This could be very slow if slots are many
            # [B,1,H,W]
            scope = torch.zeros_like(img[:, :1, ...])
            mask_logits = []
            for i in range(self._n_slots - 1):
                # [B,2,H,W]
                m = self._attention_net(torch.cat([img, scope.exp()],
                                                  dim=1))[0]
                m = torch.nn.functional.log_softmax(m, dim=1)
                mask_logits.append(scope + m[:, :1, ...])
                scope = scope + m[:, 1:, ...]
            mask_logits.append(scope)
            # [B,G,H,W]
            return torch.cat(mask_logits, dim=1)
        else:
            # the mask net outputs all mask logits at one time
            # [B,G,H,W]
            mask_logits = self._attention_net(img)[0]
            return torch.nn.functional.log_softmax(mask_logits, dim=1)

    def _encoder_step(self, inputs):
        # [B,G,H,W]
        mask_logprobs = self._compute_mask_logprobs(inputs)
        # [B,G,C,H,W]
        inputs = tensor_utils.tensor_extend_new_dim(
            inputs, dim=1, n=self._n_slots)
        # Even though the MoNet paper appends the mask in the log space,
        # a linear space is actually more numerically stable.
        # [B,G,C+1,H,W]
        inputs = torch.cat([inputs, mask_logprobs.unsqueeze(2).exp()], dim=2)
        # [B,G,2*D]
        z_mean_log_var = self._encoder(inputs)[0]
        return z_mean_log_var, mask_logprobs

    def _decoder_step(self, z):
        # z - [B,G,D]
        # [B,G,C+1,H,W]
        decoded = self._decoder(z)[0]
        # First 3 channels are RGB; last is the predicted log mask (Alpha)
        return decoded[:, :, :-1, ...], decoded[:, :, -1:, ...]

    def _rec_loss_step(self, inputs, rec, mask, mask_rec):
        # inputs: [B,C,H,W]
        # rec: [B,G,C,H,W]
        # mask: [B,G,1,H,W]
        # mask_rec: [B,G,1,H,W]
        def _reduce_loss(l):
            return l.sum(list(range(1, l.ndim)))

        def _compute_rec_loss(rec, target):
            rec_log_prob = (self._inv_var.log() -
                            self._inv_var * alf.math.square(rec - target))
            return _reduce_loss(-torch.logsumexp(rec_log_prob + mask, dim=1))

        rec_loss = _compute_rec_loss(rec, inputs.unsqueeze(1))
        with torch.no_grad():
            # Compute a lower bound loss to have a more interpretable
            # loss curve (removing constant offset). This won't affect training.
            lower_bound_loss = _compute_rec_loss(rec, rec)
        rec_loss = rec_loss - lower_bound_loss

        mask_rec = torch.nn.functional.log_softmax(mask_rec, dim=1)
        mask_rec_loss = _reduce_loss(
            torch.nn.functional.kl_div(
                input=mask_rec, target=mask, reduction='none',
                log_target=True).sum(dim=1))
        return rec_loss, mask_rec_loss  # [B]

    def train_step(self, inputs: torch.Tensor, state=()):
        """Run a training step of MoNet.

        Args:
            inputs: the input image
        Returns:
            AlgStep:
            - output (VAEOutput): contains the rsampled posterior ``z`` and the
                mode of the posterior distribution ``z_mode``.
            - state: empty
            - info (MoNetInfo):
                - loss: the overall loss
                - kld: kl divergence between posterior and prior (before ``beta``)
                - rec_loss: image reconstruction loss
                - mask_rec_loss: mask reconstruction loss (before ``gamma``)
                - full_rec: the fully reconstructed image from all slots (shape
                  ``[B,C,H,W]``)
                - mask: the attention masks output by the attention network (note
                  not the reconstructed one; shape ``[B,slots,H,W]``)
                - z_dist: the posterior distribution
        """
        z_mean_log_var, mask_logprobs = self._encoder_step(inputs)
        z_dist = self.make_gaussian(z_mean_log_var)
        output = VAEOutput(
            z=z_dist.rsample(), z_mode=alf.utils.dist_utils.get_mode(z_dist))

        if self._beta == 0:
            kld = ()
        else:
            # N(0,1) as prior
            prior_z_dist = self.make_gaussian(torch.zeros_like(z_mean_log_var))
            kld = td.kl.kl_divergence(z_dist, prior_z_dist)

        rec, mask_rec = self._decoder_step(output.z)

        # [B,G,1,H,W]
        mask = mask_logprobs.unsqueeze(2)

        rec_loss, mask_rec_loss = self._rec_loss_step(inputs, rec, mask,
                                                      mask_rec)

        info = MoNetInfo(
            kld=kld,
            rec_loss=rec_loss,
            mask_rec_loss=mask_rec_loss,
            full_rec=(rec * mask.exp()).sum(dim=1),  # [B,C,H,W]
            mask=mask_logprobs.exp(),  # [B,G,H,W]
            z_dist=z_dist)

        return AlgStep(output=output, info=info)

    def calc_loss(self, info: MoNetInfo):
        loss = info.rec_loss + self._gamma * info.mask_rec_loss
        if info.kld != ():
            loss = loss + self._beta * info.kld
        return LossInfo(
            loss=loss,
            extra=MoNetInfo(
                kld=info.kld,
                rec_loss=info.rec_loss,
                mask_rec_loss=info.mask_rec_loss))
