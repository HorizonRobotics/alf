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
"""Contrastive Unsupervised Representations for Reinforcement Learning."""

import torch
import torch.nn as nn
import numpy as np

import alf
from alf.algorithms.algorithm import Algorithm
from alf.data_structures import AlgStep, LossInfo, TimeStep
from alf.utils import common
from skimage.util.shape import view_as_windows
from skimage.io import imsave
import torchvision


def creat_encoder(input_spec, feature_dim, num_layers=2, num_filters=32):
    """
    Creats encoder for CURL Alogrithm.

    Args:

        input_spec (TensorSpec): Describing the input tensor.
        feature_dim (int): The dimension of feature at the output.
        num_layers (int): Number of hidden layers.
        num_filters (int): Number of filters in each layer

    Returns:
        
        (Alf Sequential): A module that perofrms the described operations.

    """

    stacks = [
        alf.layers.Conv2D(input_spec.shape[0], num_filters, 3, strides=2)
    ]

    for i in range(num_layers - 1):
        stacks.append(
            alf.layers.Conv2D(num_filters, num_filters, 3, strides=1))

    before_fc = alf.nn.Sequential(
        *stacks, alf.layers.Reshape((-1, )), input_tensor_spec=input_spec)
    return alf.nn.Sequential(
        before_fc,
        alf.layers.FC(
            input_size=before_fc.output_spec.shape[0],
            output_size=feature_dim,
            use_ln=True))


@alf.configurable
class curl_encoder(Algorithm):
    """
    The encoder part of contrastive unsupervised representations
    for reinforcement learning. Can be used with most reinforcement
    learning like SAC.
    """

    def __init__(self,
                 observation_spec,
                 feature_dim,
                 crop_size=84,
                 action_spec=None,
                 encoder_tau=0.05,
                 debug_summaries=False,
                 optimizer=None,
                 output_tanh=False,
                 save_image=False,
                 use_pytorch_randcrop=False,
                 detach_encoder=False,
                 name='curl_encoder'):
        """
        Args:

            observation_spec (TensorSpec): The shape of input tensor 
                (B x C x W x H) assume W = H.
            feature_dim (int): The dimension of the output vector, the 
                dim of W is (feature_dim x feature_dim).
            crop_size (int): Dim of cropped image. After crop, the image 
                look like (B x C x crop_size x crop_size).
            encoder_tau (float): Factor for soft update of target encoder
            output_tanh (boolean): Determin if attach a layer of tanh at 
                the end of encoder. 

        Retrun:
            A CURL model.
        """
        super().__init__(
            train_state_spec=observation_spec,
            optimizer=optimizer,
            debug_summaries=debug_summaries,
            name=name)
        self.observation_spec = observation_spec
        self.channels = observation_spec.shape[0]
        self.after_crop_spec = alf.BoundedTensorSpec((self.channels, crop_size,
                                                      crop_size))
        self.feature_dim = feature_dim
        self.output_spec = alf.BoundedTensorSpec((feature_dim, ))
        self.crop_size = crop_size
        self._encoding_net = creat_encoder(self.after_crop_spec, feature_dim)
        self._target_encoding_net = self._encoding_net.copy(
            name='target_encoding_net_ctor')
        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none')
        self.W = nn.Parameter(torch.rand(feature_dim, feature_dim))
        self.output_tanh = output_tanh
        self.save_image = save_image
        self.use_pytorch_randcrop = use_pytorch_randcrop
        self.detach_encoder = detach_encoder
        self._update_target = common.get_target_updater(
            models=[self._encoding_net],
            target_models=[self._target_encoding_net],
            tau=encoder_tau)
        if use_pytorch_randcrop:
            self.pytorch_randcrop = torchvision.transforms.RandomCrop(
                self.crop_size)

    def random_crop(self, obs, output_size, save_image=False):
        """
        Random crop the input images. On each image, the crop position is
        identical across the channels.

        Args:
            obs (Tensor): Batch images with shape (B,C,H,W).
            output_size (int): The hight and width of output image.
            save_image (boolean): Save the origin image and cropped image if True.

        Return:
            (Tensor): Cropped images.
        """
        if self.use_pytorch_randcrop:
            return self.pytorch_randcrop(obs)
        else:
            obs_cpu = obs.cpu()
            imgs = obs_cpu.numpy()
            n = imgs.shape[0]
            img_size = imgs.shape[-1]
            crop_max = img_size - output_size
            imgs = np.transpose(imgs, (0, 2, 3, 1))
            w1 = np.random.randint(0, crop_max, n)
            h1 = np.random.randint(0, crop_max, n)
            windows = view_as_windows(
                imgs, (1, output_size, output_size, 1))[..., 0, :, :, 0]
            cropped_imgs = windows[np.arange(n), w1, h1]
            if save_image:
                for i in range(n):
                    breakpoint()
                    imsave("~/image_test/origin" + str(i) + ".PNG",
                           imgs[i, :, :, 0])
                    imsave("~/image_test/cropped" + str(i) + ".PNG",
                           cropped_imgs[i, 0, :, :])

            return_torch = torch.from_numpy(cropped_imgs)
            return return_torch.to(torch.device("cuda:0"))

    def predict_step(self, inputs: TimeStep, state):
        #random crop
        crop_obs = self.random_crop(
            inputs.observation, self.crop_size, save_image=self.save_image)
        latent = self._encoding_net(crop_obs)[0]
        if self.output_tanh:
            output = torch.tanh(latent)
        else:
            output = latent

        return AlgStep(output=output, state=state)

    def rollout_step(self, inputs: TimeStep, state):
        #random crop
        crop_obs = self.random_crop(
            inputs.observation, self.crop_size, save_image=self.save_image)
        latent = self._encoding_net(crop_obs)[0]
        if self.output_tanh:
            output = torch.tanh(latent)
        else:
            output = latent

        return AlgStep(output=latent, state=state)

    def train_step(self, inputs: TimeStep, state, rollout_info=None):
        #random crop obs
        rc_obs_1 = self.random_crop(
            inputs.observation, self.crop_size, save_image=self.save_image)
        rc_obs_2 = self.random_crop(
            inputs.observation, self.crop_size, save_image=self.save_image)

        #generate encoded observation
        latent_q = self._encoding_net(rc_obs_1)[0]
        latent_k = self._target_encoding_net(rc_obs_2)[0].detach()

        W_z = torch.matmul(self.W, latent_k.T)
        logits = torch.matmul(latent_q, W_z)
        logits = logits - torch.max(logits, 1)[0][:, None]
        labels = torch.arange(logits.shape[0]).long()
        loss = self.CrossEntropyLoss(logits, labels)
        if self.detach_encoder:
            latent_q = latent_q.detach()
        return AlgStep(output=latent_q, state=state, info=LossInfo(loss=loss))

    def after_update(self, root_inputs=None, train_info=None):
        self._update_target()

    def _trainable_attributes_to_ignore(self):
        return ['_target_encoding_net']
