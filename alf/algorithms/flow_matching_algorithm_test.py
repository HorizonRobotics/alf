# Copyright (c) 2024 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import alf
from alf.utils.datagen import load_cifar10, load_mnist
from alf.algorithms.flow_matching_algorithm import FlowMatchingAlgorithm
from alf.algorithms.monet_algorithm import MoNetUNet

import torch
from torchvision.transforms import ToPILImage
from torchvision.transforms import Resize


class _VectorFieldNetwork(alf.networks.Network):
    """A U-Net to predict a vector field from a corrupted image, time step, and
    a label (optional).

    The time step and label embedding (optional) are first broadcast to the image
    spatial domain and then concatenated along the image channels.
    """

    def __init__(self, input_tensor_spec):
        super().__init__(
            input_tensor_spec=input_tensor_spec, name="VectorFieldNetwork")
        self._img_spec = input_tensor_spec[0]
        label_spec = None
        in_channels = self._img_spec.shape[0] + 1
        if len(input_tensor_spec) == 3:
            label_spec = input_tensor_spec[-1]
        if label_spec is not None:
            self._label_embedding = torch.nn.Embedding(
                num_embeddings=label_spec.maximum + 1, embedding_dim=4)
            in_channels += 4
        self._unet_input_resize = Resize((32, 32))
        self._unet = MoNetUNet(
            input_tensor_spec=alf.TensorSpec((in_channels, 32, 32)),
            filters=(32, 64, 64, 128, 256),
            nonskip_fc_layers=(512, ),
            output_channels=self._img_spec.shape[0])
        self._unet_output_resize = Resize(self._img_spec.shape[-2:])

    def forward(self, inputs, state=()):
        _, H, W = self._img_spec.shape
        label = None
        img, tau = inputs[:2]
        if len(inputs) == 3:
            label = inputs[-1]
        tau = tau.reshape(-1, 1, 1, 1).expand(-1, -1, H, W)  # [B] -> [B,1,H,W]
        if label is not None:
            # [B] -> [B,4]
            label = self._label_embedding(label)
            label = label[..., None, None].expand(-1, -1, H,
                                                  W)  # [B,4] -> [B,4,H,W]
            inputs = torch.cat([img, tau, label], dim=1)
        else:
            inputs = torch.cat([img, tau], dim=1)
        output = self._unet(self._unet_input_resize(inputs))[0]
        output = self._unet_output_resize(output)
        return output, state


class FlowMatchingAlgorithmTest(alf.test.TestCase):
    def test_cond_gen_images(self, name="mnist"):
        """Test the conditional generation of images using flow matching.

        Args:
            name: either "mnist" or "cifar10". Note for "cifar10" the training data
                is not enough and thus the generation result usually doesn't resemble
                any training sample.
        """

        if name == "mnist":
            image_spec = alf.TensorSpec((1, 28, 28))
        else:
            image_spec = alf.TensorSpec((3, 32, 32))

        device = alf.get_default_device()
        flow_match_alg = FlowMatchingAlgorithm(
            output_tensor_spec=image_spec,
            cond_input_tensor_spec=alf.BoundedTensorSpec((),
                                                         maximum=9,
                                                         dtype=torch.int64),
            vector_field_network_ctor=_VectorFieldNetwork,
            tau_beta_paras=(1., 1.5),
            noise_std=2.,
            integration_steps=30)

        # Train
        optimizer = torch.optim.Adam(
            list(flow_match_alg.parameters()), lr=1e-2)
        if name == "mnist":
            train_loader, test_loader = load_mnist(train_bs=128)
        else:
            train_loader, test_loader = load_cifar10(train_bs=128)
        epochs = 15
        for e in range(epochs):
            epoch_losses = []
            data_iter = iter(train_loader)
            while True:
                try:
                    # No good way to let the data loader directly load CUDA tensors,
                    # so we have to manually set the device.
                    alf.set_default_device('cpu')
                    img, label = next(data_iter)
                    if device != 'cpu':
                        img, label = img.cuda(), label.cuda()
                except StopIteration:
                    break
                finally:
                    alf.set_default_device(device)
                optimizer.zero_grad()
                alg_step = flow_match_alg.train_step((img, label))
                loss = alg_step.info.loss.mean()
                loss.backward()
                epoch_losses.append(loss)
                optimizer.step()
            print(f"Epoch {e} loss: ", sum(epoch_losses) / len(epoch_losses))

        # Generate
        flow_match_alg.eval()
        samples_per_class = 2
        with torch.no_grad():
            classes = torch.tensor(list(range(10)), dtype=torch.int64)
            classes = torch.repeat_interleave(classes, samples_per_class)
            # [B,1,H,W]
            imgs = flow_match_alg.generate(
                classes, return_intermediate_steps=True)
            # take the denoising steps every two
            imgs = imgs[::2]

            image = torch.cat(imgs, dim=-1)  # [B,C,H,NW]
            image = torch.transpose(image, 0, 1)  # [C,B,H,NW]
            image = torch.flatten(image, 1, 2)  # [C,BH,NW]

            # The image has been normalized with mean and std.
            # To convert it to [0,1], we need to unnormalize it.
            if name == "mnist":
                mean = torch.tensor([0.1307])
                std = torch.tensor([0.3081])
            else:
                mean = torch.tensor([0.4914, 0.4822, 0.4465])
                std = torch.tensor([0.2023, 0.1994, 0.2010])
            image = image * std[:, None, None] + mean[:, None, None]

            image = ToPILImage()(torch.clip(image, 0, 1))
            image.save("/tmp/flow_match_generation.png")


if __name__ == "__main__":
    alf.test.main()
