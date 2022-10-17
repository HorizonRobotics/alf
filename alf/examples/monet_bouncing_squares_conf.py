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

from functools import partial

import torch

import alf
from alf.environments import suite_simple
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.algorithms.monet_algorithm import MoNetAlgorithm, MoNetInfo, MoNetUNet
from alf.algorithms.data_transformer import ImageScaleTransformer
from alf.data_structures import LossInfo, AlgStep
import alf.summary.render as render


class MoNetVisualizer(Algorithm):
    def _tensor_to_image(self, tensor, height=None, width=None):
        # Convert a [0,1] tensor to uint8 image
        img = (torch.clamp(tensor, 0., 1.) * 255).to(torch.uint8)
        if img.ndim == 3:
            img = torch.permute(img, (1, 2, 0))
        img = render.Image(img.cpu().numpy())
        if height or width:
            img.resize(height, width)
        return img

    def predict_step(self, inputs, state=()):
        monet_info, im = inputs

        imgs = {}
        if render.is_rendering_enabled():
            imgs['rec'] = render.Image.stack_images([
                render.render_text(
                    name='', data='reconstructed image', font_size=6),
                self._tensor_to_image(
                    monet_info.full_rec.squeeze(0), height=256)
            ],
                                                    horizontal=False)

            def _overlay_slot_selection(slot_sel, height):
                slot_sel = self._tensor_to_image(
                    torch.cat(
                        (slot_sel, torch.zeros((2, ) + slot_sel.shape[1:3])),
                        dim=0),
                    height=height)
                img = self._tensor_to_image(
                    im.squeeze(0).squeeze(0), height=height)
                return render.Image(img.data / 5 + slot_sel.data / 5 * 4)

            # [B,G,H,W]
            slot_selection = monet_info.mask
            selection_maps = [
                render.Image.stack_images(
                    [
                        render.render_text(
                            name='', data='slot_assign%d' % i, font_size=6),
                        render.Image.stack_images([
                            _overlay_slot_selection(
                                slot_selection[:, i, ...],  # [1,h,w]
                                height=256)
                        ])
                    ],
                    horizontal=False) for i in range(slot_selection.shape[1])
            ]
            imgs['slot_selection'] = render.Image.stack_images(
                selection_maps, horizontal=False)

        return AlgStep(info=imgs)


class MoNetAgent(RLAlgorithm):
    """A simple agent wrapper around ``MoNetAlgorithm`` to interact with the
    ``BouncingSquares`` environment. The actions are not used by the environment
    in fact.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 reward_spec=(),
                 env=None,
                 config=None,
                 optimizer=None,
                 debug_summaries=False,
                 name="MoNetAgent"):

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            train_state_spec=(),
            optimizer=optimizer,
            is_on_policy=False,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name)

        self._monet = MoNetAlgorithm(input_tensor_spec=observation_spec)
        self._vis = MoNetVisualizer()

    def train_step(self, time_step, state, rollout_info=None):
        return self._monet.train_step(time_step.observation, state)

    def predict_step(self, time_step, state):
        alg_step = self.rollout_step(time_step, state)
        vi_step = self._vis.predict_step((alg_step.info,
                                          time_step.observation))
        return alg_step._replace(info=vi_step.info)

    def rollout_step(self, time_step, state):
        alg_step = self.train_step(time_step, state)
        action = self._action_spec.sample(time_step.reward.shape[:1])
        return alg_step._replace(output=action)

    def calc_loss(self, info: MoNetInfo):
        return self._monet.calc_loss(info)


alf.config(
    "create_environment",
    env_load_fn=suite_simple.load,
    env_name="BouncingSquares",
    num_parallel_environments=20)

alf.config(
    "suite_simple.load",
    env_args=dict(N=64, noise_level=0., color=True),
    max_episode_steps=10)

obs_spec = alf.get_raw_observation_spec()
alf.config(
    "MoNetAlgorithm",
    n_slots=3,
    slot_size=8,
    attention_unet_cls=partial(
        MoNetUNet, filters=(64, ) * 5, nonskip_fc_layers=(128, ) * 2),
    encoder_cls=partial(
        alf.networks.EncodingNetwork,
        conv_layer_params=((32, 4, 2, 1), (32, 4, 2, 1), (64, 4, 2, 1),
                           (64, 4, 2, 1), (64, 4, 2, 1)),
        fc_layer_params=(256, )),
    decoder_cls=partial(
        alf.networks.SpatialBroadcastDecodingNetwork,
        conv_layer_params=((32, 3, 1), ) * 4 + (
            (obs_spec.shape[0] + 1, 3, 1), ),
        fc_layer_params=(32, )),
    recurrent_attention=True,
    beta=0.,
    gamma=0.)

alf.config(
    'TrainerConfig',
    algorithm_ctor=partial(
        MoNetAgent, optimizer=alf.optimizers.AdamTF(lr=1e-4)),
    data_transformer_ctor=[partial(ImageScaleTransformer, min=0.)],
    whole_replay_buffer_training=False,
    clear_replay_buffer=False,
    temporally_independent_train_step=True,
    summarize_first_interval=False,
    num_iterations=100000,
    mini_batch_size=256,
    mini_batch_length=1,
    num_updates_per_train_iter=1,
    unroll_length=2,
    evaluate=False,
    summary_interval=50,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    initial_collect_steps=10000,
    replay_buffer_length=10000)
