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

from functools import partial

import torch

import alf
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.examples import ac_cart_pole_conf

import alf.summary.render as render


class ACRenderAlgorithm(ActorCriticAlgorithm):
    def predict_step(self, inputs, state):
        alg_step = super().predict_step(inputs, state)
        action = alg_step.output
        action_dist = alg_step.info.action_distribution
        with alf.summary.scope("ACRender"):
            # Render an action image of type ``render.Image``.
            action_img = render.render_action(
                name="predicted_action",
                action=action,
                action_spec=self._action_spec)
            # Render an action distribution image of type ``render.Image``.
            action_dist_img = render.render_action_distribution(
                name="predicted_action_distribution",
                act_dist=action_dist,
                action_spec=self._action_spec)
        # Put the two ``Image`` objects into ``info``. Any nest structure is
        # acceptable for the new ``info``. ALF's play will look for ``Image``
        # objects.
        return alg_step._replace(
            info=dict(
                action_img=action_img,
                action_dist_img=action_dist_img,
                ac=alg_step.info))


# configure which RL algorithm to use
alf.config(
    "TrainerConfig",
    algorithm_ctor=partial(
        ACRenderAlgorithm, optimizer=alf.optimizers.Adam(lr=1e-3)))
