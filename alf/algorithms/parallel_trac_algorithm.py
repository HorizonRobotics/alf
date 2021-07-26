# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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
"""Trusted Region Actor critic algorithm."""
import numpy as np
import torch
import torch.distributions as td

import alf
from alf.algorithms.trac_algorithm import TracAlgorithm
from alf.algorithms.parallel_actor_critic_algorithm import ParallelActorCriticAlgorithm
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import Experience, namedtuple, StepType, TimeStep
from alf.optimizers.trusted_updater import TrustedUpdater
from alf.utils import common, dist_utils, math_ops
from alf.tensor_specs import TensorSpec

nest_map = alf.nest.map_structure

TracExperience = namedtuple(
    "TracExperience",
    ["observation", "step_type", "state", "action_param", "prev_action"])

TracInfo = namedtuple(
    "TracInfo",
    ["action_distribution", "observation", "state", "ac", "prev_action"])


@alf.configurable
class ParallelTracAlgorithm(TracAlgorithm):
    """Trust-region actor-critic.

    It compares the action distributions after the SGD with the action
    distributions from the previous model. If the average distance is too big,
    the new parameters are shrinked as:

    .. code-block:: python

        w_new' = old_w + 0.9 * distance_clip / distance * (w_new - w_old)

    If the distribution is ``Categorical``, the distance is
    :math:`||logits_1 - logits_2||^2`, and if the distribution is
    ``Deterministic``, it is :math:`||loc_1 - loc_2||^2`,  otherwise it's
    :math:`KL(d1||d2) + KL(d2||d1)`.

    The reason of using :math:`||logits_1 - logits_2||^2` for categorical
    distributions is that KL can be small even if there are large differences in
    logits when the entropy is small. This means that KL cannot fully capture
    how much the change is.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 reward_spec=TensorSpec(()),
                 env=None,
                 config=None,
                 ac_algorithm_cls=ParallelActorCriticAlgorithm,
                 action_dist_clip_per_dim=0.01,
                 debug_summaries=False,
                 name="ParallelTracAlgorithm"):
        """

        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            ac_algorithm_cls (type): Actor Critic Algorithm cls.
            action_dist_clip_per_dim (float): action dist clip per dimension
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.
        """

        super().__init__(
            observation_spec = observation_spec,
            action_spec = action_spec,
            reward_spec = reward_spec,
            env = env,
            config = config,
            ac_algorithm_cls= ac_algorithm_cls,
            action_dist_clip_per_dim = action_dist_clip_per_dim,
            debug_summaries = debug_summaries,
            name = name)
        self._num_parallel_agents = config.num_parallel_agents

    def calc_loss(self, info: TracInfo):
        if self._num_parallel_agents > 1:
            for i in range(self._config.num_parallel_agents):
                if i == 0:
                    parameters_list = list(self._ac_algorithm._actor_network[i].parameters())
                else:
                    parameters_list.extend(list(self._ac_algorithm._actor_network[i].parameters())) 
        else:
            parameters_list = list(self._ac_algorithm._actor_network.parameters())

        if self._trusted_updater is None:
            self._trusted_updater = TrustedUpdater(parameters_list)
        ac_info = info.ac._replace(
            action_distribution=info.action_distribution)
        return self._ac_algorithm.calc_loss(ac_info)