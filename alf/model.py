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

from tf_agents.environments.time_step import TimeStep
from tf_agents.networks.network import Network


class Model(Network):
    def __init__(self,
                 time_step_spec: TimeStep,
                 action_spec,
                 policy_state_spec,
                 info_spec,
                 name="Model"):

        self._time_step_spec = time_step_spec
        self._action_spec = action_spec
        self._policy_state_spec = policy_state_spec
        self._info_spec = info_spec

        super(Model, self).__init__(time_step_spec.observation,
                                    policy_state_spec, name)

    @property
    def time_step_spec(self):
        """Describes the `TimeStep` tensors returned by `step()`.

        Returns:
        A `TimeStep` namedtuple with `TensorSpec` objects instead of Tensors,
        which describe the shape, dtype and name of each tensor returned by
        `step()`.
        """
        return self._time_step_spec

    @property
    def action_spec(self):
        """Describes the TensorSpecs of the Tensors expected by `step(action)`.

        `action` can be a single Tensor, or a nested dict, list or tuple of
        Tensors.

        Returns:
        An single BoundedTensorSpec, or a nested dict, list or tuple of
        `BoundedTensorSpec` objects, which describe the shape and
        dtype of each Tensor expected by `step()`.
        """
        return self._action_spec

    @property
    def policy_state_spec(self):
        """Describes the Tensors expected by `step(_, policy_state)`.

        `policy_state` can be an empty tuple, a single Tensor, or a nested dict,
        list or tuple of Tensors.

        Returns:
        An single TensorSpec, or a nested dict, list or tuple of
        `TensorSpec` objects, which describe the shape and
        dtype of each Tensor expected by `step(_, policy_state)`.
        """
        return self._policy_state_spec

    @property
    def info_spec(self):
        """Describes the Tensors emitted as info by `action` and `distribution`.

        `info` can be an empty tuple, a single Tensor, or a nested dict,
        list or tuple of Tensors.

        Returns:
        An single TensorSpec, or a nested dict, list or tuple of
        `TensorSpec` objects, which describe the shape and
        dtype of each Tensor expected by `step(_, policy_state)`.
        """
        return self._info_spec
