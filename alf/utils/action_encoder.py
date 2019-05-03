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
"""A simple parameterless action encoder."""

import tensorflow as tf
import tf_agents.specs.tensor_spec as tensor_spec


class SimpleActionEncoder(object):
    """A simple encoder for action.

    Only supports one action (discrete or continuous).
    If encode discrete action to one hot representation and use the original
    continous actions. And output the concat of all of them
    """

    def __init__(self, action_spec):
        """Create SimpleActionEncoder.

        Args:
            action_spec (nested BoundedTensorSpec): spec for actions
        """

        def check_supported_spec(spec):
            if tensor_spec.is_discrete(spec):
                assert len(spec.shape) == 0 or \
                    (len(spec.shape) == 1 and spec.shape[0] == 1)
            else:
                assert len(spec.shape) == 1

        tf.nest.map_structure(check_supported_spec, action_spec)
        self._action_spec = action_spec

    def __call__(self, inputs):
        """Generate encoded actions.

        Args:
            inputs (nested Tensor): action tensors.
        Returns:
            nested Tensor with the same structure as inputs.
        
        """
        tf.nest.assert_same_structure(inputs, self._action_spec)
        actions = inputs

        def encode_one_action(action, spec):
            if tensor_spec.is_discrete(spec):
                if len(spec.shape) == 1:
                    action = tf.reshape(action, action.shape[:-1])
                num_actions = spec.maximum - spec.minimum + 1
                return tf.one_hot(indices=action, depth=num_actions)
            else:
                return action

        actions = tf.nest.map_structure(encode_one_action, actions,
                                        self._action_spec)

        return tf.concat(tf.nest.flatten(actions), axis=-1)
