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

import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.policies.tf_policy import Base
from tf_agents.eval.metric_utils import eager_compute as tfa_eager_compute


class Policy(Base):
    """Wrap a action fn to policy """

    def __init__(self, time_step_spec,
                 action_spec, policy_state_spec, action_fn):
        super(Policy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy_state_spec=policy_state_spec)
        self._action_fn1 = action_fn

    def _action(self, time_step, policy_state=(), seed=None):
        policy_step = self._action_fn1(time_step, policy_state)
        seed_stream = tfp.distributions.SeedStream(seed=seed, salt='policy_proxy')
        action = tf.nest.map_structure(lambda d: d.sample(seed=seed_stream()),
                                       policy_step.action)
        policy_step = policy_step._replace(action=action)
        return policy_step


def eager_compute(metrics,
                  environment,
                  state_spec,
                  action_fn,
                  num_episodes=1,
                  train_step=None,
                  summary_writer=None,
                  summary_prefix=''):
    """Compute metrics using `action_fn` on the `environment`."""

    policy = Policy(environment.time_step_spec(),
                    environment.action_spec(),
                    state_spec, action_fn)
    return tfa_eager_compute(metrics,
                             environment,
                             policy,
                             num_episodes,
                             train_step,
                             summary_writer,
                             summary_prefix)
