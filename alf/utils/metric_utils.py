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

import itertools
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.policies.tf_policy import Base
from tf_agents.eval.metric_utils import eager_compute as tfa_eager_compute
from alf.utils import common


class Policy(Base):
    """Wrap an action fn to policy."""

    def __init__(self, time_step_spec, action_spec, policy_state_spec,
                 action_fn):
        """Create a Policy.

        Args:
            time_step_spec (TimeStep): spec of the expected time_steps.
            action_spec (nested BoundedTensorSpec): representing the actions.
            policy_state_spec (nested TensorSpec): representing the policy_state.
            action_fn (Callable): action function that generates next action
                given the time_step and policy_state
        """
        super(Policy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy_state_spec=(policy_state_spec, action_spec))
        # attribute named `_action_fn` already exist in parent
        self._action_fn1 = action_fn

    def _action(self, time_step, policy_state=(), seed=None):
        policy_step = self._action_fn1(
            common.make_action_time_step(time_step, policy_state[1]),
            policy_state[0])
        policy_step = policy_step._replace(
            state=(policy_step.state, policy_step.action))
        return policy_step


def eager_compute(metrics,
                  environment,
                  state_spec,
                  action_fn,
                  num_episodes=1,
                  step_metrics=(),
                  train_step=None,
                  summary_writer=None,
                  summary_prefix=''):
    """Compute metrics using `action_fn` on the `environment`.

    Args:
        metrics (list[TFStepMetric]): metrics to compute.
        environment (TFEnvironment): tf_environment instance.
        state_spec (nested TensorSpec): representing the RNN state for predict.
        action_fn (Callable): action function used to step the environment that
            generates next action given the time_step and policy_state.
        num_episodes (int): Number of episodes to compute the metrics over.
        step_metrics (list[TFStepMetric]): Iterable of step metrics to generate summaries
            against.
        train_step (Variable): An optional step to write summaries against.
        summary_writer (SummaryWriter): An optional writer for generating metric summaries.
        summary_prefix (str): An optional prefix scope for metric summaries.
    Returns:
        A dictionary of results {metric_name: metric_value}
    """

    policy = Policy(environment.time_step_spec(), environment.action_spec(),
                    state_spec, action_fn)
    metric_results = tfa_eager_compute(
        metrics=metrics,
        environment=environment,
        policy=policy,
        num_episodes=num_episodes,
        train_step=train_step,
        summary_writer=summary_writer,
        summary_prefix=summary_prefix)

    if train_step and summary_writer:
        for metric_result, step_metric in itertools.product(
                metric_results.items(), step_metrics):
            metric_name, metric_value = metric_result
            step_tag = '{}_vs_{}/{}'.format(summary_prefix, step_metric.name,
                                            metric_name)
            step = step_metric.result()
            with summary_writer.as_default():
                tf.summary.scalar(name=step_tag, data=metric_value, step=step)

    return metric_results
