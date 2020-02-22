# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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
#
"""A few metrics.
Code adapted from https://github.com/tensorflow/agents/blob/master/tf_agents/metrics/tf_metric.py
"""

import alf
import os

from torch import nn


class StepMetric(nn.Module):
    """Defines the interface for metrics."""

    def __init__(self, name, prefix='Metrics'):
        super().__init__()
        self.name = name
        self._prefix = prefix

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        """Accumulates statistics for the metric.
    Args:
      *args:
      **kwargs: A mini-batch of inputs to the Metric.
    """
        raise NotImplementedError(
            'Metrics must define a call() member function')

    def forward(self, *args, **kwargs):
        pass

    def reset(self):
        """Resets the values being tracked by the metric."""
        raise NotImplementedError(
            'Metrics must define a reset() member function')

    def result(self):
        """Computes and returns a final value for the metric."""
        raise NotImplementedError(
            'Metrics must define a result() member function')

    def gen_summaries(self, train_step=None, step_metrics=()):
        """Generates summaries against train_step and all step_metrics.
    Args:
      train_step: (Optional) Step counter for training iterations. If None, no
        metric is generated against the global step.
      step_metrics: (Optional) Iterable of step metrics to generate summaries
        against.
    Returns:
      A list of summaries.
    """
        summaries = []
        prefix = self._prefix
        tag = os.path.join(prefix, self.name)
        result = self.result()
        if train_step is not None:
            summaries.append(
                alf.summary.scalar(name=tag, data=result, step=train_step))
        if prefix:
            prefix += '_'
        for step_metric in step_metrics:
            # Skip plotting the metrics against itself.
            if self.name == step_metric.name:
                continue
            step_tag = '{}vs_{}/{}'.format(prefix, step_metric.name, self.name)
            # Summaries expect the step value to be an int64.
            step = step_metric.result().int64()
            summaries.append(
                alf.summary.scalar(name=step_tag, data=result, step=step))
        return summaries
