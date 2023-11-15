# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
from typing import Dict

import torch
from torch import nn


class StepMetric(nn.Module):
    """Defines the interface for metrics."""

    def __init__(self, name, dtype, prefix='Metrics'):
        super().__init__()
        self.name = name
        self._dtype = dtype
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

    def std(self):
        """Computes the standard deviation of the metric.

        Note that not all metrics have a meaningful standard deviation. For
        those metrics, this function returns 0.
        """
        return torch.zeros(())

    def gen_summaries(self,
                      train_step=None,
                      step_metrics=(),
                      other_steps: Dict[str, int] = dict()):
        """Generates summaries against train_step and all step_metrics.

        Args:
            train_step: (Optional) Step counter for training iterations. If None, no
                metric is generated against the global step.
            step_metrics: (Optional) Iterable of step metrics to generate summaries
                against.
            other_steps: A dictionary of steps to generate summaries against.
        """
        prefix = self._prefix
        result = self.result()

        if not (isinstance(result, dict) or alf.nest.is_namedtuple(result)):
            result = {self.name: result}

        def _gen_summary(name, res):
            tag = os.path.join(prefix, name)
            if train_step is not None:
                alf.summary.scalar(name=tag, data=res, step=train_step)
            for step_metric in step_metrics:
                # Skip plotting the metrics against itself.
                if self.name == step_metric.name:
                    continue
                step_tag = '{}_vs_{}/{}'.format(prefix, step_metric.name, name)
                # Summaries expect the step value to be an int64.
                step = step_metric.result().to(torch.int64)
                alf.summary.scalar(name=step_tag, data=res, step=step)
            for other_name, step in other_steps.items():
                step_tag = '{}_vs_{}/{}'.format(prefix, other_name, name)
                alf.summary.scalar(name=step_tag, data=res, step=step)

        alf.nest.py_map_structure_with_path(_gen_summary, result)
