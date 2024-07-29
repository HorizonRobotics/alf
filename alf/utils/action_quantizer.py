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
"""Action Quantizer."""

from absl import logging
import numpy as np
import functools

import torch

import alf


@alf.configurable
class ActionQuantizer(object):
    def __init__(self,
                 action_spec,
                 sampling_method="uniform",
                 action_bins=7,
                 rep_mode="center"):
        """Quantize actions in a specified way.

        Args:
            action_spec (BoundedTensorSpec): action spec
            sampling_method (str): sampling space, uniform or log space：

                - "uniform": the original space
                - "log": the logarithm space

            action_bins (int): number of bins used for discretization
            rep_mode (str): the mode of representation for quantization:

                - "center": linspace(lb + bin-size/2, ub - bin_size/2, bin_num)
                - "boundary": linspace(lower_bound, upper_bound, bin_num)
        """
        super().__init__()

        self._action_spec = action_spec
        self._action_bins = action_bins

        self._sampling_method = sampling_method
        self._rep_mode = rep_mode

        # action_dim: the length of the continuous control vector
        # action_bins: the number of elements per action_dim

        self._action_dim = action_spec.shape[0]
        assert (() == action_spec.maximum.shape) and \
                (() == action_spec.minimum.shape), \
                    "Only support scalar action maximum and minimum bound"

        self._upper_bound = action_spec.maximum.item()
        self._lower_bound = action_spec.minimum.item()

        # TODO: currently use the same quantization across action dims;
        # can make it different for different dims in the future
        if self._sampling_method == "uniform":
            if self._rep_mode == "center":
                bin_size = (
                    self._upper_bound - self._lower_bound) / self._action_bins
                # [lb + bin_size/2, up - bin_size/2]
                # center value representation
                LUT_BA = torch.linspace(
                    self._lower_bound + bin_size / 2,
                    self._upper_bound - bin_size / 2,
                    steps=self._action_bins)
            elif self._rep_mode == "boundary":
                LUT_BA = torch.linspace(
                    self._lower_bound,
                    self._upper_bound,
                    steps=self._action_bins)
        else:
            raise NotImplementedError("Unimplemented sampling method!")

        # look-up-table, bin-to-action mapping
        self._LUT_BA = LUT_BA

        self._bin_size = self._LUT_BA[1] - self._LUT_BA[0]

    def ind_to_action(self, action_ind):
        action = action_ind * self._bin_size + self._LUT_BA[0]
        return action

    def action_to_ind(self, action):
        bin_size = self._bin_size
        if self._rep_mode == "center":
            action_ind = torch.round((action - self._LUT_BA[0]) / bin_size)
        elif self._rep_mode == "boundary":
            action_ind = (action - self._LUT_BA[0]) // (bin_size)
        else:
            raise NotImplementedError("Unsupported representation mode!")

        return action_ind

    @property
    def action_bins(self):
        return self._action_bins
