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

import numpy as np
import torch
import torch.nn as nn


class MultiGPU(nn.DataParallel):
    """Multi-GPU Parallel
    """

    def __init__(self, module, gpu_ids=None):
        """
        Args:
            module: module to be wrapped; the module is assumed to be on gpu to
                be able to use MultiGPU parallization; if module is
                on cpu and gpu_ids=None, it is equivalent to cpu mode
            gpu_ids (list[int]): gpu ids, e.g. [0], [0, 1], [1, 3]. If None, will
                use all available gpus
        """

        gpu_ids = np.unique(
            gpu_ids).tolist() if gpu_ids is not None else gpu_ids

        assert gpu_ids is None or len(gpu_ids) <= torch.cuda.device_count(), \
            ("only {} GPUs are available; ".format(torch.cuda.device_count()),
            "input `gpu_ids` is {}".format(gpu_ids))

        super().__init__(module, device_ids=gpu_ids)

    def __getattr__(self, name):
        """ Making the customized function visible
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
