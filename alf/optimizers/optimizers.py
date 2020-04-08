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

import copy
import gin
import torch

import alf
from alf.utils import common
from alf.utils import tensor_utils
from alf.optimizers.adam_tf import AdamTFUnwrapped


def wrap_optimizer(cls):
    """A helper function to construct torch optimizers with
    params as [{'params': []}]. After construction, new parameter
    groups can be adde by using the add_param_group() method.

    This wrapper also clips gradients first before calling ``step()``.
    """
    NewClsName = cls.__name__ + "_"
    NewCls = type(NewClsName, (cls, ), {})
    NewCls.counter = 0

    @common.add_method(NewCls)
    def __init__(self,
                 gradient_clipping=None,
                 clip_by_global_norm=False,
                 name=None,
                 **kwargs):
        """
        Args:
            gradient_clipping (float): If not None, serve as a positive threshold
            clip_by_global_norm (bool): If True, use `tensor_utils.clip_by_global_norm`
                to clip gradient. If False, use `tensor_utils.clip_by_norms` for
                each grad.
            name (str): the name displayed when summarizing the gradient norm. If
                None, then a global name in the format of "class_name_i" will be
                created, where "i" is the global optimizer id.
        """
        super(NewCls, self).__init__([{'params': []}], **kwargs)
        self._gradient_clipping = gradient_clipping
        self._clip_by_global_norm = clip_by_global_norm
        self._name = name
        if name is None:
            self._name = NewClsName + str(NewCls.counter)
            NewCls.counter += 1

    @common.add_method(NewCls)
    def step(self, closure=None):
        """This function first clips the gradients if needed, then call the
        parent's ``step()`` function.
        """
        if self._gradient_clipping is not None:
            params = []
            for param_group in self.param_groups:
                params.extend(param_group["params"])
            grads = alf.nest.map_structure(lambda p: p.grad, params)
            if self._clip_by_global_norm:
                _, global_norm = tensor_utils.clip_by_global_norm(
                    grads, self._gradient_clipping, in_place=True)
                if alf.summary.should_record_summaries():
                    alf.summary.scalar("global_grad_norm/%s" % self._name,
                                       global_norm)
            else:
                tensor_utils.clip_by_norms(
                    grads, self._gradient_clipping, in_place=True)
        super(NewCls, self).step(closure=closure)

    return NewCls


Adam = gin.external_configurable(wrap_optimizer(torch.optim.Adam), 'Adam')

AdamW = gin.external_configurable(wrap_optimizer(torch.optim.AdamW), 'AdamW')

SGD = gin.external_configurable(wrap_optimizer(torch.optim.SGD), 'SGD')

AdamTF = gin.external_configurable(wrap_optimizer(AdamTFUnwrapped), 'AdamTF')
