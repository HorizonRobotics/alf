# Copyright (c) 2023 Horizon Robotics and Hobot Contributors. All Rights Reserved.
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

import torch
import torch.nn as nn

from typing import Callable, List

from .model_adapters.lora import LoRA


class PretrainedModel(nn.Module):
    """A wrapper class for managing pretrained models.

    A pretrained model is generally large and its weights will always be frozen.
    For finetuning, we can add small adapters to some of its layers and only the
    adapter weights will be trained on downstream tasks. See
    `<https://docs.adapterhub.ml/methods.html>`_.
    """

    def __init__(self,
                 model: nn.Module,
                 adapter_cls: List[Callable] = [],
                 name: str = 'PretrainedModel'):
        """
        Args:
            model: the base pretrained model whose weights will be used as frozen
            adapter_cls: an optional list of adapter classes applied to the base
                model layers. An adapter instance of each class will be created
                for each of all qualified layers. The adapter weights will be
                stored in the adapter instance.
            name: name of the pretrained model
        """
        super().__init__()
        # Freeze all parameters
        for para in model.parameters():
            para.requires_grad = False
        self._model = model
        self._adapters = nn.ModuleList()
        for m in self._model.modules():
            for acls in adapter_cls:
                if acls.can_adapt(m):
                    self._adapters.append(acls(m))
        self._name = name

    def remove_adapter(self) -> nn.ModuleList:
        """Remove the adapter (if existed).

        After the removal, ``forward()`` will only use the frozen weights. This
        operation is irreversible as the adapter can no longer be added back.

        Returns:
            nn.ModueList: a list of the adapters, in case their weights are needed.
        """
        for a in self._adapters:
            a.detach()
        adapters = self._adapters
        self._adapters = nn.ModuleList()
        return adapters

    def unmerge_adapter(self):
        """Unmerge adapter weights to enable training.
        """
        for a in self._adapters:
            a.unmerge()

    def merge_adapter(self):
        """Merge adapter weights into the model for efficient inference.
        """
        for a in self._adapters:
            a.merge()

    def reset_adapter(self):
        """Reset the adapter weights.
        """
        for a in self._adapters:
            a.reset_parameters()

    def forward(self, input):
        return self._model(input)

    def state_dict(self, *args, **kwargs):
        """Only return the adapter's state dict because we don't want to
        put the large base model into our checkpoints.
        """
        return self._adapters.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """Only load the state dict for the adapter.
        """
        return self._adapters.load_state_dict(*args, **kwargs)
