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

import re
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
                 module_blacklist: List[str] = None,
                 module_whitelist: List[str] = None,
                 name: str = 'PretrainedModel'):
        """
        Args:
            model: the base pretrained model whose weights will be used as frozen
            adapter_cls: an optional list of adapter classes applied to the base
                model layers. An adapter instance of each class will be created
                for each of all qualified layers. The adapter weights will be
                stored in the adapter instance.
            module_blacklist: an optional blacklist of modules not to be adapted.
                Each entry can be a regex or a substring of the module name.
            module_whitelist: an optional whitelist of modules not to be adapted.
                Each entry can be a regex or a substring of the module name. By
                default this is None which means all modules are valid. Only at
                most one of ``module_blacklist`` and ``module_whitelist`` can be
                provided.
            name: name of the pretrained model
        """
        super().__init__()
        assert not (module_blacklist and module_whitelist), (
            "Blacklist and whitelist cannot be provided at the same time!")
        self._name = name
        # Freeze all parameters
        for para in model.parameters():
            para.requires_grad = False
        # Use a list trick to let pytorch ignore this module for checkpointing
        self._model = [model]
        # This is the ONLY module that affects checkpointing
        self._adapters = nn.ModuleList()
        self._adapted_module_names = []
        for name, m in self.model.named_modules():
            skip = False
            if module_blacklist is not None:
                for b in module_blacklist:
                    if re.search(b, name):
                        skip = True
                        break
            elif module_whitelist is not None:
                skip = True
                for w in module_whitelist:
                    if re.search(w, name):
                        skip = False
                        break
            if skip:
                continue

            for acls in adapter_cls:
                if acls.can_adapt(m):
                    self._adapters.append(acls(m))
                    self._adapted_module_names.append(name)

    @property
    def model(self) -> nn.Module:
        """Return the base model."""
        return self._model[0]

    @property
    def adapted_module_names(self) -> List[str]:
        """Return a list of adapted module names, in the adapter adding order.
        """
        return self._adapted_module_names

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

        Note that even after merging, when we save a checkpoint for this pretrained
        model, we still only save the adapter weights only. In other words, whether
        the adapters are merged or not is transparent to pytorch's checkpointing.
        """
        for a in self._adapters:
            a.merge()

    def reset_adapter(self):
        """Reset the adapter weights.
        """
        for a in self._adapters:
            a.reset_parameters()

    def forward(self, input):
        return self.model(input)
