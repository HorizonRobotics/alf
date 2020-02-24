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

import os
import warnings
import torch


class Checkpointer(object):
    """A checkpoint manager for saving and loading checkpoints."""

    def __init__(self, ckpt_dir, **kwargs):
        """A class for making checkpoints.

        Args:
            ckpt_dir: The directory to save checkpoints. Create ckpt_dir if
                it doesn't exist.
            kwargs: Items to be included in the checkpoint. Each item needs
                to have state_dict and load_state_dict implemented.
        """

        self._modules = kwargs
        self._ckpt_dir = ckpt_dir
        self._global_step = -1

        os.makedirs(self._ckpt_dir, exist_ok=True)

    def load(self, global_step="latest"):
        """Load checkpoint
        Args:
            global_step (int|str): the number of training steps which is used to
                specify the checkpoint to be loaded. If global_step is 'latest',
                the most recent checkpoint named 'latest' will be loaded.
        Returns:
            current_step_num (int): the current step number for the loaded
                checkpoint.
        """

        def _load_checkpoint(checkpoint):
            self._global_step = checkpoint["global_step"]
            for k, v in self._modules.items():
                self._modules[k].load_state_dict(checkpoint[k])

        f_path_latest = os.path.join(self._ckpt_dir, "latest")
        f_path = os.path.join(self._ckpt_dir, "ckpt-{0}".format(global_step))
        if global_step == "latest" and os.path.isfile(f_path_latest):
            checkpoint = torch.load(f_path_latest)
            _load_checkpoint(checkpoint)
        elif os.path.isfile(f_path):
            checkpoint = torch.load(f_path)
            _load_checkpoint(checkpoint)
        else:
            warnings.warn(("Checkpoint 'ckpt-{}' does not exist. "
                           "Train from scratch.".format(global_step)))

        return self._global_step

    def save(self, global_step):
        """Save states of all modules to checkpoint
        Args:
            global_step (int): the number of training steps corresponding to the
                current state to be saved. It will be appended to the name of
                the checkpoint as a suffix. This function will also save a copy
                of the latest checkpoint in a file named 'latest'.
        """
        self._global_step = global_step
        f_path = os.path.join(self._ckpt_dir, "ckpt-{0}".format(global_step))
        state = {
            k: v.module.state_dict()
            if type(v) == torch.nn.DataParallel else v.state_dict()
            for k, v in self._modules.items()
        }
        state['global_step'] = global_step
        torch.save(state, f_path)

        f_path_latest = os.path.join(self._ckpt_dir, "latest")
        torch.save(state, f_path_latest)
