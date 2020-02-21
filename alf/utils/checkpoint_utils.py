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

import torch
import os


class Checkpointer(object):
    """Checkpoints training state, policy state, and replay_buffer state."""

    def __init__(self, ckpt_dir, **kwargs):
        """A class for making checkpoints.

        If ckpt_dir doesn't exists it creates it.

        Args:
        ckpt_dir: The directory to save checkpoints.
        max_to_keep: Maximum number of checkpoints to keep (if greater than the
            max are saved, the oldest checkpoints are deleted).
        **kwargs: Items to include in the checkpoint.
        """

        # def _to_dict(**kwargs):
        #     return {k: v for k, v in kwargs.items()}

        self._modules = kwargs
        self._ckpt_dir = ckpt_dir
        self._global_step = -1

        os.makedirs(self._ckpt_dir, exist_ok=True)

    def load(self, model_pass="latest"):
        """Load checkpoint
        """

        def _load_checkpoint(checkpoint):
            self._global_step = checkpoint["global_step"]
            for k, v in self._modules.items():
                self._modules[k].load_state_dict(checkpoint[k])

        f_path_latest = os.path.join(self._ckpt_dir, "latest")
        f_path = os.path.join(args.model_dir,
                              ('checkpoint-%s' % args.init_model_pass))
        if model_pass == "latest" and os.path.isfile(f_path_latest):
            checkpoint = torch.load(f_path_latest)
            _load_checkpoint(checkpoint)
        elif os.path.isfile(f_path):
            _load_checkpoint(checkpoint)

    def save(self, global_step):
        """Save states of all modules to checkpoint."""
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
