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
from absl import logging

from alf.algorithms.algorithm import Algorithm


class Checkpointer(object):
    """A checkpoint manager for saving and loading checkpoints."""

    def __init__(self, ckpt_dir, **kwargs):
        """A class for making checkpoints.

        Args:
            ckpt_dir: The directory to save checkpoints. Create ckpt_dir if
                it doesn't exist.
            kwargs: Items to be included in the checkpoint. Each item needs
                to have state_dict and load_state_dict implemented.
                For instance of Algorithm, only the root need to be passed in,
                all the children modules and optimizers are automatically
                extracted and checkpointed. If a children module is also passed
                in, it will be treated as the root to be recursively processed.
        Example usage:
        ```python
            alg_root = MyAlg(params=[p1, p2], sub_algs=[a1, a2], optimizer=opt)
            ckpt_mngr = ckpt_utils.Checkpointer(ckpt_dir,
                                alg=alg_root)
        ```

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
                checkpoint. current_step_num is set to - 1 if the specified
                checkpoint does not exist.
        """

        def _load_checkpoint(checkpoint):
            self._global_step = checkpoint["global_step"]
            try:
                for k, v in self._modules.items():
                    self._modules[k].load_state_dict(checkpoint[k])
            except:
                raise RuntimeError(
                    ("Checkpoint loading failed due to a mis-match between the"
                     "checkpoint and the model."))

        f_path_latest = os.path.join(self._ckpt_dir, "latest")
        f_path = os.path.join(self._ckpt_dir, "ckpt-{0}".format(global_step))
        map_location = None
        if not torch.cuda.is_available():
            map_location = torch.device('cpu')
        if global_step == "latest" and os.path.isfile(f_path_latest):
            checkpoint = torch.load(f_path_latest, map_location=map_location)
            _load_checkpoint(checkpoint)
            logging.info("Checkpoint 'latest' is loaded successfully.")
        elif os.path.isfile(f_path):
            checkpoint = torch.load(f_path, map_location=map_location)
            _load_checkpoint(checkpoint)
            logging.info("Checkpoint 'ckpt-{}' is loaded successfully.".format(
                global_step))
        else:
            warnings.warn(
                ("Checkpoint '{}' does not exist. "
                 "Train from scratch.".
                 format(global_step if global_step == "latest" else "ckpt-%d" %
                        global_step)))

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
        logging.info(
            "Checkpoint 'ckpt-{}' is saved successfully.".format(global_step))

        f_path_latest = os.path.join(self._ckpt_dir, "latest")
        torch.save(state, f_path_latest)
        logging.info("Checkpoint 'latest' is saved successfully.")
