# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import torch.multiprocessing as mp


class PerProcessContext(object):
    """A singletone that maintains the per process runtime properties.

    It is used mainly in multi-process distributed training mode,
    where properties such as the rank of the process and the total
    number of processes can be accessed via this interface.
    """
    _instance = None

    def __new__(cls):
        """Construct the singleton instance.

        This initializes the singleton and default values are assigned
        to the properties.
        """
        if cls._instance is None:
            cls._instance = super(PerProcessContext, cls).__new__(cls)
            cls._instance._read_only = False
            cls._instance._ddp_rank = -1
            cls._instance._num_processes = 1
        return cls._instance

    def finalize(self) -> None:
        """Lock the context so that it becomes read only.
        """
        self._read_only = True

    def set_distributed(self, rank: int, num_processes: int) -> None:
        """Set the distributed properties.

        Args:
            rank (int): the ID of the process
            num_processes (int): the total number of processes
        """
        if self._read_only:
            raise AttributeError(
                'Cannot mutate PerProcessContext after it is finalized')
        self._ddp_rank = rank
        self._num_processes = num_processes

    def set_paras_queue(self, paras_queue: mp.Queue):
        """Set the parameter queue.

        The queue is used for checking the consistency of model parameters across
        different worker processes, if multi-gpu training is used.
        """
        if self._read_only:
            raise AttributeError(
                'Cannot mutate PerProcessContext after it is finalized')
        self._paras_queue = paras_queue

    @property
    def paras_queue(self) -> mp.Queue:
        return self._paras_queue

    @property
    def is_distributed(self):
        return self._ddp_rank >= 0

    @property
    def ddp_rank(self):
        return self._ddp_rank

    @property
    def num_processes(self):
        return self._num_processes
