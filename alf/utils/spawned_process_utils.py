# Copyright (c) 2023 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""spawned_process_utils.py: Manage global context for subprocesses in ALF.

This module provides functionality to handle global context for subprocesses
spawned by the main ALF process, particularly for creating a `ProcessEnvironment`.
It alters the behavior of ALF functions like `get_env()` in spawned subprocesses
by maintaining and accessing a shared context.

"""

from typing import Any, Callable, List, NamedTuple, Optional, Tuple

from alf.environments.alf_environment import AlfEnvironment


class SpawnedProcessContext(NamedTuple):
    """Stores context information inherited from the main process.

    """
    ddp_num_procs: int
    ddp_rank: int
    env_id: int
    env_ctor: Callable[..., AlfEnvironment]
    pre_configs: List[Tuple[str, Any]]

    def create_env(self):
        """Creates an environment instance using the stored context."""
        return self.env_ctor(env_id=self.env_id)


_SPAWNED_PROCESS_CONTEXT = None


def set_spawned_process_context(context: SpawnedProcessContext):
    """Sets the context for the current spawned process.

    Should be called at the start of a spawned process by the main ALF process.

    Args:
        context (SpawnedProcessContext): The context to be stored.
    """
    global _SPAWNED_PROCESS_CONTEXT
    _SPAWNED_PROCESS_CONTEXT = context


def get_spawned_process_context() -> Optional[SpawnedProcessContext]:
    """Retrieves the spawned process context, if available.

    Returns:
        Optional[SpawnedProcessContext]: The current spawned process context.
    """
    return _SPAWNED_PROCESS_CONTEXT
