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
