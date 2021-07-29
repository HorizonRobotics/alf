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
r"""Utility that provides APIs to store and load abseil flags

.. code-block:: python
    flags_cache.store(root_dir)

This stores the flags into a cache file under root_dir.

.. code-block:: python
    flags_cache.load(root_dir)

This loads the flags from the cache file under root_dir. If the load
is successful, the FLAGS object will also be marked as parsed, so that
it is ready for normal use.
"""

from absl import flags
from pathlib import Path


def store(root_dir: str) -> None:
    """Snapshot and store the current flags to the cache file

    Args:
        root_dir (str): Path to the directory under which we save the
            flags cache file to.
    """
    cache_file = Path(root_dir, 'flags', 'cached_flags.txt')
    cache_file.parent.mkdir(exist_ok=True, parents=True)
    if cache_file.exists():
        cache_file.unlink()
    flags.FLAGS.append_flags_into_file(cache_file)


def load(root_dir: str) -> None:
    """Load the flags from the cacche file

    The absl FLAGS object will be marked as parsed after a successful load.
    After load() FLAGS can be used just as parsed from command line.

    Args:
        root_dir (str): Path to the directory under which we load the
            flags cache file from.
    """
    cache_file = Path(root_dir, 'flags', 'cached_flags.txt')
    if not cache_file.exists():
        raise FileNotFoundError(
            f'Cannot load flags cache: {cache_file} does not exist')
    argv = flags.FLAGS.read_flags_from_files([f'--flagfile={cache_file}'])
    flags.FLAGS([__file__] + argv, known_only=True)
    flags.FLAGS.mark_as_parsed()
