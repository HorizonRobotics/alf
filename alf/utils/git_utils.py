# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""Git utilities."""
import os


def _exec(command, module_root):
    cwd = os.getcwd()
    os.chdir(module_root)
    stream = os.popen(command)
    ret = stream.read()
    stream.close()
    os.chdir(cwd)
    return ret


def get_revision(module_root: str):
    """Get the current revision of a python module at HEAD.

    Args:
        module_root: the path to the module root
    """
    return _exec("git rev-parse HEAD", module_root).strip()


def get_diff(module_root: str):
    """Get the diff of ALF at HEAD.

    If the repo is clean, the returned value is an empty string.

    Args:
        module_root: the path to the module root

    Returns:
        current diff.
    """
    return _exec("git -c core.fileMode=false diff --diff-filter=M",
                 module_root)
