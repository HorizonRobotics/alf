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
"""Git utilities."""
import os


def _get_repo_root():
    """Get ALF repo root path."""
    return os.path.join(os.path.dirname(__file__), "..", "..")


def _exec(command):
    cwd = os.getcwd()
    os.chdir(_get_repo_root())
    stream = os.popen(command)
    ret = stream.read()
    stream.close()
    os.chdir(cwd)
    return ret


def get_revision():
    """Get the current revision of ALF at HEAD."""
    return _exec("git rev-parse HEAD").strip()


def get_diff():
    """Get the diff of ALF at HEAD.

    If the repo is clean, the returned value is an empty string.
    Returns:
        current diff.
    """
    return _exec("git diff")
