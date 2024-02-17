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

import os
import sys
import subprocess


def gen_penv():
    current_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(current_path)
    os.chdir(dir_path)
    # install pybind11 if it's not installed.
    try:
        import pybind11
    except:
        assert os.system(
            "pip install pybind11") == 0, "Fail to pip install pybind11"
    python = f"python{sys.version_info.major}.{sys.version_info.minor}"
    cmd = (f"g++ -O3 -Wall -shared -std=c++17 -fPIC -fvisibility=hidden "
           f"`{python} -m pybind11 --includes` parallel_environment.cpp "
           f"-o _penv`{python}-config --extension-suffix` -lrt")
    ret = subprocess.run(["/bin/bash", "-c", cmd])
    assert ret.returncode == 0, "Fail to execute " + cmd


if __name__ == "__main__":
    gen_penv()
