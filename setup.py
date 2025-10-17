# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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

from pathlib import Path

from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

try:
    # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for Python 3.10
    import tomli as tomllib

PYPROJECT_PATH = Path(__file__).with_name("pyproject.toml")
with PYPROJECT_PATH.open("rb") as f:
    pyproject = tomllib.load(f)

project = pyproject.get("project", {})
optional_dependencies = project.get("optional-dependencies", {})

setup(
    name=project.get('name', 'alf'),
    version=project.get('version', '0.0.0'),
    python_requires=project.get('requires-python', '>=3.10.0'),
    install_requires=project.get('dependencies', []),
    cmdclass={'build_ext': build_ext},
    extras_require=optional_dependencies,
    package_data={'': ['*.gin']},
    packages=find_packages(),
)
