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

from setuptools import setup, find_packages

setup(
    name='alf',
    version='0.0.1',
    install_requires=[
        'atari_py == 0.1.7',
        'fasteners',
        'gin-config',
        'gym == 0.12.5',
        'matplotlib',
        'numpy',
        'opencv-python >= 3.4.1.15',
        'pathos == 0.2.4',
        'pillow',
        'psutil',
        'pybullet == 2.5.0',
        'tensorboard == 2.1.0',
        'torch == 1.4.0',
        'torchvision == 0.5.0',
    ],  # And any other dependencies foo needs
    package_data={'': ['*.gin']},
    packages=find_packages(),
)
