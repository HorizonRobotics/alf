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
import os

os.system("pip install -e ./alf/nest/cnest")

setup(
    name='alf',
    version='0.0.3',
    python_requires='>3.6.0',
    install_requires=[
        'atari_py == 0.1.7',
        'cpplint',
        'clang-format == 9.0',
        'fasteners',
        'gin-config@git+https://github.com/HorizonRobotics/gin-config.git',
        'gym == 0.12.5',
        'pyglet == 1.3.2',  # higher version breaks classic control rendering
        'matplotlib==3.4.1',
        'numpy',
        'opencv-python >=4.0, <=4.2',
        'pathos == 0.2.4',
        'pillow',
        'psutil',
        'pybullet == 2.5.0',
        'rectangle-packer==2.0.0',
        'sphinx==2.4.4',
        'sphinxcontrib-napoleon==0.7',
        'sphinx-rtd-theme==0.4.3',  # used to build html docs locally
        'tensorboard == 2.1.0',
        'torch == 1.8.1',
        'torchvision == 0.9.1',
    ],  # And any other dependencies foo needs
    package_data={'': ['*.gin']},
    packages=find_packages(),
)
