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

setup(
    name='alf',
    version='0.0.6',
    python_requires='>=3.7.0',
    install_requires=[
        'atari_py==0.1.7',
        # used by Box2D-based environments (e.g. BipedalWalker, LunarLander)
        'box2d-py',
        'cpplint',
        'clang-format==9.0',
        'fasteners',
        'gin-config@git+https://github.com/HorizonRobotics/gin-config.git',
        'gym==0.15.4',
        'gym3==0.3.3',
        'h5py==3.5.0',
        'matplotlib==3.4.1',
        'numpy',
        'opencv-python',
        'pathos==0.2.4',
        # with python3.7, the default version of pillow (PIL) is 8.2.0,
        # which breaks some pyglet based rendering in gym
        'pillow==7.2.0',
        'procgen==0.10.4',
        'psutil',
        'pybullet==2.5.0',
        'pyglet==1.3.2',  # higher version breaks classic control rendering
        'rectangle-packer==2.0.0',
        'sphinx==3.0',
        'sphinx-autobuild',
        'sphinx-autodoc-typehints@git+https://github.com/hnyu/sphinx-autodoc-typehints.git',
        'sphinxcontrib-napoleon==0.7',
        'sphinx-rtd-theme==0.4.3',  # used to build html docs locally
        'tensorboard==2.6.0',
        'torch==1.8.1',
        'torchvision==0.9.1',
        'torchtext==0.9.1',
        'cnest==1.0.4',
    ],  # And any other dependencies alf needs
    extras_require={
        'metadrive': ['metadrive-simulator==0.2.5.1', ],
    },
    package_data={'': ['*.gin']},
    packages=find_packages(),
)
