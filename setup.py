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
from pybind11.setup_helpers import Pybind11Extension, build_ext

setup(
    name='alf',
    version='0.1.0',
    python_requires='>=3.8.0',
    install_requires=[
        'atari_py',
        # used by Box2D-based environments (e.g. BipedalWalker, LunarLander)
        'fasteners',
        'gym',
        'gym3',
        'matplotlib',
        'numpy',
        'pathos',
        # with python3.7, the default version of pillow (PIL) is 8.2.0,
        # which breaks some pyglet based rendering in gym
        'pillow',
        'procgen',
        'protobuf',
        'psutil',
        'pybullet',
        'pyglet',  # higher version breaks classic control rendering
        'rectangle-packer',
        'tensorboard',
        'torch',
        'torchvision',
    ],  # And any other dependencies alf needs
    ext_modules=[
        Pybind11Extension(
            'alf.environments._penv',
            sources=['alf/environments/parallel_environment.cpp'],
            extra_compile_args=[
                '-O3', '-Wall', '-std=c++17', '-fPIC', '-fvisibility=hidden'
            ])
    ],
    cmdclass={'build_ext': build_ext},
    extras_require={
        'metadrive': ['metadrive-simulator', ],
    },
    package_data={'': ['*.gin']},
    packages=find_packages(),
)
