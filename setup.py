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
        'absl-py==2.1.0',
        'atari_py==0.2.9',
        # used by Box2D-based environments (e.g. BipedalWalker, LunarLander)
        'fasteners',
        'gym==0.15.4',
        'gym3==0.3.3',
        'matplotlib==3.4.1',
        'numpy',
        'pathos==0.2.4',
        # with python3.7, the default version of pillow (PIL) is 8.2.0,
        # which breaks some pyglet based rendering in gym
        'pillow==7.2.0',
        # 'procgen>=0.10.4',
        'protobuf==3.20.1',
        'psutil==5.9.8',
        'pybullet==2.5.0',
        'pyglet==1.3.2',  # higher version breaks classic control rendering
        'rectangle-packer==2.0.0',
        'tensorboard==2.6.0',
        'torch==1.8.1',
        'torchvision==0.9.1',
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
        'metadrive': ['metadrive-simulator==0.2.5.1', ],
    },
    package_data={'': ['*.gin']},
    packages=find_packages(),
)
