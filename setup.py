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
    python_requires='>=3.10.0',
    install_requires=[
        'absl-py==2.2.2',
        'atari_py==0.2.9',
        'box2d-py==2.3.8',  # used by Box2D-based environments (e.g. BipedalWalker, LunarLander)
        'clang-format==17.0.6',
        'cnest@git+https://github.com/HorizonRobotics/cnest.git',
        'cpplint==2.0.1',
        'fasteners==0.19',
        'gin-config@git+https://github.com/HorizonRobotics/gin-config.git',
        'gym==0.15.4',
        'gym3==0.3.3',
        'h5py==3.13.0',
        'matplotlib==3.10.1',
        'numpy==1.26',
        'opencv-python==4.11.0.86',
        'pathos==0.3.3',
        'pillow>=8',
        'pre-commit==4.2.0',
        'protobuf==6.30.2',
        'psutil==7.0.0',
        'pybind11==2.13.6',
        'pybullet==2.5.0',
        'pylint==3.3.6',
        'pyglet==1.3.2',  # higher version breaks classic control rendering
        'rectangle-packer==2.0.4',
        'tensorboard==2.19.0',
        'threadpoolctl==3.6.0',
        'torch==2.6.0',
        'torchtext==0.18.0',
        'torchvision==0.21.0',
        'wheel'
    ],  # And any other dependencies alf needs
    cmdclass={'build_ext': build_ext},
    extras_require={
        'metadrive': ['metadrive-simulator==0.2.5.1', ],
        'docs': [
            'sphinx==3.0',
            'sphinx-autobuild',
            'sphinx-autodoc-typehints@git+https://github.com/hnyu/sphinx-autodoc-typehints.git',
            'sphinxcontrib-napoleon==0.7',
            'sphinx-rtd-theme==0.4.3',  # used to build html docs locally
        ]
    },
    package_data={'': ['*.gin']},
    packages=find_packages(),
)
