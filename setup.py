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
        'absl-py',
        'atari_py==0.2.9',
        # used by Box2D-based environments (e.g. BipedalWalker, LunarLander)
        'box2d-py',
        'cpplint',
        'clang-format==17.0.6',
        'fasteners',
        'gin-config@git+https://github.com/HorizonRobotics/gin-config.git',
        'gym==0.15.4',
        'gym3==0.3.3',
        'h5py',
        'matplotlib',
        'numpy==1.26',
        'opencv-python',
        'pathos',
        'pillow>=8',
        'pre-commit',
        'protobuf',
        'psutil',
        'pybullet==2.5.0',
        'pybind11',
        'pyglet==1.3.2',  # higher version breaks classic control rendering
        'pylint',
        'rectangle-packer',
        'tensorboard',
        'threadpoolctl',
        'torch',
        'torchvision',
        'torchtext',
        'cnest@git+https://github.com/HorizonRobotics/cnest.git',
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
