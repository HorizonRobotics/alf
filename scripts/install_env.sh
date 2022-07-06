#! /usr/bin/env bash

conda create --name alf python=3.8 && conda activate alf
conda install -c anaconda swig
pip install -e .
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
conda install patchelf
pip install install -U 'mujoco-py<2.2,>=2.1'
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
pip install wandb ninja functorch magic-wormhole
pip uninstall pathos && pip install pathos
