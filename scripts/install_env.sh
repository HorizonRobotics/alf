#! /usr/bin/env bash

conda create --name alf python=3.8 && conda activate alf
conda install -c anaconda swig
pip install -e .
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
conda install patchelf
pip install install -U 'mujoco-py<2.2,>=2.1'
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
# pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install wandb ninja functorch magic-wormhole graphviz
pip uninstall pathos && pip install pathos
