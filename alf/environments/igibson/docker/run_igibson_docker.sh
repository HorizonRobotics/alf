#!/usr/bin/env bash

while getopts t: flag
do
    case "${flag}" in
        t) tag=${OPTARG};;
    esac
done
echo "Running image: $tag";
sudo docker run --gpus all --runtime=nvidia -v $HOME/.bashrc:/home/.bashrc -v $HOME/alf:/home/alf -v $HOME/iGibson/results:/home/results -it $tag /bin/bash
