#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

if [[ "$NVIDIA_VISIBLE_DEVICES" =~ "GPU" ]]; then
    unset IDS
    GPUS=(${NVIDIA_VISIBLE_DEVICES//,/ })
    for gpu in ${GPUS[@]}
    do
        ID=$(nvidia-smi -L|grep $gpu| awk -F ":" '{print $1}' |awk '{print $2}')
        DISPLAY=":0.$ID"
        if [[ $IDS ]]; then
           IDS="$IDS,$ID"
        else
           IDS=$ID
        fi
    done
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES=$IDS
    export DISPLAY=$DISPLAY
    echo "export CUDA_DEVICE_ORDER=PCI_BUS_ID" >> ~/.bashrc
    echo "export CUDA_VISIBLE_DEVICES=$IDS" >> ~/.bashrc
    echo "export DISPLAY=$DISPLAY" >> ~/.bashrc
    echo "alias nvidia-smi='nvidia-smi -i $NVIDIA_VISIBLE_DEVICES'" >> ~/.bashrc
fi
