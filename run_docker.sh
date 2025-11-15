# flashdrive_cu124 == c43088098a8e
# flashdrive:cuda12.4_py3.10_alf_carla0.10 == 7736f9827fcc
# flashdrive:cuda12.4_py3.10_alf_carla0.10_jupyter == 50c7fa3a2677
# flashdrive-cpu-ubuntu22.04-py3.10-torch2.6.0-v3 == 021e70da91a9

# Note: For nvidia-smi to work, ensure NVIDIA Container Toolkit is installed on host:
#   sudo apt-get install -y nvidia-container-toolkit
#   sudo systemctl restart docker
docker run --gpus all \
        -d \
        --shm-size=256gb \
        --network host \
        --cap-add=SYS_ADMIN \
        -v /mnt/nas25:/mnt/nas25 \
        -v /mnt/nas26:/mnt/nas26 \
        -v /mnt/nas20:/mnt/nas20 \
        -v /home/users/qiang.liu/alf:/alf \
        -v /home/users/qiang.liu/wrk/E2E-RL:/E2E-RL \
        -e HF_ENDPOINT=https://hf-mirror.com \
        -e PYTHONPATH=/alf:/E2E-RL \
        -e PIP_NO_PROGRESS_BAR=1 \
        -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
        -p 8888:8888 \
        -it 50c7fa3a2677 \
        bash -c "apt update && apt install -y screen openssh-client && pip install --progress-bar off --no-cache-dir lxml ninja mediapy pybullet&& cd /E2E-RL && bash"
