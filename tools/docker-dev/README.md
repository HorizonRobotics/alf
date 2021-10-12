# Alf Docker Development Environment Tool

This tool is a script offering "one-click" experience of creating a docker-based developmenet environment that replicates the environment we use in Alf's CI.

## Usage

1. First you need to have a clone of the [alf repo](https://github.com/HorizonRobotics/alf/blob/pytorch/.ci-cd/build.sh).
2. Navigate to where the script is

   ``` bash
   $ cd alf/tools/docker-dev
   ```
3. Run the script 

   ``` bash
   $ ./box.sh
   ```

And after a few seconds you should be in the docker container that has everything needed to run alf, almost identical to the CI environment.

## Features

1. The only prerequisites: `docker` and `alf` repo.
2. It will check whether a docker container `alf-dev-box` exists. Only when the container does not exist, will it create a new `alf-dev-box` container.
2. It gets you directly to inside the docker container.
3. Your `$HOME` directory is automatically mounted in the container. Your host `alf` clone will be mounted as well if it is under `$HOME`.
4. The `bashrc` and `inputrc` are already configured.
5. `pip3 install -e <alf-root>` is automatically upon the creation of the docker container.
6. You should be able to run unit test directly with something like

   ``` bash
   $ python3 -m unittest -v alf.bin.train_play_test.TrainPlayTest.test_ppo_rnd_mrevenge
   ```
   
   Or whatever you aim to debug on, such as training
   
   ``` bash
   $ python3 -m alf.bin.train --root_dir ~/tmp/alf_sessions/ac_cart_pole/2 --conf alf/examples/ac_cart_pole_conf.py
   ```
