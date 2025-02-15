# ALF
<p align="center">
    <img src="docs/images/logo.png" width = "500" alt="ALF-logo"/>
</p>

![CI](https://github.com/HorizonRobotics/alf/workflows/CI/badge.svg?branch=pytorch)

Agent Learning Framework (ALF) is a reinforcement learning framework emphasizing on the flexibility and easiness of implementing complex algorithms involving many different components. ALF is built on [PyTorch](https://pytorch.org/). The development of [previous version](https://github.com/HorizonRobotics/alf/tree/tensorflow) based on [Tensorflow 2.1](https://www.tensorflow.org/) has stopped as of Feb 2020.

## Tutorial
A draft tutorial can be accessed on [RTD](https://alf.readthedocs.io/en/latest/tutorial.html). This tutorial is still under
construction and some chapters are unfinished yet.

## Documentation

Read the ALF documentation [here](https://alf.readthedocs.io/).

## Algorithms

|Algorithm|Type|Reference|
|---------|----|---------|
|[A2C](alf/algorithms/actor_critic_algorithm.py)|On-policy RL|[OpenAI Baselines: ACKTR & A2C](https://openai.com/blog/baselines-acktr-a2c/)|
|[PPO](alf/algorithms/ppo_algorithm.py)|On-policy RL|Schulman et al. "Proximal Policy Optimization Algorithms" [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)|
|[PPG](alf/algorithms/ppo_algorithm.py)|On-policy RL|Cobbe et al. "Phasic Policy Gradient" [arXiv:2009.04416](https://arxiv.org/abs/2009.04416)|
|[DDQN](alf/algorithms/dqn_algorithm.py)|Off-policy RL|Hasselt et al. "Deep Reinforcement Learning with Double Q-learning" [arXiv:1509.06461](https://arxiv.org/abs/1509.06461)|
|[DDPG](alf/algorithms/ddpg_algorithm.py)|Off-policy RL|Lillicrap et al. "Continuous control with deep reinforcement learning" [arXiv:1509.02971](https://arxiv.org/abs/1509.02971)|
|[QRSAC](alf/algorithms/qrsac_algorithm.py)|Off-policy RL|Dabney et al. "Distributional Reinforcement Learning with Quantile Regression" [arXiv:1710.10044](https://arxiv.org/abs/1710.10044)|
|[SAC](alf/algorithms/sac_algorithm.py)|Off-policy RL|Haarnoja et al. "Soft Actor-Critic Algorithms and Applications" [arXiv:1812.05905](https://arxiv.org/abs/1812.05905)|
|[RLPD](alf/algorithms/rlpd_algorithm.py)|Off-policy RL|Ball et al. "Efficient Online Reinforcement Learning with Offline Data" [arXiv:2302.02948](https://arxiv.org/abs/2302.02948)|
|[OAC](alf/algorithms/oac_algorithm.py)|Off-policy RL|Ciosek et al. "Better Exploration with Optimistic Actor-Critic" [arXiv:1910.12807](https://arxiv.org/abs/1910.12807)|
|[HER](https://github.com/HorizonRobotics/alf/blob/911d9573866df41e9e3adf6cdd94ee03016bf5a8/alf/algorithms/data_transformer.py#L672)|Off-policy RL|Andrychowicz et al. "Hindsight Experience Replay" [arXiv:1707.01495](https://arxiv.org/abs/1707.01495)|
|[TAAC](alf/algorithms/taac_algorithm.py)|Off-policy RL|Yu et al. "TAAC: Temporally Abstract Actor-Critic for Continuous Control" [arXiv:2104.06521](https://arxiv.org/abs/2104.06521)|
|[SEditor](alf/algorithms/seditor_algorithm.py)|Off-policy/Safe RL|Yu et al. "Towards Safe Reinforcement Learning with a Safety Editor Policy" [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/11afefdd848d1bc9ac9f1604d9f45817-Paper-Conference.pdf)|
|[DIAYN](alf/algorithms/diayn_algorithm.py)|Intrinsic motivation/Exploration|Eysenbach et al. "Diversity is All You Need: Learning Diverse Skills without a Reward Function" [arXiv:1802.06070](https://arxiv.org/abs/1802.06070)|
|[ICM](alf/algorithms/icm_algorithm.py)|Intrinsic motivation/Exploration|Pathak et al. "Curiosity-driven Exploration by Self-supervised Prediction" [arXiv:1705.05363](https://arxiv.org/abs/1705.05363)|
|[RND](alf/algorithms/rnd_algorithm.py)|Intrinsic motivation/Exploration|Burda et al. "Exploration by Random Network Distillation" [arXiv:1810.12894](https://arxiv.org/abs/1810.12894)|
|[MuZero](alf/algorithms/muzero_algorithm.py)|Model-based RL|Schrittwieser et al. "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" [arXiv:1911.08265](https://arxiv.org/abs/1911.08265)|
|[BC](alf/algorithms/bc_algorithm.py)|Offline RL|Pomerleau "ALVINN: An Autonomous Land Vehicle in a Neural Network" [NeurIPS 1988](https://papers.nips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html) <br>  Bain et al. "A framework for behavioural cloning" [Machine Intelligence 1999](http://www.cse.unsw.edu.au/~claude/papers/MI15.pdf)|
|[Causal BC](alf/algorithms/causal_bc_algorithm.py)|Offline RL|Swamy et al. "Causal Imitation Learning under Temporally Correlated Noise" [ICML2022](https://proceedings.mlr.press/v162/swamy22a/swamy22a.pdf)|
|[IQL](alf/algorithms/iql_algorithm.py)|Offline RL|Kostrikov, et al. "Offline Reinforcement Learning with Implicit Q-Learning" [arXiv:2110.06169](https://arxiv.org/abs/2110.06169)|
|[Flow matching](alf/algorithms/flow_matching_algorithm.py)|Imitation/General|Physical Intelligence "π0: A Vision-Language-Action Flow Model for General Robot Control" [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)|
|[MERLIN](alf/algorithms/merlin_algorithm.py)|Unsupervised learning|Wayne et al. "Unsupervised Predictive Memory in a Goal-Directed Agent"[arXiv:1803.10760](https://arxiv.org/abs/1803.10760)|
|[MoNet](alf/algorithms/monet_algorithm.py)|Unsupervised learning|Burgess et al. "MONet: Unsupervised Scene Decomposition and Representation" [arXiv:1901.11390](https://arxiv.org/abs/1901.11390)|
|[Amortized SVGD](alf/algorithms/generator.py)|General|Feng et al. "Learning to Draw Samples with Amortized Stein Variational Gradient Descent" [arXiv:1707.06626](https://arxiv.org/abs/1707.06626)|
|[HyperNetwork](alf/algorithms/hypernetwork_algorithm.py)|General|Ratzlaff and Fuxin. "HyperGAN: A Generative Model for Diverse, Performant Neural Networks" [arXiv:1901.11058](https://arxiv.org/abs/1901.11058)|
|[MCTS](alf/algorithms/mcts_algorithm.py)|General|Grill et al. "Monte-Carlo tree search as regularized policy optimization" [arXiv:2007.12509](https://arxiv.org/abs/2007.12509)|
|[MINE](alf/algorithms/mi_estimator.py)|General|Belghazi et al. "Mutual Information Neural Estimation" [arXiv:1801.04062](https://arxiv.org/abs/1801.04062)|
|[ParticleVI](alf/algorithms/particle_vi_algorithm.py)|General|Liu and Wang. "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm" [arXiv:1608.04471](https://arxiv.org/abs/1608.04471) <br> Liu et al. "Understanding and accelerating particle-based variational inference" [arXiv:1807.01750](https://arxiv.org/abs/1807.01750)|
|[GPVI](alf/algorithms/generator.py)|General|Ratzlaff, Bai et al. "Generative Particle Variational Inference via Estimation of Functional Gradients" [arXiv:2103.01291](https://arxiv.org/abs/2103.01291)|
|[SVGD optimizer](alf/optimizers/optimizers.py)|General|Liu et al. "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm" [arXiv:1608.04471](https://arxiv.org/abs/1608.04471)|
|[VAE](alf/algorithms/vae.py)|General|Higgins et al. "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" [ICLR2017](https://openreview.net/forum?id=Sy2fzU9gl)|
|[RealNVP](alf/networks/normalizing_flow_networks.py)|General|Dinh et al. "Density estimation using Real NVP" [arXiv:1605.08803](https://arxiv.org/abs/1605.08803)|
|[SpatialBroadcastDecoder](alf/networks/encoding_networks.py)|General|Watters et al. "Spatial Broadcast Decoder: A Simple Architecture for Learning Disentangled Representations in VAEs" [arXiv:1901.07017](https://arxiv.org/abs/1901.07017)|
|[VQ-VAE](alf/algorithms/vq_vae.py)|General|A van den Oord et al. "Neural Discrete Representation Learning" [NeurIPS2017](https://proceedings.neurips.cc/paper/2017/file/7a98af17e63a0ac09ce2e96d03992fbc-Paper.pdf)|

## Installation

#### OS software
The following installation was tested on Ubuntu22.04 with CUDA 11.8.

Python3.11 is currently supported by ALF. Note that some pip packages (e.g., pybullet) need python dev files, so make sure python3.11-dev is installed:

```
sudo apt install -y python3.11 python3.11-dev
```

Boost is also required by ALF for fast parallel environments.

```
sudo apt install libboost-all-dev
```

#### Python environment
[Virtualenv](https://virtualenv.pypa.io/en/latest/) is recommended for the installation. After creating and activating a virtual env, you can run the following commands to install ALF:

```
git clone https://github.com/HorizonRobotics/alf
cd alf
pip install pybind11
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118
```

#### For Nix Users

There is a built-in Nix-based development environment defined in [flake.nix](./flake.nix). To activate it, run

```bash
$ nix develop
```

in the root of your local repository.

## Docker
We also provide a docker image of ALF for convenience. In order to use this image, you need to have [docker](https://www.docker.com/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (for ALF gpu usage) installed first.

```bash
docker run --gpus all -it horizonrobotics/cuda:11.8.0-py3.11-torch2.2-ubuntu22.04 /bin/bash
```
This will give you a shell that have all ALF and dependencies pre-installed.

The current docker image contains an ALF version on Feb 21, 2024. Regular version updates are expected in the future.

## Examples

You can train any `_conf.py` file under `alf/examples` as follows:
```bash
python -m alf.bin.train --conf=CONF_FILE --root_dir=LOG_DIR
```
* CONF_FILE is the path to your conf file which follows ALF configuration file format (basically python).
* LOG_DIR is the directory when you want to store the training results. Note that if you want to train from scratch, LOG_DIR must point to a location that doesn't exist. Otherwise, it is assumed to resume the training from a previous checkpoint (if any).

During training, we use tensorboard to show the progress of training:
```bash
tensorboard --logdir=LOG_DIR
```

After training, you can evaluate the trained model and visualize environment
frames using the following command:
```bash
python -m alf.bin.play --root_dir=LOG_DIR
```

#### **Deprecated**

An older version of ALF used [gin](https://github.com/google/gin-config)
for job configuration. Its syntax is not as flexible as ALF conf (e.g., you can't easily
do math computation in a gin file). There are still some examples with `.gin`
under `alf/examples`. We are in the process of converting all `.gin` examples to `_conf.py`
examples.

You can train any `.gin` file under `alf/examples` using the following command:
```bash
cd alf/examples; python -m alf.bin.train --gin_file=GIN_FILE --root_dir=LOG_DIR
```
* GIN_FILE is the path to the gin conf (some `.gin` files under `alf/examples` might be invalid; they have not been converted to use the latest pytorch version of ALF).
* LOG_DIR has the same meaning as in the ALF conf example above.

*Warning*: When using **gin**, ALF has to be launched in the same directory with the gin file(s). If an error says that no configuration file is found, then probably you've launched ALF in a wrong place.

All the examples below are trained on a single machine Intel(R) Core(TM) i9-7960X CPU @ 2.80GHz with 32 CPUs and one RTX 2080Ti GPU.

### A2C
* [Cart pole](alf/examples/ac_cart_pole.gin). The training score took only 30 seconds to reach 200, using 8 environments.

  <img src="alf/examples/ac_cart_pole.png" width="300" height="200" alt="breakout-training-curve"/> <img src="alf/examples/ac_cart_pole.gif" height="200" alt="cartpole-video"/>

* [Atari games](alf/examples/ac_breakout_conf.py). Need to install python package atari-py for atari game environments. The evaluation score (by taking argmax of the policy) took 1.5 hours to reach 800 on Breakout, using 64 environments.

  <img src="alf/examples/ac_breakout.png" width="300" height="200" alt="breakout-training-curve"/> <img src="alf/examples/ac_breakout.gif" width="150" height="200" alt="breakout-playing-screen"/>

* [Simple navigation with visual input](alf/examples/ac_simple_navigation.gin). Follow the instruction at [SocialRobot](https://github.com/HorizonRobotics/SocialRobot) to install the environment.

  <img src="alf/examples/ac_simple_navigation.png" width="300" height="200" alt="simple-navigation-curve"/> <img src="alf/examples/ac_simple_navigation.gif" height="200" alt="simple0navigation-video"/>

### PPO
* [PR2 grasping state only](alf/examples/ppo_pr2.gin). Follow the instruction at [SocialRobot](https://github.com/HorizonRobotics/SocialRobot) to install the environment.

  <img src="alf/examples/ppo_pr2.png" width="300" height="200" alt="ppo-pr2-curve"/> <img src="alf/examples/ppo_pr2.gif" height="200" alt="pr2-video"/>


* [Humanoid](alf/examples/ppo_bullet_humanoid.gin). Learning to walk using the pybullet Humanoid environment. Need to install python pybullet>=2.5.0 for the environment. The evaluation score reaches 3k in 50M steps, using 96 parallel environments.

  <img src="alf/examples/ppo_bullet_humanoid.png" width = "300" height ="200" alt="Humanoid-training-curve"/> <img src="alf/examples/ppo_bullet_humanoid.gif" width = "300" height ="200" alt="Humanoid-video"/>

### PPG

* [procgen](alf/examples/ppg_procgen_bossfight_conf.py). Game "bossfight" as an example. Need to install python package [procgen](https://github.com/openai/procgen).

  <img src="alf/examples/ppg_procgen_bossfight.png" width="300" height="200" alt="ppg-bossfight-curve"/> <img src="alf/examples/ppg_bossfight.gif" height="200" alt="ppg-bossfight-video"/>

* [MetaDrive](alf/examples/metadrive/ppg_metadrive_conf.py). Learning to drive on randomly generated map with interaction on the [MetaDrive](https://metadriverse.github.io/metadrive/) simulator, with BEV as input. Need to install python package [metadrive-simulator](https://github.com/metadriverse/metadrive).

  <img src="alf/examples/metadrive/ppg_metadrive.jpg" width="300" height="200" alt="ppg_metadrive-curve"/> <img src="alf/examples/metadrive/ppg_metadrive.gif" height="200" alt="ppg-metadrive-video"/>

### DDQN
* [DDQN on Atari](alf/examples/dqn_breakout_conf.py).  Game "Q*Bert" performance.

  <img src="alf/examples/dqn_breakout_conf_Qbert.png" width = "600" height ="200" alt="Q*Bert-DDQN-training-curve"/>

### DDPG
* [FetchSlide (sparse rewards)](alf/examples/ddpg_fetchslide_conf.py). Need to install the [MuJoCo](https://www.roboti.us/index.html) simulator first. This example reproduces the performance of vanilla DDPG reported in the OpenAI's Robotics environment [paper](https://arxiv.org/pdf/1802.09464.pdf). Our implementation doesn't use MPI, but obtains (evaluation) performance on par with the original implementation. (*The original MPI implementation has 19 workers, each worker containing 2 environments for rollout and sampling a minibatch of size 256 from its replay buffer for computing gradients. All the workers' gradients will be summed together for a centralized optimizer step. Our implementation simply samples a minibatch of size 5000 from a common replay buffer per optimizer step.*) The training took about 1 hour with 38 (19*2) parallel environments on a single GPU.

  <img src="alf/examples/ddpg_fetchslide.png" width="300" height="200" alt="ddpg-fetchslide-training-curve"/> <img src="alf/examples/ddpg_fetchslide.gif" width="300" height="200" alf="ddpg-fetchslide-video"/>

### SAC
* [Bipedal Walker](alf/examples/sac_bipedal_walker_conf.py).

  <img src="alf/examples/sac_bipedal_walker.png" width = "300" height ="200" alt="bipedal-walker-training-curve"/> <img src="alf/examples/sac_bipedal_walker.gif" width = "300" height ="200" alt="bipedal-walker-video"/>

* [FetchReach (sparse rewards)](alf/examples/sac_fetchreach.gin). Need to install the [MuJoCo](https://www.roboti.us/index.html) simulator first. The training took about 20 minutes with 20 parallel environments on a single GPU.

  <img src="alf/examples/sac_fetchreach.png" width="300" height="200" alt="sac-fetchreach-training-curve"/> <img src="alf/examples/sac_fetchreach.gif" width="300" height="200" alf="sac-fetchreach-video"/>

* [FetchSlide (sparse rewards)](alf/examples/sac_fetchslide.gin). Need to install the [MuJoCo](https://www.roboti.us/index.html) simulator first. This is the same task with the DDPG example above, but with SAC as the learning algorithm.
Also it has only 20 (instead of 38) parallel environments to improve sample efficiency. The training took about 2 hours on a single GPU.

  <img src="alf/examples/sac_fetchslide.png" width="300" height="200" alt="sac-fetchslide-training-curve"/> <img src="alf/examples/sac_fetchslide.gif" width="300" height="200" alf="sac-fetchslide-video"/>

* [Fetch Environments (sparse rewards) w/ Action Repeat](alf/examples/sac_actrepeat_fetch/sac_actrepeat_fetch.gin). We are able to achieve even better performance than reported by DDPG + Hindsight Experience Replay in some cases simply by using SAC + Action Repeat with length 3 timesteps. See [this note](docs/notes/fetch_env_sac_actrepeat_notes.rst) to view learning curves, videos, and more details.

### ICM
* [Super Mario](alf/examples/icm_super_mario_intrinsic_only.gin). Playing Super Mario only using intrinsic reward.
  Python package gym-retro>=0.7.0 is required for this experiment and also a suitable `SuperMarioBros-Nes` rom should be obtained and imported (roms are not included in gym-retro). See [this doc](https://retro.readthedocs.io/en/latest/getting_started.html#importing-roms) on how to import roms.

  <img src="alf/examples/icm_super_mario_intrinsic_only.png" width = "300" height ="200" alt="super-mario-training-curve"/> <img src="alf/examples/icm_super_mario_intrinsic_only.gif" width = "300" height ="200" alt="super-mario-video"/>

### RND
* [Montezuma's Revenge](alf/examples/ppo_rnd_mrevenge.gin). Training the hard exploration game Montezuma's Revenge with intrinsic rewards generated by [RND](alf/algorithms/rnd_algorithm.py). A lucky agent can get an episodic score of 6600 in 160M frames (40M steps with `frame_skip=4`). A normal agent would get an episodic score of 4000~6000 in the same number of frames. The training took about 6.5 hours with 128 parallel environments on a single GPU.

<img src="alf/examples/rnd_mrevenge.png" width="300" height="200" alt="mrevenge-training-curve"><img src="alf/examples/rnd_mrevenge.gif" width="150" height="200" alt="mrevenge-video">

### DIAYN
* [Pendulum](alf/examples/diayn_pendulum.gin). Learning diverse skills without external rewards.

  <img src="alf/examples/diayn_pendulum.png" width="300" alt="Discriminator loss">
  <img src="alf/examples/diayn_pendulum.gif" width = "600" alt="Skills learned with DIAYN"/>

### BC
* [Pendulum](alf/examples/hybrid_rl/bc_pendulum_conf.py). Learning a control policy from [offline demonstrations](alf/examples/hybrid_rl/replay_buffer_data/pendulum_replay_buffer_from_sac_10k).

  <img src="alf/examples/hybrid_rl/bc_pendulum_return.png" width="300" height ="200" alt="Loss">
  <img src="alf/examples/hybrid_rl/bc_pendulum_loss.png" width="300" height ="200" alt="Return">



### Merlin
* [Collect Good Objects](alf/examples/merlin_dmlab_collect_good_objects.gin). Learn to collect good objects and avoid bad objects.
  `DeepmindLab` is required,  Follow the instruction at [DeepmindLab](https://github.com/deepmind/lab/blob/master/python/pip_package/README.md) to install the environment.

  <img src="alf/examples/merlin_dmlab_collect_good_objects.png" width = "300" height ="200" alt="room-collect-good-objects-training-curve"/> <img src="alf/examples/merlin_dmlab_collect_good_objects.gif" width = "300" height ="200" alt="room-collect-good-objects"/>


### MuZero
* [6x6 Go](alf/examples/muzero_go_6x6.gin). It took about a day to train a reasonable agent to play 6x6 go using one GPU.

  <img src="alf/examples/muzero_go_6x6.gif" width = "200" height ="200" alt="6x6-go"/>


## Citation
If you use ALF for research and find it useful, please consider citing:
```
@software{Xu2021ALF,
  title={{{ALF}: Agent Learning Framework}},
  author={Xu, Wei and Yu, Haonan and Zhang, Haichao and Hong, Yingxiang and Yang, Break and Zhao, Le and Bai, Jerry and ALF contributors},
  url={https://github.com/HorizonRobotics/alf},
  year={2021}
}
```


## Contribute to ALF

You are welcome to contribute to ALF. Please follow the guideline [here](https://alf.readthedocs.io/en/latest/contributing.html).
