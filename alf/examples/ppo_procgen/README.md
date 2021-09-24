# Procgen Environment with PPO

This directory provides configuration to train PPO algorithm on [procgen](https://openai.com/blog/procgen-benchmark/) benchmark environments.

## Run with command line argument (Recommended)

Procgen benchmark provide 16 different games (environments). Alf supports all 16 of them. The main configuration parameters are in [base_conf.py](./base_conf.py). To run PPO Procgen training for a specific game such as `coinrun`, you can invoke a training using [base_conf.py](./base_conf.py) as the configuration while supplying the `env_name` parameter for `create_environment`:

```bash
python -m alf.bin.train \
    --conf alf/examples/ppo_procgen/base_conf.py \ # assuming invoked from alf root
    --conf_param create_environment.env_name='coinrun' \
    --root_dir <your/root/dir>
```

## Run with a customized configuration file

Sometimes you may want to create a configuration file for a specific game and run without having to specify the `env_name` every time. Here we have one configurations for `bossfight` as an example:

* bossfight: [bossfight_conf.py](./bossfight_conf.py)

It is merely a thin wrapper over [base_conf.py](./base_conf.py) to provide `env_name`. For example, to create a configuration for `coinrun`, the following 3-line configuration will suffice.

```python

import alf
import alf.examples.ppo_procgen.base_conf

alf.config('create_environment', env_name='coinrun')
```
