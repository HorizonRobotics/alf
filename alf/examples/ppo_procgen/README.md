# Procgen Environment with PPO

This directory provides configuration to train PPO algorithm on [procgen](https://openai.com/blog/procgen-benchmark/) benchmark environments.

Procgen benchmark provide 16 different games (environments). Alf support all 16 of them and here we have configurations for 3 of them as examples:

1. bossfight: [bossfight_conf.py](./bossfight_conf.py)
2. bigfish: [bigfish_conf.py](./bigfish_conf.py)
3. chaser: [chaser_conf.py](./chaser_conf.py)

The main configuration parameters are in [base_conf.py](./base_conf.py). To create a configuration for a different game in Procgen, you just need to have 3 lines and fill `env_name` with the game of choice. For example, to create a configuration for `coinrun`, the following configuration will suffice.

```python

import alf
import alf.examples.ppo_procgen.base_conf

alf.config('create_environment', env_name='coinrun')
```
