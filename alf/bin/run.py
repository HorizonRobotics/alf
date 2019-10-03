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
r"""Run training and playing in a simple one liner command.
Allows running on clusters or locally.

To run actor_critic on gym CartPole:
```bash
cd ${PROJECT}/alf/examples;
python -m alf.bin.run \
  actargetnav-randgoal_False-goalname_ball_car^wheel-env_30-image_84_84-unroll_100-msteps_100-mepisteps_1000-faildist_5-rnn_True-lstm_128_128 \  # first parameter is the run tag, which will be parsed
  --cluster=None \
  --path=~/tmp \  # path to store output logs
  --play
```

Use '=' to replace '-', and '^' to replace '_' in argument values.

Tips:

Use alias run='python -m alf.bin.run' to make command even shorter.

Outputs statistics from the run.

"""

import os
import re
import sys

from absl import app
from absl import flags
from absl import logging

flags.DEFINE_string("path", os.path.join(os.path.expanduser("~"), "tmp"),
                    "Root directory for all runs and all root_dir's.")
flags.DEFINE_string('cluster', None, 'idc, us or None for local.')
flags.DEFINE_bool('play', False, 'Whether to play the model.')

FLAGS = flags.FLAGS

ARG_SEP = "-"
VALUE_SEP = "_"


# Utils:
def get_env_load_fn(gin_file):
    #create_environment.env_load_fn=@suite_socialbot.load
    with open(gin_file, 'r') as file:
        data = file.read().replace('\n', ' ')
        m = re.search('create_environment\.env_load_fn\=\@(\S+)', data)
        if m:
            return m.group(1)
    return "NO_ENV_LOAD_FN"


def gen_args(named_tags, task="NO_TASK", load_fn="NO_LOAD_FN"):
    args = []
    for name, value in named_tags.items():
        arg_name_or_list = get_arg_name(name, task=task, load_fn=load_fn)
        if arg_name_or_list:
            arg_names = []
            if isinstance(arg_name_or_list, str):
                arg_names = [arg_name_or_list]
            for arg_name in arg_names:
                args.append("--gin_param='{}={}'".format(arg_name, value))
    return ' '.join(args)


# Mappings:
def get_gin_file(gin_file_tag):
    switcher = {
        "actargetnav": "ac_target_navigation",
        "acbreakout": "ac_break_out",
        "accartpole": "ac_cart_pole",
    }
    return switcher.get(gin_file_tag,
                        "no_such_gin_file_tag_" + gin_file_tag) + ".gin"


def get_task_class(gin_file_tag):
    switcher = {
        "actargetnav": "GroceryGroundGoalTask",
        "ppotargetnav": "GroceryGroundGoalTask",
    }
    return switcher.get(gin_file_tag, "NO_TASK")


def get_parent_task(task):
    switcher = {
        "GroceryGroundGoalTask": "GroceryGround",
    }
    return switcher.get(task, "NO_PARENT_TASK")


def get_arg_name(name, task="NO_TASK", load_fn="NO_LOAD_FN"):
    switcher = {
        'env':
            'create_environment.num_parallel_environments',
        'vtrace':
            'ActorCriticLoss.use_vtrace',
        'msteps':
            task + '.max_steps',
        'mepisteps':
            load_fn + '.load.max_episode_steps',
        'randgoal':
            task + '.random_goal',
        'goalname':
            task + '.goal_name',
        'randrange':
            task + '.random_range',
        'faildist':
            task + '.fail_distance_thresh',
        'image':
            get_parent_task(task) + '.resized_image_size',
        'rnn':
            'create_ac_algorithm.use_rnns',
        'lstm': [
            'ActorDistributionRnnNetwork.lstm_size',
            'ValueRnnNetwork.lstm_size'
        ],
        # curriculum learning:
        'startr':
            task + '.start_range',
        'mixfull':
            task + '.percent_full_range_in_curriculum',
        'qlen':
            task + '.max_reward_q_length',
        'rewthd':
            task + '.reward_thresh_to_increase_range',
        'incrange':
            task + '.increase_range_by_percent',
    }
    return switcher.get(name, None)


def map_list_value(value):
    values = value.split(VALUE_SEP)
    translated = [
        v.replace('=', ARG_SEP).replace('^', VALUE_SEP) for v in values
    ]
    return ','.join(values)


def brace_map_list_value(value):
    return '(' + map_list_value(value) + ')'


def maybe_map_value(key, value=None):
    switcher = {
        'image': brace_map_list_value(value),  # 84_84 => (84,84)
        'lstm': brace_map_list_value(value),
        'goalname': map_list_value(value),  # ball_car^wheel => ball,car_wheel
    }
    return switcher.get(key, value)


def main(argv):
    # parse tag:
    run_tag = argv[1]
    tags = run_tag.split(ARG_SEP)
    named_tags = {}
    i = 0
    for tag in tags:
        if i == 0:
            gin_file_tag = tag
        i += 1
        kv = tag.split(VALUE_SEP, 1)
        if len(kv) == 2:
            named_tags[kv[0]] = maybe_map_value(kv[0], value=kv[1])
        else:
            named_tags[tag] = "True"

    if "gin" in named_tags:
        gin_file_tag = named_tags["gin"]
    gin_file = get_gin_file(gin_file_tag)

    task = get_task_class(gin_file_tag)
    train = "play" if FLAGS.play else "train"
    root_dir = FLAGS.path + "/" + run_tag
    log_file = root_dir + "/log-{}.txt".format(train)

    load_fn = get_env_load_fn(gin_file)

    args = gen_args(named_tags, task, load_fn)

    if os.path.exists("../bin/train.py"):
        module = train
        additional_flag = ""
    else:
        module = "main"
        additional_flag = " --play" if FLAGS.play else ""
    command = ((
        'python -m alf.bin.{module} --gin_file={gin_file} --root_dir={root_dir} '
        + args + additional_flag + ' 2> {log_file}').format(
            module=module,
            gin_file=gin_file,
            root_dir=root_dir,
            log_file=log_file))

    print(command)
    if FLAGS.cluster:
        pass
    else:
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        os.system(command)


if __name__ == '__main__':
    app.run(main)
