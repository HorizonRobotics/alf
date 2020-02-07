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

To run target navigation:
```bash
cd ${PROJECT}/alf/examples;
python -m alf.bin.run \
  actargetnav-randgoal_True-goalname_ball_car__wheel-hitdist_0.25-hitpenal_0.5-env_30-eval_1-image_84_84-internalst_1-unroll_100-msteps_100-mepisteps_1000-faildist_5-rnn_True-lstm_128_128 \  # first parameter is the run tag, which will be parsed
  --cluster=None \
  --path=~/tmp \  # path to store output logs
  --play
```

Use '--' to replace '-', and '__' to replace '_' in argument values.

Tips:

Use alias run='python -m alf.bin.run' to make command even shorter.

Outputs statistics from the run.

There are 3 typical running scenarios for training:
1. just running locally and dumping all stderr and stdout to a log file.
2. running on gpu-dev007 machine and submitting a job to gpu cluster, use --cluster=idc-rl parameter.
3. managed by run.yaml and running via run.sh in a pod on the cluster, use --log_to_file=False so
all stdout and stderr can be captured and dumped to the right output directory.

"""

from collections import OrderedDict
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
flags.DEFINE_string('mp4_file', '', 'Output to mp4 file')
flags.DEFINE_integer('ckpt', 0, 'checkpoint file number')
flags.DEFINE_float('sleep', 0, 'number of seconds to sleep between steps')
flags.DEFINE_bool('log_to_file', None, 'Whether to log to file or stdout.')
flags.DEFINE_string('network_to_debug', '',
                    'actor or value network to plot attention and debug')
flags.DEFINE_bool('tffn', True, 'Whether to use graph mode / tf functions.')

FLAGS = flags.FLAGS

ARG_SEP = "-"
VALUE_SEP = "_"

# Gin configs:

TARGET_NAV_GIN_TAGS = [
    'actargetnav', 'actargetnavattn', 'ppotargetnav', 'ddpgtargetnav',
    'sactargetnav'
]
GYM_GIN_TAGS = [
    'acbreakout', 'offpolicyacbreakout', 'acbreakoutvtrace', 'accartpole'
]


def get_gin_file(gin_file_tag):
    switcher = {
        "actargetnav": "ac_target_navigation",
        "actargetnavattn": "ac_target_navigation_attn",
        "ppotargetnav": "ppo_target_navigation",
        "ddpgtargetnav": "ddpg_target_navigation",
        "sactargetnav": "sac_target_navigation",
        "acbreakout": "ac_breakout",
        "offpolicyacbreakout": "off_policy_ac_breakout",
        "acbreakoutvtrace": "off_policy_ac_breakout_vtrace",
        "accartpole": "ac_cart_pole",
        "dmexploregoal": "off_policy_ac_dm_explore_goal_vtrace"
    }
    return switcher.get(gin_file_tag,
                        "no_such_gin_file_tag_" + gin_file_tag) + ".gin"


TARGET_NAV_GIN_FILES = [get_gin_file(t) for t in TARGET_NAV_GIN_TAGS]
GYM_GIN_FILES = [get_gin_file(t) for t in GYM_GIN_TAGS]


# Utils:
def get_env_load_fn(gin_file):
    #create_environment.env_load_fn=@suite_socialbot.load
    with open(gin_file, 'r') as file:
        data = file.read().replace('\n', ' ')
        m = re.search('create_environment\.env_load_fn\=\@(\S+)', data)
        if m:
            return m.group(1)
        elif gin_file in TARGET_NAV_GIN_FILES:
            return "suite_socialbot.load"
        elif gin_file in GYM_GIN_FILES:
            return "suite_gym.load"
    assert False, "NO ENV LOAD FN in {}.".format(gin_file)


def create_run_yaml(tag):
    out_fn = None
    with open('run.yaml', 'r') as f_in:
        data = f_in.read()
        out_data = data.replace('RUNTAG', tag)
        out_fn = 'run-{}.yaml'.format(tag)
        with open(out_fn, 'w') as f_out:
            f_out.write(out_data)
    assert out_fn, 'Cannot write out job yaml based on run.yaml'
    return out_fn


def gen_args(named_tags, task="NO_TASK", load_fn="NO_LOAD_FN", gin_file=""):
    args = []
    remaining_args = []
    m_arg_value = {}
    for name, value in named_tags.items():
        arg_name_or_list = get_arg_name(name, task=task, load_fn=load_fn)
        if arg_name_or_list:
            arg_names = arg_name_or_list
            if isinstance(arg_name_or_list, str):
                arg_names = [arg_name_or_list]
            for arg_name in arg_names:
                args.append("--gin_param='{}={}'".format(arg_name, value))
            m_arg_value[name] = value
        elif name not in ['rnn']:
            remaining_args.append(name)
    if 'image' not in m_arg_value and gin_file in TARGET_NAV_GIN_FILES:
        args.append("--gin_param='image_scale_transformer.fields=[]'")
        args.append("--gin_param='Agent.observation_transformer=[]'")
    if FLAGS.network_to_debug:
        args.append(
            "--gin_param='get_ac_networks.network_to_debug=\"{}\"'".format(
                FLAGS.network_to_debug))
        args.append(
            "--use_tf_functions=False  --gin_param='TrainerConfig.use_tf_functions=False'"
        )
    if 'entmax' in m_arg_value:
        args.append("--gin_param='Agent.enforce_entropy_target=True'")
        args.append(
            "--gin_param='ActorCriticLoss.entropy_regularization=None'")

    print('\n'.join(args))
    print("========================================")
    for name in remaining_args:
        print('* Warning: parameter name {} not understood by parser.'.format(
            name))

    return ' '.join(args), m_arg_value


# Mappings:


def get_task_class(gin_file_tag):
    res = "NO_TASK"
    if "targetnav" in gin_file_tag:
        res = "GoalTask"
    return res


def get_parent_task(task):
    switcher = {
        "GoalTask": "PlayGround",
    }
    return switcher.get(task, "NO_PARENT_TASK")


def get_arg_name(name, task="NO_TASK", load_fn="NO_LOAD_FN"):
    parent_task = get_parent_task(task)
    switcher = {
        'attnmax':
            'get_ac_networks.output_attention_max',
        'attnresid':
            'get_ac_networks.use_attn_residual',
        'env':
            'create_environment.num_parallel_environments',
        'eval':
            'TrainerConfig.evaluate',
        'lr':
            'ac/Adam.learning_rate',
        'tffn':
            'TrainerConfig.use_tf_functions',
        'unroll':
            'TrainerConfig.unroll_length',
        'td':
            'ActorCriticLoss.td_lambda',
        'ent':
            'ActorCriticLoss.entropy_regularization',
        'vtrace':
            'ActorCriticLoss.use_vtrace',
        'tilestsen': [
            'get_value_network.num_state_tiles',
            'get_value_network.num_sentence_tiles',
            'get_actor_network.num_state_tiles',
            'get_actor_network.num_sentence_tiles',
        ],
        'msteps': [task + '.max_steps', parent_task + '.max_steps'],
        'stept':
            parent_task + '.step_time',
        'mepisteps':
            load_fn + '.max_episode_steps',
        'endsucc':
            task + '.end_on_success',
        'endhit':
            task + '.end_on_hitting_distraction',
        'gymwrap':
            load_fn + '.gym_env_wrappers',
        'randgoal':
            task + '.random_goal',
        'goalname':
            task + '.goal_name',
        'distract':
            task + '.distraction_list',
        'curridistr':
            task + '.curriculum_distractions',
        'curriangl':
            task + '.curriculum_target_angle',
        'episwgoal':
            task + '.switch_goal_within_episode',
        'randrange':
            task + '.random_range',
        'faildist':
            task + '.fail_distance_thresh',
        'hitdist':
            task + '.distraction_penalty_distance_thresh',
        'hitpenal':
            task + '.distraction_penalty',
        'world':
            parent_task + '.world_name',
        'imgobs':
            parent_task + '.use_image_observation',
        'lang':
            parent_task + '.with_language',
        'image':
            parent_task + '.resized_image_size',
        'prevact':
            task +
            '.with_prev_action',  # This only works for simple state observation.
        # need to set GazeboAgent.with_prev_action to enable prev_action with image observation.
        'pool': [
            'get_value_network.pooling',
            'get_actor_network.pooling',
        ],
        'internalst':
            parent_task + '.image_with_internal_states',
        'simplefullst':
            task + '.use_egocentric_states',
        'anglelimit': [
            task + '.egocentric_perception_range',
            'get_ac_networks.angle_limit'
        ],
        'orderobjbyview': [
            task + '.order_obj_by_view', 'get_ac_networks.has_obj_id'
        ],
        'rnn':
            'get_ac_networks.rnn',
        'lstm': [
            'ActorDistributionRnnNetwork.lstm_size',
            'ValueRnnNetwork.lstm_size'
        ],
        'heads':
            'get_ac_networks.n_heads',
        'entmax':
            'EntropyTargetAlgorithm.max_entropy',
        'initalpha':
            'EntropyTargetAlgorithm.initial_alpha',
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
        'finstepmode':
            'OnPolicyDriver.final_step_mode',
    }
    return switcher.get(name, None)


def tokenize(v, quoted=False):
    v = v.replace('^', VALUE_SEP)
    v = v.replace('+', VALUE_SEP)
    v = v.replace('=', ARG_SEP)
    if quoted:
        v = '"' + v + '"'
    return v


def map_list_value(value, quoted=False):
    values = value.split(VALUE_SEP)

    translated = [tokenize(v, quoted=quoted) for v in values]
    return ','.join(translated)


def brace_map_list_value(value):
    return '(' + map_list_value(value) + ')'


def maybe_map_value(key, value=None):
    switcher = {
        'image': brace_map_list_value(value),  # 84_84 => (84,84)
        'lstm': brace_map_list_value(value),
        'goalname': map_list_value(value, quoted=True),
        'distract':
            '[' + map_list_value(value, quoted=True) +
            ']',  # ball_car__wheel => ball_car+wheel => [ball,car_wheel]
        'world': tokenize(value, quoted=True)
    }
    return switcher.get(key, map_list_value(value))


def v(map, key, default=None):
    if key in map and map[key] != "0":
        return map[key]
    else:
        return default


def main(argv):
    # parse tag:
    run_tag = argv[1]
    # externally, -- means literal -, and __ is literal _, but
    # internally we use = to represent literal -, and + for literal _
    _run_tag = run_tag.replace(ARG_SEP + ARG_SEP, '=')
    _run_tag = _run_tag.replace(VALUE_SEP + VALUE_SEP, '+')
    tags = _run_tag.split(ARG_SEP)
    gin_file_tag = tags[0]
    tags = tags[1:]
    named_tags = OrderedDict()
    for tag in tags:
        kv = tag.split(VALUE_SEP, 1)  # try split once
        if len(kv) == 2:
            named_tags[kv[0]] = maybe_map_value(kv[0], value=kv[1])
        else:
            named_tags[tag] = "True"

    if "gin" in named_tags:
        gin_file_tag = named_tags["gin"]
    gin_file = get_gin_file(gin_file_tag)

    task = get_task_class(gin_file_tag)
    parent_task = get_parent_task(task)
    train = "play" if FLAGS.play else "train"
    root_dir = FLAGS.path + "/" + run_tag
    log_file = root_dir + "/{}.log".format(train)

    load_fn = get_env_load_fn(gin_file)

    args, m_arg_value = gen_args(named_tags, task, load_fn, gin_file)

    if os.path.exists("../bin/train.py"):
        module = train
        additional_flag = ""
    else:
        module = "main"
        additional_flag = " --play" if FLAGS.play else ""

    if FLAGS.log_to_file is None:
        if FLAGS.play:
            log_to_file = False
        else:
            log_to_file = True
    else:
        log_to_file = FLAGS.log_to_file

    if log_to_file:
        log_cmd = " 2> {log_file}".format(log_file=log_file)
    else:
        log_cmd = ""
    if FLAGS.play:
        # additional_flag += " --epsilon_greedy=0.1"
        if not FLAGS.mp4_file:
            additional_flag += " --num_episodes=100"
        if FLAGS.mp4_file:
            additional_flag += " --record_file='{}'".format(FLAGS.mp4_file)
        if FLAGS.ckpt:
            additional_flag += " --checkpoint_name=ckpt-{}".format(FLAGS.ckpt)
        if gin_file_tag in TARGET_NAV_GIN_TAGS:
            additional_flag += " --gin_param='{}.start_range=1'".format(task)
            additional_flag += " --gin_param='{}.max_steps=100'".format(
                parent_task)
            additional_flag += " --gin_param='suite_socialbot.load.max_episode_steps=1000'"
            additional_flag += " --verbosity=1"
        if FLAGS.sleep:
            additional_flag += " --sleep_time_per_step={}".format(FLAGS.sleep)
        if ('tffn' in m_arg_value
                and m_arg_value['tffn'] in ['False', '0']) or not FLAGS.tffn:
            additional_flag += " --use_tf_functions=False  --gin_param='TrainerConfig.use_tf_functions=False'"
    if gin_file_tag in TARGET_NAV_GIN_TAGS:
        additional_flag += " --gin_param='{}.polar_coord=False'".format(task)

    command = (
        "python3 ../bin/{module}.py --gin_file={gin_file} --root_dir={root_dir} "
        + args + additional_flag + log_cmd).format(
            module=module, gin_file=gin_file, root_dir=root_dir)

    if FLAGS.cluster:
        os.chdir('/home/users/le.zhao/jobs/gail')
        run_yaml = create_run_yaml(run_tag)
        print("========================================")
        print('Submitting:\n' + command)
        command = 'traincli submit -f {}'.format(run_yaml)
        print('via:\n' + command)
        ret = os.system(command)
        print("========================================")
        feedback = "Failed" if ret else "Succeeded"
        print(feedback)
    else:
        print(command)
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        ret = os.system(command)
        if ret and log_to_file:
            print("========================================")
            os.system('tail -n 100 {}'.format(log_file))


if __name__ == '__main__':
    app.run(main)
