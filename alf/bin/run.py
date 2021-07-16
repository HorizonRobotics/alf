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

# Play deterministically:
run hertargetnavst-curri_0-goalgen_@SubgoalPlanningGoalGeneratorLPRP-auxach_1-xyaux-auxdim_4-sgmargin_0.2-msgsteps_2-goalgensrc_exp-herorigg_0.5-herk_0.6-numsg_1-cempopu_1000-cemiters_100-gmincost_10-hitpenal_100-iters_2000 \
  --play --seed=17 --ckpt=1000 --episodes=1
```

Use '--' to replace '-', and '__' to replace '_' in argument values.

Tips:

Use alias run='python -m alf.bin.run' to make command even shorter.

Outputs statistics from the run.

There are 3 typical running scenarios for training:
1. just running locally and dumping all stderr and stdout to a log file.
2. running on gpu-dev007 machine and submitting a job to gpu cluster, use --cluster=idc-rl parameter.
3. managed by run.yaml and running via run.sh in a pod on the cluster, use --log=0 so
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
flags.DEFINE_string(
    "root_dir", "", "Root directory for one run.  Overrides the default path.")
flags.DEFINE_string('cluster', None, 'idc, us or None for local.')
flags.DEFINE_bool('play', False, 'Whether to play the model.')
flags.DEFINE_integer('episodes', 0, 'Number of episodes to play.')
flags.DEFINE_string('mp4_file', '', 'Output to mp4 file')
flags.DEFINE_integer('ckpt', 0, 'checkpoint file number')
flags.DEFINE_float('sleep', 0, 'number of seconds to sleep between steps')

flags.DEFINE_integer('seed', 0, 'random seed')
flags.DEFINE_bool('log', None, 'Whether to log to file, otherwise to screen.')
flags.DEFINE_bool('nolog', False, 'Whether to not log to file.')
flags.DEFINE_string('gin', '', 'extra gin params.')
flags.DEFINE_string('network_to_debug', '',
                    'actor or value network to plot attention and debug')
flags.DEFINE_bool('tffn', True, 'Whether to use graph mode / tf functions.')

FLAGS = flags.FLAGS

ARG_SEP = "-"
VALUE_SEP = "_"

# Gin configs:

TARGET_NAV_GIN_TAGS = [
    'actargetnav', 'actargetnavst', 'actargetnavattn', 'ppotargetnav',
    'ddpgtargetnav', 'sactargetnav', 'ddpgpushst', 'herpushst', 'herpushstg'
]

SOCIAL_BOT_GIN_TAGS = TARGET_NAV_GIN_TAGS + [
    'offsimplenav', 'simplenav', 'ddpgtargetnavst', 'ddpgtargetspnavst',
    'hertargetnavst', 'sactargetnavst', "hersactargetnavst",
    "hertargetnavstgoalgen", "htgstg", "hertargetspnavst",
    "hertargetspnavstgoalgen", 'ddpgpushst', 'herpushst', 'herpushstg'
]

GYM_GIN_TAGS = [
    'acbreakout',
    'offpolicyacbreakout',
    'acbreakoutvtrace',
    'accartpole',
]

MJC_GIN_TAGS = [
    "sacfetchreach",
    "sacfetchpush",
    "sacfetchslide",
    "ddpgfetchreach",
    "ddpgfetchpush",
    "herfetchpush",
    "ddpgfetchslide",
    "herfetchslide",
    "ddpgfetchpickplace",
    "herfetchpickplace",
]


def get_gin_file(gin_file_tag):
    switcher = {
        "actargetnav":
            "ac_target_navigation",
        "actargetnavst":
            "ac_target_navigation_states",
        "actargetnavattn":
            "ac_target_navigation_attn",
        "ppotargetnav":
            "ppo_target_navigation",
        "ddpgpushst":
            'ddpg_push_states',
        "ddpgtargetnav":
            "ddpg_target_navigation",
        "ddpgtargetnavst":
            "ddpg_target_navigation_states",
        "ddpgtargetspnavst":
            "ddpg_target_speed_navigation_states",
        "hertargetnavst":
            "her_target_navigation_states",
        "herpushst":
            "her_push_states",
        'herpushstg':
            "her_push_states_goalgen",
        "htgstg":
            "her_target_navigation_states_goalgen",
        "hertargetnavstgoalgen":
            "her_target_navigation_states_goalgen",
        "hertargetspnavst":
            "her_target_speed_navigation_states",
        "hertargetspnavstgoalgen":
            "her_target_speed_navigation_states_goalgen",
        "sactargetnav":
            "sac_target_navigation",
        "sactargetnavst":
            "sac_target_navigation_states",
        "hersactargetnavst":
            "her_sac_target_navigation_states",
        "acbreakout":
            "ac_breakout",
        "offpolicyacbreakout":
            "off_policy_ac_breakout",
        "offsimplenav":
            "off_policy_ac_simple_navigation",
        "simplenav":
            "ac_simple_navigation",
        "acbreakoutvtrace":
            "off_policy_ac_breakout_vtrace",
        "accartpole":
            "ac_cart_pole",
        "dmexploregoal":
            "off_policy_ac_dm_explore_goal_vtrace",
        "sacfetchreach":
            "sac_fetchreach",
        "sacfetchpush":
            "sac_fetchpush",
        "sacfetchslide":
            "sac_fetchslide",
        "ddpgfetchreach":
            "ddpg_fetchreach",
        "ddpgfetchpush":
            "ddpg_fetchpush",
        "herfetchpush":
            "her_fetchpush",
        "ddpgfetchslide":
            "ddpg_fetchslide",
        "herfetchslide":
            "her_fetchslide",
        "ddpgfetchpickplace":
            "ddpg_fetchpickplace",
        "herfetchpickplace":
            "her_fetchpickplace",
        "mbpendu":
            "mbrl_pendulum",
    }
    return switcher.get(gin_file_tag,
                        "no_such_gin_file_tag_" + gin_file_tag) + ".gin"


TARGET_NAV_GIN_FILES = [get_gin_file(t) for t in TARGET_NAV_GIN_TAGS]
SOCIAL_BOT_GIN_FILES = [get_gin_file(t) for t in SOCIAL_BOT_GIN_TAGS]
GYM_GIN_FILES = [get_gin_file(t) for t in GYM_GIN_TAGS]
MJC_GIN_FILES = [get_gin_file(t) for t in MJC_GIN_TAGS]


# Utils:
def get_env_load_fn(gin_file):
    #create_environment.env_load_fn=@suite_socialbot.load
    with open(gin_file, 'r') as file:
        data = file.read().replace('\n', ' ')
        m = re.search('create_environment\.env_load_fn\=\@(\S+)', data)
        if m:
            return m.group(1)
        elif gin_file in SOCIAL_BOT_GIN_FILES:
            return "suite_socialbot.load"
        elif gin_file in GYM_GIN_FILES:
            return "suite_gym.load"
        elif gin_file in MJC_GIN_FILES:
            return "suite_robotics.load"
    assert False, "NO ENV LOAD FN in {}.".format(gin_file)


def create_run_yaml(tag, gazebo=False):
    out_fn = None
    with open('run.yaml', 'r') as f_in:
        data = f_in.read()
        out_data = data.replace('RUNTAG', tag)
        out_data = out_data.replace('RUNSCRIPT',
                                    "run-gazebo.sh" if gazebo else "run.sh")
        out_data = out_data.replace('__gpu_per_worker__', '1')
        alf_v = '0.0.6-mujoco200-cu10'
        out_data = out_data.replace('__alf_version__', alf_v)
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
                arg = "--gin_param='{}={}'".format(arg_name, value)
                if value == "" or value is None:
                    arg = "--gin_param='{}'".format(arg_name)
                args.append(arg)
            m_arg_value[name] = value
        elif name == 'obstc':
            arg_name = task + ".distraction_list"
            if value == "1":
                value = '["coke_can"]'
            args.append("--gin_param='{}={}'".format(arg_name, value))
            m_arg_value[name] = value
        elif name not in ['rnn']:
            remaining_args.append(name)
    if ('obtransObNorm' in named_tags
            or 'obnorm' in named_tags) and 'ddpg_' in gin_file:
        args.append(
            "--gin_param='suite_robotics.load.concat_desired_goal=False'")
        args.append(
            "--gin_param='actor/ActorNetwork.preprocessing_combiner=@NestConcat()'"
        )
        args.append(
            "--gin_param='critic/CriticNetwork.observation_preprocessing_combiner=@NestConcat()'"
        )
        args.append(
            "--gin_param='critic/CriticNetwork.action_preprocessing_combiner=@NestConcat()'"
        )
    # if 'obnorm' in named_tags:
    #     args.append("--gin_param='Agent.observation_transformer=[]'")
    if 'prjn' in m_arg_value and m_arg_value['prjn'] == "True":
        args.append("--gin_param='vae_output_dims.use_projection_net=True'")
    if 'image' not in m_arg_value and gin_file in TARGET_NAV_GIN_FILES:
        args.append("--gin_param='image_scale_transformer.fields=[]'")
        # args.append("--gin_param='Agent.observation_transformer=[]'")
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
    args.sort()
    # print('gin_params:\n' + '\n'.join(args))
    print("\n========================================")
    if len(remaining_args) > 0:
        for name in remaining_args:
            print(
                '* Warning: parameter name \033[91m{}\033[00m not understood by parser.'
                .format(name))
        print("========================================")
    return ' '.join(args), m_arg_value


# Mappings:


def get_task_class(gin_file_tag):
    res = "NO_TASK"
    if "targetnav" in gin_file_tag or "targetspnav" in gin_file_tag:
        res = "GoalTask"
    elif "push" in gin_file_tag and "fetch" not in gin_file_tag:
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
        'eit':
            'TrainerConfig.eval_interval',
        'evit':
            'TrainerConfig.eval_interval',
        'lr': ['ac/AdamTF.lr', 'AdamTF.lr', 'Adam.lr'],
        'adameps': ['ac/AdamTF.eps', 'AdamTF.eps', 'Adam.eps'],
        'tffn':
            'TrainerConfig.use_tf_functions',
        'unr':
            'TrainerConfig.unroll_length',
        'unroll':
            'TrainerConfig.unroll_length',
        'upit':
            'TrainerConfig.num_updates_per_train_iter',
        'updatepit':
            'TrainerConfig.num_updates_per_train_iter',
        'rblen':
            'TrainerConfig.replay_buffer_length',
        'rbrecsteps':
            'ReplayBuffer.recent_data_steps',
        'rbrecratio':
            'ReplayBuffer.recent_data_ratio',
        'saverb':
            'ReplayBuffer.enable_checkpoint',
        'td':
            'ActorCriticLoss.td_lambda',
        'ent':
            'ActorCriticLoss.entropy_regularization',
        'ddpgrnd':
            'DdpgAlgorithm.rollout_random_action',
        'crp':
            'DdpgAlgorithm.num_critic_replicas',
        'crirep':
            'DdpgAlgorithm.num_critic_replicas',
        'crireptrn':
            'DdpgAlgorithm.num_critics_for_training',
        'pn':
            'DdpgAlgorithm.use_parallel_network',
        'paranet':
            'DdpgAlgorithm.use_parallel_network',
        'actl2':
            'DdpgAlgorithm.action_l2',
        # 'rmf': 'DdpgAlgorithm.remove_field_for_training',
        'oustddev':
            'DdpgAlgorithm.ou_stddev',
        'ddpgtargupdt':
            'DdpgAlgorithm.target_update_period',
        'nhercritic':
            'DdpgAlgorithm.use_non_her_critic',
        'keepher':
            'DdpgAlgorithm.keep_her_rate',
        'dshv':
            'DdpgAlgorithm.down_sample_high_value',
        # 'preactcost': 'DdpgAlgorithm.pre_activation_cost',
        # 'gausseps': 'DdpgAlgorithm.gaussian_eps',
        'spltexp':
            'DdpgAlgorithm.split_exp_on_replicas',
        'gnetpos':
            'DdpgAlgorithm.goal_net_position_only',
        'reqs':
            'DdpgAlgorithm.replan_value',
        'qsip1':
            'DdpgAlgorithm.bump_target_value',
        'reqsip1':
            'DdpgAlgorithm.replan_target_value',
        'replanqsip1':
            'DdpgAlgorithm.replan_target_value',
        'pvlr':
            'DdpgAlgorithm.plan_value_learning_rate',
        'dvclp':
            'DdpgAlgorithm.value_clipping',
        'dvclpsigmd': [
            'DdpgAlgorithm.value_clipping',
            'DdpgAlgorithm.sigmoid_value_clipping'
        ],
        'gdist':
            'TDLoss.improve_w_goal_return',
        'herk':
            'HindsightExperienceTransformer.her_proportion',  # 'hindsight_relabel_fn.her_proportion',
        'herorigg':
            'hindsight_relabel_fn.use_original_goals_from_info',
        'herfin':
            'hindsight_relabel_fn.relabel_final_goal',
        'rdthd': [
            'hindsight_relabel_fn.threshold',
            'l2_dist_close_reward_fn.threshold'
        ],
        'rwddistthd':
            'l2_dist_close_reward_fn.threshold',
        'mdimgr': [
            'hindsight_relabel_fn.multi_dim_goal_reward',
            'SubgoalPlanningGoalGenerator.multi_dim_goal_reward'
        ],
        'possprwd': [
            'hindsight_relabel_fn.combine_position_speed_reward',
            'l2_dist_close_reward_fn.combine_position_speed_reward'
        ],
        "gnoise":
            'hindsight_relabel_fn.add_noise_to_goals',
        "cobj":
            'PushReachTask.close_to_agent',
        'goalgen':
            'Agent.goal_generator',  #@SubgoalPlanningGoalGenerator()
        'numsg':
            'SubgoalPlanningGoalGenerator.num_subgoals',
        'gcreps':
            'SubgoalPlanningGoalGenerator.use_critic_replicas',
        'combhnhv':
            'SubgoalPlanningGoalGenerator.combine_her_nonher_value_weight',
        'nextgoalonsucc':
            'SubgoalPlanningGoalGenerator.next_goal_on_success',
        'fingoal':
            'SubgoalPlanningGoalGenerator.final_goal',
        'normg':
            'SubgoalPlanningGoalGenerator.normalize_goals',
        'normbg': [
            'SubgoalPlanningGoalGenerator.normalize_goals',
            'SubgoalPlanningGoalGenerator.bound_goals'
        ],
        'msgsteps':
            'SubgoalPlanningGoalGenerator.max_subgoal_steps',
        'mplansteps':
            'SubgoalPlanningGoalGenerator.max_replan_steps',
        'apln':
            'SubgoalPlanningGoalGenerator.always_adopt_plan',
        'adoptplan':
            'SubgoalPlanningGoalGenerator.always_adopt_plan',
        'vclp':
            'SubgoalPlanningGoalGenerator.value_clipping',
        'sgmargin':
            'SubgoalPlanningGoalGenerator.plan_margin',
        'gmincost':
            'SubgoalPlanningGoalGenerator.min_goal_cost_to_use_plan',
        'pgoalv':
            'SubgoalPlanningGoalGenerator.plan_with_goal_value_only',
        'premainsg':
            'SubgoalPlanningGoalGenerator.plan_remaining_subgoals',
        'pgach':
            'SubgoalPlanningGoalGenerator.plan_after_goal_achieved',
        'cvae':
            'SubgoalPlanningGoalGenerator.use_cvae',
        'prjn':
            'SubgoalPlanningGoalGenerator.use_projection_net',
        'vaew':
            'SubgoalPlanningGoalGenerator.vae_weight',
        'vaewada':
            'SubgoalPlanningGoalGenerator.vae_weight_adaptive',
        'vaeb':
            'VariationalAutoEncoder.beta',
        'vaethd':
            'SubgoalPlanningGoalGenerator.vae_penalize_above',
        'vaethdada':
            'SubgoalPlanningGoalGenerator.vae_threshold_adaptive',
        'vaebias':
            'SubgoalPlanningGoalGenerator.vae_bias',
        'vaensp':
            'SubgoalPlanningGoalGenerator.vae_samples',
        'vaezdim':
            'encoding_dim',
        'vaeintarg':
            'SubgoalPlanningGoalGenerator.use_vae_in_target',
        'subdesire':
            'SubgoalPlanningGoalGenerator.subtract_desired',
        'infyaw':
            'SubgoalPlanningGoalGenerator.infer_yaw',
        'sglnnorm':
            'SubgoalPlanningGoalGenerator.plan_cost_ln_norm',
        'sglpenal':
            'SubgoalPlanningGoalGenerator.sg_leng_penalty',
        'sglthd':
            'SubgoalPlanningGoalGenerator.subgoal_cost_thd',
        'gspeedzero':
            'SubgoalPlanningGoalGenerator.goal_speed_zero',
        'auxach': [
            'goal_gen/SubgoalPlanningGoalGenerator.use_aux_achieved',
            task + '.use_aux_achieved'
        ],
        'xyaux':
            task + '.xy_only_aux',
        'auxdim':
            'SubgoalPlanningGoalGenerator.aux_dim',
        'ctrlaux': [
            'SubgoalPlanningGoalGenerator.control_aux',
            # 'vae_output_spec.control_aux',
            'hindsight_relabel_fn.control_aux'
        ],
        'goustd':
            'SubgoalPlanningGoalGenerator.gou_stddev',
        'goalgensrc':
            'ConditionalGoalGenerator.train_with_goal',
        'collect':
            'TrainerConfig.initial_collect_steps',
        'coll':
            'TrainerConfig.initial_collect_steps',
        'vtrace':
            'ActorCriticLoss.use_vtrace',
        'impclip':
            'action_importance_ratio.importance_ratio_clipping',
        'tilestsen': [
            'get_value_network.num_state_tiles',
            'get_value_network.num_sentence_tiles',
            'get_actor_network.num_state_tiles',
            'get_actor_network.num_sentence_tiles',
        ],
        'asyncreplay':
            'AsyncOffPolicyDriver.exp_replayer',
        'batlen':
            'TrainerConfig.mini_batch_length',
        'bs':
            'TrainerConfig.mini_batch_size',
        'batsz':
            'TrainerConfig.mini_batch_size',
        'mstp': [task + '.max_steps', parent_task + '.max_steps'],
        'msteps': [task + '.max_steps', parent_task + '.max_steps'],
        'stept':
            parent_task + '.step_time',
        'mestp':
            load_fn + '.max_episode_steps',
        'mepisteps':
            load_fn + '.max_episode_steps',
        'spsr': [
            load_fn + '.sparse_reward',
            'HindsightExperienceTransformer.sparse_reward',
            #'hindsight_relabel_fn.sparse_reward',
            #'SubgoalPlanningGoalGenerator.sparse_reward',
        ],
        'sparserwd': [
            load_fn + '.sparse_reward',
            'HindsightExperienceTransformer.sparse_reward',
            #'hindsight_relabel_fn.sparse_reward',
            #'SubgoalPlanningGoalGenerator.sparse_reward',
        ],
        'rwdcap': [
            'transform_reward.reward_cap',
            'transform_reward_tensor.reward_cap',
            'suite_robotics.SparseReward.reward_cap',
            'TDLoss.reward_multiplier',
        ],
        'fskip':
            'FrameSkip.skip',
        'esu':
            task + '.end_episode_after_success',
        'endsucc':
            task + '.end_episode_after_success',
        'endhit':
            task + '.end_on_hitting_distraction',
        'mdimr':
            task + '.multi_dim_reward',
        'mvgoalinepi':
            task + '.move_goal_during_episode',
        'resetclockonsucc':
            task + '.reset_time_limit_on_success',
        'chaintask':
            task + '.chain_task_rate',
        'gymwrap':
            load_fn + '.gym_env_wrappers',
        'succwrap':
            load_fn + '.use_success_wrapper',
        'succwrapsteps':
            'SuccessWrapper.since_episode_steps',
        'aclipmin':
            'ContinuousActionClip.min_v',
        'aclipmax':
            'ContinuousActionClip.max_v',
        'shootopt':
            'RandomShootingAlgorithm.plan_optimizer',
        'cemiters':
            'CEMOptimizer.iterations',
        'it':
            'TrainerConfig.num_iterations',
        'iters':
            'TrainerConfig.num_iterations',
        'shoothoriz':
            'RandomShootingAlgorithm.planning_horizon',
        'shootpopu':
            'RandomShootingAlgorithm.population_size',
        'cempopu':
            'CEMOptimizer.population_size',
        'randgoal':
            task + '.random_goal',
        'goalname':
            task + '.goal_name',
        'succangle':
            task + '.success_with_angle_requirement',
        'splimit':
            task + '.speed_goal_limit',
        'distract':
            task + '.distraction_list',
        'curridistr':
            task + '.curriculum_distractions',
        'curriangl':
            task + '.curriculum_target_angle',
        'episwgoal':
            task + '.switch_goal_within_episode',
        'rrg':
            task + '.random_range',
        'randrange':
            task + '.random_range',
        'mindist':
            task + '.min_distance',
        'faildist':
            task + '.fail_distance_thresh',
        'hitdist':
            task + '.distraction_penalty_distance_thresh',
        'hpen':
            task + '.distraction_penalty',
        'hitpenal':
            task + '.distraction_penalty',
        'randagentpp': [
            task + '.random_agent_orientation',
            task + '.random_agent_position',
        ],
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
        'fullst':
            task + '.use_full_states',
        # used to be simplefullst=>use_egocentric_states
        'egocent':
            task + '.use_egocentric_states',
        'anglelimit': [
            task + '.egocentric_perception_range',
            'get_ac_networks.angle_limit'
        ],
        # 'orderobjbyview': [
        #     task + '.order_obj_by_view', 'get_ac_networks.has_obj_id'
        # ],
        'rnn':
            'get_ac_networks.rnn',
        'lstm': [
            'ActorDistributionRNNNetwork.lstm_hidden_size',
            'ValueRNNNetwork.lstm_hidden_size'
        ],
        'actact': ['actor/ActorNetwork.activation', ],
        'fc': [
            'actor/ActorNetwork.fc_layer_params',
            'actor/ActorDistributionNetwork.fc_layer_params',
            'critic/CriticNetwork.joint_fc_layer_params',
            'value/ValueNetwork.fc_layer_params',
        ],
        'actsqsh':
            'actor/ActorNetwork.squashing_func',
        'obnorm':
            'TrainerConfig.normalize_observations',  # branch obn: gsp
        # 'obtrans': 'Agent.observation_transformer',
        'normeps':
            'AdaptiveNormalizer.variance_epsilon',
        'heads':
            'get_ac_networks.n_heads',
        'gft':
            'get_ac_networks.gft',
        'entmax':
            'EntropyTargetAlgorithm.max_entropy',
        'initalpha':
            'EntropyTargetAlgorithm.initial_alpha',
        # curriculum learning:
        'curri':
            task + '.use_curriculum_training',
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
        # 'obclip': load_fn + '.clip_observation', disabled, use gymwrap_@ObservationClipWrapper_... instead
    }
    return switcher.get(name, None)


def tokenize(v, quoted=False):
    v = v.replace('^', VALUE_SEP)
    v = v.replace('+', VALUE_SEP)
    v = v.replace('=', ARG_SEP)
    v = v.replace('LP', '(')
    v = v.replace('RP', ')')
    if quoted:
        v = '"' + v + '"'
    return v


def map_list_value(value, quoted=False):
    values = value.split(VALUE_SEP)

    translated = [tokenize(v, quoted=quoted) for v in values]
    return ','.join(translated)


def brace_map_list_value(value):
    if value in ('None', '0', 'False'):
        return "()"
    return '(' + map_list_value(value) + ',)'


def maybe_map_value(key, value=None):
    switcher = {
        'image': brace_map_list_value(value),  # 84_84 => (84,84)
        'lstm': brace_map_list_value(value),
        'fc': brace_map_list_value(value),
        'gymwrap': brace_map_list_value(value),
        'goalname': map_list_value(value, quoted=True),
        'spltexp': map_list_value(value, quoted=True),
        'distract':
            '[' + map_list_value(value, quoted=True) +
            ']',  # ball_car__wheel => ball_car+wheel => [ball,car_wheel]
        'world': tokenize(value, quoted=True),
        'asyncreplay': tokenize(value, quoted=True),
        'rmf': tokenize(value, quoted=True),
        'shootopt': tokenize(value, quoted=True),
        'goalgensrc': tokenize(value, quoted=True),
        # 'obtrans': tokenize(value, quoted=True),
    }
    return switcher.get(key, map_list_value(value))


def v(map, key, default=None):
    if key in map and map[key] != "0":
        return map[key]
    else:
        return default


def choose_cluster():
    with open('gpucluster.yaml', 'r') as f:
        cluster_str = f.read()
    from conf import Config
    assert FLAGS.cluster in Config.clusters, \
        "Cluster name {} is unrecognized!".format(FLAGS.cluster)
    id_and_key = Config.clusters[FLAGS.cluster]
    cluster_str = cluster_str.replace("__appid__", id_and_key["appid"])
    cluster_str = cluster_str.replace("__appkey__", id_and_key["appkey"])
    cluster_str = cluster_str.replace(
        "__endpoint__", id_and_key.get("endpoint", "idc-v2.hobot.cc"))
    with open("gpucluster-c.yaml", 'w') as f:
        f.write(cluster_str)

    os.system(
        "mkdir -p $HOME/.hobot; cp gpucluster-c.yaml $HOME/.hobot/gpucluster.yaml"
    )


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
            v = kv[1]
        else:
            if "=" in kv:
                v = ""
            else:
                v = "True"
        named_tags[kv[0]] = maybe_map_value(kv[0], value=v)

    if "gin" in named_tags:
        gin_file_tag = named_tags["gin"]
    gin_file = get_gin_file(gin_file_tag)

    task = get_task_class(gin_file_tag)
    parent_task = get_parent_task(task)
    train = "play" if FLAGS.play else "train"
    root_dir = FLAGS.path + "/" + run_tag
    root_dir = root_dir.replace("(", "LP")
    root_dir = root_dir.replace(")", "RP")
    if FLAGS.root_dir:
        root_dir = FLAGS.root_dir
    log_file = root_dir + "/{}.log".format(train)

    load_fn = get_env_load_fn(gin_file)

    args, m_arg_value = gen_args(named_tags, task, load_fn, gin_file)

    if os.path.exists("../bin/train.py"):
        module = train
        additional_flag = ""
    else:
        module = "main"
        additional_flag = " --play" if FLAGS.play else ""

    if FLAGS.log is None:
        log_to_file = not FLAGS.nolog
        if FLAGS.play:
            log_to_file = False
    else:
        log_to_file = FLAGS.log

    if log_to_file:
        log_cmd = " 2> {log_file} > {log_file}.out".format(log_file=log_file)
    else:
        log_cmd = ""
    if FLAGS.play:
        # additional_flag += " --epsilon_greedy=0.1"
        additional_flag += " --random_seed={}".format(FLAGS.seed)
        if not FLAGS.mp4_file:
            if FLAGS.episodes:
                additional_flag += " --num_episodes={}".format(FLAGS.episodes)
                if FLAGS.episodes == 1:
                    additional_flag += " --epsilon_greedy=0"
            else:
                additional_flag += " --num_episodes=100"
        if FLAGS.mp4_file:
            additional_flag += " --record_file='{}'".format(FLAGS.mp4_file)
        if FLAGS.ckpt:
            additional_flag += " --checkpoint_step={}".format(FLAGS.ckpt)
        if gin_file_tag in TARGET_NAV_GIN_TAGS:
            additional_flag += " --gin_param='{}.start_range=3'".format(task)
            additional_flag += " --gin_param='{}.max_steps=100'".format(
                parent_task)
            additional_flag += " --gin_param='suite_socialbot.load.max_episode_steps=1000'"
        if gin_file_tag in TARGET_NAV_GIN_TAGS or FLAGS.verbosity > 0:
            additional_flag += " --verbosity=1"
        if FLAGS.sleep:
            additional_flag += " --sleep_time_per_step={}".format(FLAGS.sleep)
    if gin_file_tag in TARGET_NAV_GIN_TAGS and "push" not in gin_file_tag:
        additional_flag += " --gin_param='{}.polar_coord=False'".format(task)

    additional_flag += " --gin_param='TrainerConfig.summarize_output=True'"
    if FLAGS.gin:
        additional_flag += " " + FLAGS.gin

    command = (
        "python3 ../bin/{module}.py --gin_file={gin_file} --root_dir={root_dir} "
        + args + additional_flag + log_cmd).format(
            module=module, gin_file=gin_file, root_dir=root_dir)

    # WFH remote playing with vglrun.
    # Run "/opt/TurboVNC/bin/vncserver :5" before playing.
    if FLAGS.play:  # and os.environ.get('DISPLAY') == ':5':
        command = 'DISPLAY=:5 vglrun -d :7 ' + command

    if FLAGS.cluster:
        choose_cluster()
        os.chdir('/home/users/le.zhao/test_gail/alf_submit/gail')
        run_yaml = create_run_yaml(
            run_tag, gazebo=gin_file_tag in SOCIAL_BOT_GIN_TAGS)
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
            os.system('tail -n 200 {}'.format(log_file))


if __name__ == '__main__':
    app.run(main)
