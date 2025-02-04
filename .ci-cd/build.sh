#!/usr/bin/env bash
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


function print_usage() {
    echo -e "\nUsage:
     [OPTION]
     \nOptions:
     test: run all unit tests
     check_style: run code style check
    "
}

function check_style() {
    trap 'abort' 0
    set -e

    export PATH=/usr/bin:$PATH
    git config --global --add safe.directory `pwd`
    pre-commit install
    clang-format --version

    if ! pre-commit run -a ; then
        git diff
        exit 1
    fi

    trap : 0
}

function test() {
    cd alf
    python3 environments/make_penv.py
    python3 -m unittest -v \
        alf.algorithms.actor_critic_algorithm_test \
        alf.algorithms.actor_critic_loss_test \
        alf.algorithms.agent_test \
        alf.algorithms.algorithm_test \
        alf.algorithms.async_unroller_test \
        alf.algorithms.containers_test \
        alf.algorithms.data_transformer_test \
        alf.algorithms.ddpg_algorithm_test \
        alf.algorithms.dsac_algorithm_test \
        alf.algorithms.diayn_algorithm_test \
        alf.algorithms.entropy_target_algorithm_test \
        alf.algorithms.functional_particle_vi_algorithm_test \
        alf.algorithms.hypernetwork_algorithm_test \
        alf.algorithms.icm_algorithm_test \
        alf.algorithms.generator_test \
        alf.algorithms.inverse_mvp_algorithm_test \
        alf.algorithms.lagrangian_reward_weight_algorithm_test \
        alf.algorithms.mcts_algorithm_test \
        alf.algorithms.merlin_algorithm_test \
        alf.algorithms.mi_estimator_test \
        alf.algorithms.muzero_algorithm_test \
        alf.algorithms.particle_vi_algorithm_test \
        alf.algorithms.ppo_algorithm_test \
        alf.algorithms.predictive_representation_learner_test \
        alf.algorithms.prior_actor_test \
        alf.algorithms.qrsac_algorithm_test \
        alf.algorithms.rl_algorithm_test \
        alf.algorithms.rlpd_algorithm_test \
        alf.algorithms.sarsa_algorithm_test \
        alf.algorithms.sac_algorithm_test \
        alf.algorithms.oac_algorithm_test \
        alf.algorithms.trac_algorithm_test \
        alf.algorithms.vae_test \
        alf.algorithms.ppg.disjoint_policy_value_network_test \
        alf.bin.train_play_test \
        alf.norm_layers_test \
        alf.config_util_test \
        alf.data_structures_test \
        alf.device_ctx_test \
        alf.environments.gym_wrappers_test \
        alf.environments.parallel_environment_test \
        alf.environments.process_environment_test \
        alf.environments.random_alf_environment_test \
        alf.environments.simple.noisy_array_test \
        alf.environments.suite_dmc_test \
        alf.environments.suite_go_test \
        alf.environments.suite_gym_test \
        alf.environments.suite_mario_test \
        alf.environments.suite_socialbot_test \
        alf.environments.suite_tic_tac_toe_test \
        alf.environments.suite_unittest_test \
        alf.environments.alf_environment_test \
        alf.environments.alf_gym_wrapper_test \
        alf.environments.alf_wrappers_test \
        alf.experience_replayers.replay_buffer_test \
        alf.experience_replayers.segment_tree_test \
        alf.initializers_test \
        alf.layers_test \
        alf.metrics.metrics_test \
        alf.nest.nest_test \
        alf.networks.action_encoder_test \
        alf.networks.actor_distribution_networks_test \
        alf.networks.actor_networks_test \
        alf.networks.containers_test \
        alf.networks.critic_networks_test \
        alf.networks.encoding_networks_test \
        alf.networks.memory_test \
        alf.networks.network_test \
        alf.networks.networks_test \
        alf.networks.normalizing_flow_networks_test \
        alf.networks.param_networks_test \
        alf.networks.preprocessors_test \
        alf.networks.projection_networks_test \
        alf.networks.q_networks_test \
        alf.networks.relu_mlp_test \
        alf.networks.value_networks_test \
        alf.optimizers.nero_plus_test \
        alf.optimizers.optimizers_test \
        alf.optimizers.trusted_updater_test \
        alf.summary.summary_ops_test \
        alf.tensor_specs_test \
        alf.trainers.policy_trainer_test \
        alf.utils.action_samplers_test \
        alf.utils.checkpoint_utils_test \
        alf.utils.common_test \
        alf.utils.data_buffer_test \
        alf.utils.dist_utils_test \
        alf.utils.distributions_test \
        alf.utils.lean_function_test \
        alf.utils.losses_test \
        alf.utils.math_ops_test \
        alf.utils.normalizers_test \
        alf.utils.schedulers_test \
        alf.utils.tensor_utils_test \
        alf.utils.tensorrt_utils_test \
        alf.utils.value_ops_test \
        alf.examples.networks.impala_cnn_encoder_test

    cd ..
}

function main() {
    set -e
    local CMD=$1
    case $CMD in
        check_style)
          check_style
          ;;
        test)
          test
          ;;
        *)
          print_usage
          exit 0
          ;;
    esac
}

main $@
