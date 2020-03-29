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
    python3 -m unittest -v \
        alf.tests.actor_critic_algorithm_test \
        alf.tests.actor_critic_loss_test \
        alf.tests.actor_distribution_networks_test \
        alf.tests.actor_networks_test \
        alf.tests.agent_test \
        alf.tests.algorithm_test \
        alf.tests.checkpoint_utils_test \
        alf.tests.critic_networks_test \
        alf.tests.data_buffer_test \
        alf.tests.data_structures_test \
        alf.tests.ddpg_algorithm_test \
        alf.tests.device_ctx_test \
        alf.tests.diayn_algorithm_test \
        alf.tests.dist_utils_test \
        alf.tests.encoding_networks_test \
        alf.tests.math_ops_test \
        alf.tests.memory_test \
        alf.tests.metrics_test \
        alf.tests.nest_test \
        alf.tests.network_test \
        alf.tests.noisy_array_test \
        alf.tests.normalizers_test \
        alf.tests.policy_trainer_test \
        alf.tests.ppo_algorithm_test \
        alf.tests.projection_networks_test \
        alf.tests.q_networks_test \
        alf.tests.replay_buffer_test \
        alf.tests.rl_algorithm_test \
        alf.tests.sac_algorithm_test \
        alf.tests.sarsa_algorithm_test \
        alf.tests.suite_unittest_test \
        alf.tests.summary_ops_test \
        alf.tests.tensor_specs_test \
        alf.tests.tensor_utils_test \
        alf.tests.trac_algorithm_test \
        alf.tests.train_play_test \
        alf.tests.trusted_updater_test \
        alf.tests.vae_test \
        alf.tests.value_networks_test \
        alf.tests.value_ops_test \

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
