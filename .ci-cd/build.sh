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
        alf.algorithms.actor_critic_algorithm_test \
        alf.algorithms.actor_critic_loss_test \
        alf.algorithms.algorithm_test \
        alf.algorithms.ddpg_algorithm_test \
        alf.algorithms.diayn_algorithm_test \
        alf.algorithms.ppo_algorithm_test \
        alf.algorithms.rl_algorithm_test \
        alf.algorithms.trac_algorithm_test \
        alf.algorithms.vae_test \
        alf.bin.train_play_test \
        alf.data_structures_test \
        alf.device_ctx_test \
        alf.environments.suite_unittest_test \
        alf.environments.simple.noisy_array_test \
        alf.experience_replayers.replay_buffer_test \
        alf.metrics.metrics_test \
        alf.nest.nest_test \
        alf.networks.network_test \
        alf.networks.actor_distribution_networks_test \
        alf.networks.actor_networks_test \
        alf.networks.critic_networks_test \
        alf.networks.value_networks_test \
        alf.networks.encoding_networks_test \
        alf.networks.projection_networks_test \
        alf.networks.q_networks_test \
        alf.optimizers.trusted_updater_test \
        alf.summary.summary_ops_test \
        alf.tensor_specs_test \
        alf.trainers.policy_trainer_test \
        alf.utils.checkpoint_utils_test \
        alf.utils.data_buffer_test \
        alf.utils.dist_utils_test \
        alf.utils.normalizers_test \
        alf.utils.tensor_utils_test \
        alf.utils.value_ops_test \

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
