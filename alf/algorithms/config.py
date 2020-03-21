# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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

import gin


@gin.configurable
class TrainerConfig(object):
    """TrainerConfig

    Note: This is a mixture collection configuration for all training setups,
    not all parameter is operative and its not necessary to config it.

    1. `num_steps_per_iter` is only for on_policy_trainer.

    2. `initial_collect_steps`, `num_updates_per_train_step`, `mini_batch_length`,
    `mini_batch_size`, `clear_replay_buffer`, `num_envs` are used by sync_off_policy_trainer and
    async_off_policy_trainer.
    """

    def __init__(self,
                 root_dir,
                 algorithm_ctor=None,
                 random_seed=None,
                 num_iterations=1000,
                 num_env_steps=0,
                 unroll_length=8,
                 use_rollout_state=False,
                 num_checkpoints=10,
                 evaluate=False,
                 eval_interval=10,
                 epsilon_greedy=0.1,
                 num_eval_episodes=10,
                 summary_interval=50,
                 update_counter_every_mini_batch=False,
                 summaries_flush_secs=1,
                 summary_max_queue=10,
                 debug_summaries=False,
                 summarize_grads_and_vars=False,
                 summarize_action_distributions=False,
                 num_steps_per_iter=10000,
                 initial_collect_steps=0,
                 num_updates_per_train_step=4,
                 mini_batch_length=None,
                 mini_batch_size=None,
                 exp_replayer="uniform",
                 whole_replay_buffer_training=True,
                 replay_buffer_length=1024,
                 clear_replay_buffer=True,
                 num_envs=1):
        """Configuration for Trainers

        Args:
            root_dir (str): directory for saving summary and checkpoints
            algorithm_ctor (Callable): callable that create an
                `OffPolicyAlgorithm` or `OnPolicyAlgorithm` instance
            random_seed (None|int): random seed, a random seed is used if None
            num_iterations (int): number of update iterations (ignored if 0)
            num_env_steps (int): number of environment steps (ignored if 0). The
                total number of FRAMES will be (`num_env_steps`*`frame_skip`) for
                calculating sample efficiency. See alf/environments/wrappers.py
                for the definition of FrameSkip.
            unroll_length (int):  number of time steps each environment proceeds per
                iteration. The total number of time steps from all environments per
                iteration can be computed as: `num_envs` * `env_batch_size`
                * `unroll_length`.
            use_rollout_state (bool): Include the RNN state for the experiences
                used for off-policy training
            num_checkpoints (int): how many checkpoints to save for the training
            evaluate (bool): A bool to evaluate when training
            eval_interval (int): evaluate every so many iteration
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation.
            num_eval_episodes (int) : number of episodes for one evaluation
            summary_interval (int): write summary every so many training steps
            update_counter_every_mini_batch (bool): whether to update counter
                for every mini batch. The `summary_interval` is based on this
                counter. Typically, this should be False. Set to True if you
                want to have summary for every mini batch for the purpose of
                debugging.
            summaries_flush_secs (int): flush summary to disk every so many seconds
            summary_max_queue (int): flush to disk every so mary summaries
            debug_summaries (bool): A bool to gather debug summaries.
            summarize_grads_and_vars (bool): If True, gradient and network variable
                summaries will be written during training.
            num_steps_per_iter (int): number of steps for one iteration. It is the
                total steps from all individual environment in the batch
                environment.
            initial_collect_steps (int): if positive, number of steps each single
                environment steps before perform first update
            num_updates_per_train_step (int): number of optimization steps for
                one iteration
            mini_batch_size (int): number of sequences for each minibatch. If None,
                it's set to the replayer's `batch_size`.
            mini_batch_length (int): the length of the sequence for each
                sample in the minibatch. If None, it's set to `unroll_length`.
            exp_replayer (str): "uniform" or "one_time"
            whole_replay_buffer_training (bool): whether use all data in replay
                buffer to perform one update
            clear_replay_buffer (bool): whether use all data in replay buffer to
                perform one update and then wiped clean
            replay_buffer_length (int): the maximum number of steps the replay
                buffer store for each environment.
            num_envs (int): the number of environments to run asynchronously.
        """
        parameters = dict(
            root_dir=root_dir,
            algorithm_ctor=algorithm_ctor,
            random_seed=random_seed,
            num_iterations=num_iterations,
            num_env_steps=num_env_steps,
            unroll_length=unroll_length,
            use_rollout_state=use_rollout_state,
            num_checkpoints=num_checkpoints,
            evaluate=evaluate,
            eval_interval=eval_interval,
            epsilon_greedy=epsilon_greedy,
            num_eval_episodes=num_eval_episodes,
            summary_interval=summary_interval,
            update_counter_every_mini_batch=update_counter_every_mini_batch,
            summaries_flush_secs=summaries_flush_secs,
            summary_max_queue=summary_max_queue,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            summarize_action_distributions=summarize_action_distributions,
            num_steps_per_iter=num_steps_per_iter,
            initial_collect_steps=initial_collect_steps,
            num_updates_per_train_step=num_updates_per_train_step,
            mini_batch_length=mini_batch_length,
            mini_batch_size=mini_batch_size,
            exp_replayer=exp_replayer,
            whole_replay_buffer_training=whole_replay_buffer_training,
            clear_replay_buffer=clear_replay_buffer,
            replay_buffer_length=replay_buffer_length,
            num_envs=num_envs)
        for k, v in parameters.items():
            self.__setattr__(k, v)
