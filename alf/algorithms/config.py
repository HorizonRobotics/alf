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
import alf


@gin.configurable
class TrainerConfig(object):
    """Configuration for training."""

    def __init__(self,
                 root_dir,
                 algorithm_ctor=None,
                 random_seed=None,
                 num_iterations=1000,
                 num_env_steps=0,
                 unroll_length=8,
                 unroll_with_grad=False,
                 use_rollout_state=False,
                 temporally_independent_train_step=None,
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
                 summarize_output=False,
                 initial_collect_steps=0,
                 num_updates_per_train_iter=4,
                 mini_batch_length=None,
                 mini_batch_size=None,
                 whole_replay_buffer_training=True,
                 replay_buffer_length=1024,
                 priority_replay=False,
                 priority_replay_alpha=0.7,
                 priority_replay_beta=0.4,
                 priority_replay_eps=1e-6,
                 clear_replay_buffer=True,
                 num_envs=1):
        """
        Args:
            root_dir (str): directory for saving summary and checkpoints
            algorithm_ctor (Callable): callable that create an
                ``OffPolicyAlgorithm`` or ``OnPolicyAlgorithm`` instance
            random_seed (None|int): random seed, a random seed is used if None
            num_iterations (int): number of update iterations (ignored if 0). Note
                that for off-policy algorithms, if ``initial_collect_steps>0``,
                then the first ``initial_collect_steps//(unroll_length*num_envs)``
                iterations won't perform any training.
            num_env_steps (int): number of environment steps (ignored if 0). The
                total number of FRAMES will be (``num_env_steps*frame_skip``) for
                calculating sample efficiency. See alf/environments/wrappers.py
                for the definition of FrameSkip.
            unroll_length (int):  number of time steps each environment proceeds per
                iteration. The total number of time steps from all environments per
                iteration can be computed as: ``num_envs * env_batch_size * unroll_length``.
            unroll_with_grad (bool): a bool flag indicating whether we require
                grad during ``unroll()``. This flag is only used by
                ``OffPolicyAlgorithm`` where unrolling with grads is usually
                unnecessary and turned off for saving memory. However, when there
                is an on-policy sub-algorithm, we can enable this flag for its
                training. ``OnPolicyAlgorithm`` always unrolls with grads and this
                flag doesn't apply to it.
            use_rollout_state (bool): If True, when off-policy training, the RNN
                states will be taken from the replay buffer; otherwise they will
                be set to 0. In the case of True, the ``train_state_spec`` of an
                algorithm should always be a subset of the ``rollout_state_spec``.
            temporally_independent_train_step (bool|None): If True, the ``train_step``
                is called with all the experiences in one batch instead of being
                called sequentially with ``mini_batch_length`` batches. Only used
                by ``OffPolicyAlgorithm``. In general, this option can only be
                used if the algorithm has no state. For Algorithm with state (e.g.
                ``SarsaAlgorithm`` not using RNN), if there is no need to
                recompute state at train_step, this option can also be used. If
                ``None``, its value is inferred based on whether the algorithm
                has RNN state (``True`` if there is RNN state, ``False`` if not).
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
                for every mini batch. The ``summary_interval`` is based on this
                counter. Typically, this should be False. Set to True if you
                want to have summary for every mini batch for the purpose of
                debugging. Only used by ``OffPolicyAlgorithm``.
            summaries_flush_secs (int): flush summary to disk every so many seconds
            summary_max_queue (int): flush to disk every so mary summaries
            debug_summaries (bool): A bool to gather debug summaries.
            summarize_grads_and_vars (bool): If True, gradient and network variable
                summaries will be written during training.
            summarize_output (bool): If True, summarize output of certain networks.
            initial_collect_steps (int): if positive, number of steps each single
                environment steps before perform first update. Only used
                by ``OffPolicyAlgorithm``.
            num_updates_per_train_iter (int): number of optimization steps for
                one iteration. Only used by ``OffPolicyAlgorithm``.
            mini_batch_size (int): number of sequences for each minibatch. If None,
                it's set to the replayer's ``batch_size``. Only used by
                ``OffPolicyAlgorithm``.
            mini_batch_length (int): the length of the sequence for each
                sample in the minibatch. If None, it's set to ``unroll_length``.
                Only used by ``OffPolicyAlgorithm``.
            whole_replay_buffer_training (bool): whether use all data in replay
                buffer to perform one update. Only used by ``OffPolicyAlgorithm``.
            clear_replay_buffer (bool): whether use all data in replay buffer to
                perform one update and then wiped clean. Only used by
                ``OffPolicyAlgorithm``.
            replay_buffer_length (int): the maximum number of steps the replay
                buffer store for each environment. Only used by
                ``OffPolicyAlgorithm``.
            priority_replay (bool): Use prioritized sampling if this is True.
            priority_replay_alpha (float): The priority from LossInfo is powered
                to this as an argument for ``ReplayBuffer.update_priority()``.
                Note that the effect of ``ReplayBuffer.initial_priority``
                may change with different values of ``priority_replay_alpha``.
                Hence you may need to adjust ``ReplayBuffer.initial_priority``
                accordingly.
            priority_replay_beta (float): weight the loss of each sample by
                ``importance_weight**(-priority_replay_beta)``, where ``importance_weight``
                is from the BatchInfo returned by ``ReplayBuffer.get_batch()``.
                This is only useful if ``prioritized_sampling`` is enabled for
                ``ReplayBuffer``.
            priority_replay_eps (float): minimum priority for priority replay.
            num_envs (int): the number of environments to run asynchronously.
        """
        assert priority_replay_beta >= 0.0, ("importance_weight_beta should "
                                             "be non-negative be")

        parameters = dict(
            root_dir=root_dir,
            algorithm_ctor=algorithm_ctor,
            random_seed=random_seed,
            num_iterations=num_iterations,
            num_env_steps=num_env_steps,
            unroll_length=unroll_length,
            unroll_with_grad=unroll_with_grad,
            use_rollout_state=use_rollout_state,
            temporally_independent_train_step=temporally_independent_train_step,
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
            summarize_output=summarize_output,
            initial_collect_steps=initial_collect_steps,
            num_updates_per_train_iter=num_updates_per_train_iter,
            mini_batch_length=mini_batch_length,
            mini_batch_size=mini_batch_size,
            whole_replay_buffer_training=whole_replay_buffer_training,
            clear_replay_buffer=clear_replay_buffer,
            replay_buffer_length=replay_buffer_length,
            priority_replay=priority_replay,
            priority_replay_alpha=priority_replay_alpha,
            priority_replay_beta=priority_replay_beta,
            priority_replay_eps=priority_replay_eps,
            num_envs=num_envs)
        for k, v in parameters.items():
            self.__setattr__(k, v)


@gin.configurable
class SupervisedTrainerConfig(object):
    """Configuration for supervised training."""

    def __init__(self,
                 root_dir,
                 algorithm_ctor=None,
                 random_seed=None,
                 epochs=2e+5,
                 eval_accuracy=True,
                 eval_uncertainty=False,
                 summary_interval=50,
                 summaries_flush_secs=1,
                 summary_max_queue=100,
                 debug_summaries=False,
                 summarize_grads_and_vars=False):
        """
        Args:
            root_dir (str): directory for saving summary and checkpoints
            algorithm_ctor (Callable): callable that create an
                ``OffPolicyAlgorithm`` or ``OnPolicyAlgorithm`` instance
            random_seed (None|int): random seed, a random seed is used if None
            epochs (int): number of training epoches
            eval_accuracy (bool): whether to evluate accuracy after training
            eval_uncertainty (bool): whether to evluate uncertainty after training
            summary_interval (int): write summary every so many training steps
            summaries_flush_secs (int): flush summary to disk every so many seconds
            summary_max_queue (int): flush to disk every so mary summaries
            summarize_grads_and_vars (bool): If True, gradient and network variable
                summaries will be written during training.
        """
        parameters = dict(
            root_dir=root_dir,
            algorithm_ctor=algorithm_ctor,
            random_seed=random_seed,
            epochs=epochs,
            evaluate=eval_accuracy,
            summary_interval=summary_interval,
            summaries_flush_secs=summaries_flush_secs,
            summary_max_queue=summary_max_queue,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars)
        for k, v in parameters.items():
            self.__setattr__(k, v)
