# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import alf
from alf.utils.schedulers import as_scheduler


@alf.configurable
class TrainerConfig(object):
    """Configuration for training."""

    def __init__(self,
                 root_dir,
                 ml_type='rl',
                 algorithm_ctor=None,
                 data_transformer_ctor=None,
                 random_seed=None,
                 num_iterations=1000,
                 num_env_steps=0,
                 unroll_length=8,
                 unroll_with_grad=False,
                 use_rollout_state=False,
                 temporally_independent_train_step=None,
                 num_checkpoints=10,
                 confirm_checkpoint_upon_crash=True,
                 no_thread_env_for_conf=False,
                 load_checkpoint_strict=True,
                 evaluate=False,
                 num_evals=None,
                 eval_interval=10,
                 epsilon_greedy=0.,
                 eval_uncertainty=False,
                 num_eval_episodes=10,
                 num_summaries=None,
                 summary_interval=50,
                 summarize_first_interval=True,
                 update_counter_every_mini_batch=False,
                 summaries_flush_secs=1,
                 summary_max_queue=10,
                 metric_min_buffer_size=10,
                 debug_summaries=False,
                 profiling=False,
                 enable_amp=False,
                 code_snapshots=None,
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
                 clear_replay_buffer=True):
        """
        Args:
            root_dir (str): directory for saving summary and checkpoints
            ml_type (str): type of learning task, one of ['rl', 'sl']
            algorithm_ctor (Callable): callable that create an
                ``OffPolicyAlgorithm`` or ``OnPolicyAlgorithm`` instance
            data_transformer_ctor (Callable|list[Callable]): Function(s)
                for creating data transformer(s). Each of them will be called
                as ``data_transformer_ctor(observation_spec)`` to create a data
                transformer. Available transformers are in ``algorithms.data_transformer``.
                The data transformer constructed by this can be access as
                ``TrainerConfig.data_transformer``.
                Important Note: ``HindsightExperienceTransformer``, ``FrameStacker`` or
                any data transformer that need to access the replay buffer
                for additional data need to be before all other data transformers.
                The reason is the following:
                In off policy training, the replay buffer stores raw input w/o being
                processed by any data transformer.  If say ``ObservationNormalizer`` is
                applied before hindsight, then data retrieved by replay will be
                normalized whereas hindsight data directly pulled from the replay buffer
                will not be normalized.  Data will be in mismatch, causing training to
                suffer and potentially fail.
            random_seed (None|int): random seed, a random seed is used if None
            num_iterations (int): For RL trainer, indicates number of update
                iterations (ignored if 0). Note that for off-policy algorithms, if
                ``initial_collect_steps>0``, then the first
                ``initial_collect_steps//(unroll_length*num_envs)`` iterations
                won't perform any training. For SL trainer, indicates the number
                of training epochs.
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
            confirm_checkpoint_upon_crash (bool): whether to prompt for whether
                do checkpointing after crash.
            no_thread_env_for_conf (bool): not to create an unwrapped env for
                the purpose of showing operative configurations. If True, no
                ``ThreadEnvironment`` will ever be created, regardless of the
                value of ``TrainerConfig.evaluate``. If False, a
                ``ThreadEnvironment`` will be created if ``TrainerConfig.evaluate``
                or the training env is a ``ParallelAlfEnvironment`` instance.
                For an env that consume lots of resources, this flag can be set to
                ``True`` if no evaluation is needed to save resources. The decision
                of creating an unwrapped env won't affect training; it's used to
                correctly display inoperative configurations in subprocesses.
            load_checkpoint_strict (bool): whether to strictly enforce that the keys
                in ``state_dict`` match the keys returned by module's
                ``torch.nn.Module.state_dict`` function. If True, will
                keep lists of missing and unexpected keys and raise error when
                any of the lists is non-empty; if ``strict=False``, missing/unexpected
                keys will be omitted and no error will be raised.
            evaluate (bool): A bool to evaluate when training
            num_evals (int): how many evaluations are needed throughout the training.
                If not None, an automatically calculated ``eval_interval`` will
                replace ``config.eval_interval``.
            eval_interval (int): evaluate every so many iteration
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation.
            eval_uncertainty (bool): whether to evluate uncertainty after training.
            num_eval_episodes (int) : number of episodes for one evaluation
            num_summaries (int): how many summary calls are needed throughout the
                training. If not None, an automatically calculated ``summary_interval``
                will replace ``config.summary_interval``. Note that this number
                doesn't include the summary steps of the first interval if
                ``summarize_first_interval=True``. In this case, the actual number
                of summaries will be roughly this number plus the calculated
                summary interval.
            summary_interval (int): write summary every so many training steps
            summarize_first_interval (bool): whether to summarize every step of
                the first interval (default True). It might be better to turn
                this off for an easier post-processing of the curve.
            update_counter_every_mini_batch (bool): whether to update counter
                for every mini batch. The ``summary_interval`` is based on this
                counter. Typically, this should be False. Set to True if you
                want to have summary for every mini batch for the purpose of
                debugging. Only used by ``OffPolicyAlgorithm``.
            summaries_flush_secs (int): flush summary to disk every so many seconds
            summary_max_queue (int): flush to disk every so mary summaries
            metric_min_buffer_size (int): a minimal size of the buffer used to
                construct some average episodic metrics used in ``RLAlgorithm``.
            debug_summaries (bool): A bool to gather debug summaries.
            profiling (bool): If True, use cProfile to profile the training. The
                profile result will be written to ``root_dir``/py_train.INFO.
            enable_amp: whether to use automatic mixed precision for training.
                This can makes the training faster if the algorithm is GPU intensive.
                However, the result may be different (mostly likely due to random
                fluctuation).
            code_snapshots (list[str]): an optional list of code files to write
                to tensorboard text. Note: the code file path should be relative
                to "<ALF_ROOT>/alf", e.g., "algorithms/agent.py". This can be
                useful for tracking code changes when running a job.
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
                sample in the minibatch. Only used by ``OffPolicyAlgorithm``.
            whole_replay_buffer_training (bool): whether use all data in replay
                buffer to perform one update. Only used by ``OffPolicyAlgorithm``.
            clear_replay_buffer (bool): whether use all data in replay buffer to
                perform one update and then wiped clean. Only used by
                ``OffPolicyAlgorithm``.
            replay_buffer_length (int): the maximum number of steps the replay
                buffer store for each environment. Only used by
                ``OffPolicyAlgorithm``.
            priority_replay (bool): Use prioritized sampling if this is True.
            priority_replay_alpha (float|Scheduler): The priority from LossInfo is powered
                to this as an argument for ``ReplayBuffer.update_priority()``.
                Note that the effect of ``ReplayBuffer.initial_priority``
                may change with different values of ``priority_replay_alpha``.
                Hence you may need to adjust ``ReplayBuffer.initial_priority``
                accordingly.
            priority_replay_beta (float|Scheduler): weight the loss of each sample by
                ``importance_weight**(-priority_replay_beta)``, where ``importance_weight``
                is from the BatchInfo returned by ``ReplayBuffer.get_batch()``.
                This is only useful if ``prioritized_sampling`` is enabled for
                ``ReplayBuffer``.
            priority_replay_eps (float): minimum priority for priority replay.
        """
        if isinstance(priority_replay_beta, float):
            assert priority_replay_beta >= 0.0, (
                "importance_weight_beta should be non-negative")
        assert ml_type in ('rl', 'sl')
        self.root_dir = root_dir
        self.ml_type = ml_type
        self.algorithm_ctor = algorithm_ctor
        self.data_transformer_ctor = data_transformer_ctor
        self.data_transformer = None  # to be set by Trainer
        self.random_seed = random_seed
        self.num_iterations = num_iterations
        self.num_env_steps = num_env_steps
        self.unroll_length = unroll_length
        self.unroll_with_grad = unroll_with_grad
        self.use_rollout_state = use_rollout_state
        self.temporally_independent_train_step = temporally_independent_train_step
        self.num_checkpoints = num_checkpoints
        self.confirm_checkpoint_upon_crash = confirm_checkpoint_upon_crash
        self.no_thread_env_for_conf = no_thread_env_for_conf
        self.load_checkpoint_strict = load_checkpoint_strict
        self.evaluate = evaluate
        self.num_evals = num_evals
        self.eval_interval = eval_interval
        self.epsilon_greedy = epsilon_greedy
        self.eval_uncertainty = eval_uncertainty
        self.num_eval_episodes = num_eval_episodes
        self.num_summaries = num_summaries
        self.summary_interval = summary_interval
        self.summarize_first_interval = summarize_first_interval
        self.update_counter_every_mini_batch = update_counter_every_mini_batch
        self.summaries_flush_secs = summaries_flush_secs
        self.summary_max_queue = summary_max_queue
        self.metric_min_buffer_size = metric_min_buffer_size
        self.debug_summaries = debug_summaries
        self.profiling = profiling
        self.enable_amp = enable_amp
        self.code_snapshots = code_snapshots
        self.summarize_grads_and_vars = summarize_grads_and_vars
        self.summarize_action_distributions = summarize_action_distributions
        self.summarize_output = summarize_output
        self.initial_collect_steps = initial_collect_steps
        self.num_updates_per_train_iter = num_updates_per_train_iter
        self.mini_batch_length = mini_batch_length
        self.mini_batch_size = mini_batch_size
        self.whole_replay_buffer_training = whole_replay_buffer_training
        self.clear_replay_buffer = clear_replay_buffer
        self.replay_buffer_length = replay_buffer_length
        self.priority_replay = priority_replay
        self.priority_replay_alpha = as_scheduler(priority_replay_alpha)
        self.priority_replay_beta = as_scheduler(priority_replay_beta)
        self.priority_replay_eps = priority_replay_eps
