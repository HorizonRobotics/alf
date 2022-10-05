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
"""Trainer for training an Algorithm on given environments."""

import abc
from absl import logging
from typing import Dict
import math
import os
import pprint
from pathlib import Path
import re
import signal
import sys
import time
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

import alf
from alf.algorithms.algorithm import Algorithm, Loss
from alf.networks import Network
from alf.algorithms.config import TrainerConfig
from alf.algorithms.data_transformer import (create_data_transformer,
                                             IdentityDataTransformer)
from alf.data_structures import StepType
from alf.environments.utils import create_environment
from alf.nest import map_structure
from alf.tensor_specs import TensorSpec
from alf.utils import common
from alf.utils import git_utils
from alf.utils import math_ops
from alf.utils.checkpoint_utils import Checkpointer
import alf.utils.datagen as datagen
from alf.utils.summary_utils import record_time
from .evaluator import Evaluator


class _TrainerProgress(nn.Module):
    def __init__(self):
        super(_TrainerProgress, self).__init__()
        self.register_buffer("_iter_num", torch.zeros((), dtype=torch.int64))
        self.register_buffer("_env_steps", torch.zeros((), dtype=torch.int64))
        self._num_iterations = None
        self._num_env_steps = None
        self._progress = None

    def set_termination_criterion(self, num_iterations, num_env_steps=0):
        self._num_iterations = float(num_iterations)
        self._num_env_steps = float(num_env_steps)
        # might be loaded from a checkpoint, so we update first
        self.update()

    def update(self, iter_num=None, env_steps=None):
        if iter_num is not None:
            self._iter_num.fill_(iter_num)
        if env_steps is not None:
            self._env_steps.fill_(env_steps)

        assert not (self._num_iterations is None
                    and self._num_env_steps is None), (
                        "You must first call set_terimination_criterion()!")
        if self._num_iterations > 0:
            self._progress = float(
                self._iter_num.to(torch.float64) / self._num_iterations)
        else:
            self._progress = float(
                self._env_steps.to(torch.float64) / self._num_env_steps)

    def set_progress(self, value: float):
        """Manually set the current progress.

        Args:
            value: a float number in [0, 1]
        """
        self._progress = value

    @property
    def progress(self):
        assert self._progress is not None, "Must call update() first!"
        return self._progress


def _visualize_alf_tree(module: Algorithm):
    """Generate a graphviz graph of the module tree structure.

    This is useful to visualize the hierarchy of the current AFL algorithm.

    Args:
        module: An ALF algorithm.

    Returns:
        A graphviz directed graph that can be rendered as pdf.
    """
    try:
        import graphviz
    except ImportError:
        logging.warn(
            'Need "graphviz" installed if you want to visualize modules')
        return None

    def _is_layer(node):
        class_name = node.__class__.__name__
        return (isinstance(node, nn.Module) and class_name in dir(alf.layers))

    def _visual_style(node: torch.nn.Module) -> Dict[str, str]:
        """Loss: 'gray',
           Algorithm: 'blue',
           Network: 'orange',
           Layer: 'yellow'
        """
        if isinstance(node, Loss):
            return {
                'style': 'filled',
                'fillcolor': '#DCDCDC',
            }
        elif isinstance(node, Algorithm):
            return {
                'style': 'filled',
                'fillcolor': '#00BFFF',
            }
        elif isinstance(node, Network):
            return {
                'style': 'filled',
                'fillcolor': '#FF8C00',
            }
        elif _is_layer(node):
            return {'style': 'filled', 'fillcolor': '#ffdc7d', 'fontsize': '8'}
        return {}

    def _generate_node_label(node):
        """Generate the proper label for a given node.
        """

        def _get_func_name(match_obj):
            """Further extract the function name from the <...> representation.

            For example, if the match_obj corresponds to a string like below:

                <built-in method relu_ of type object at 0x7ff7a790f620>

            This function extracts "relu_" out of it.
            """
            # Such representation can start with either "bound method",
            # "built-in method" or "function".
            res = re.match(
                r'<(bound method|built-in method|function) (\S+) .*>',
                match_obj.group())
            if res is None:
                # In case there is an outlier, return "NOT_PARSED" instead.
                return 'NOT_PARSED'
            if len(res.group(2)) > 10:
                # Shorten the function name if it is very long.
                return f'{res.group(2)[:10]}...'
            return res.group(2)

        if _is_layer(node):
            # We need to parse function repr with pattern <... at 0x???> because
            # graphviz doesn't support '<' or '>' in the label
            return re.sub("<[^<]*>", _get_func_name, repr(node))
        else:
            return getattr(node, "name", type(node).__name__)

    def _filter_child(field, child):
        """A set of rules to filter out certain components in the rendered graph.
        """
        conditions = [
            # Every Algorithm will contain a default identity transformer.
            (field == "_data_transformer"
             and isinstance(child, IdentityDataTransformer)),
        ]
        return any(conditions)

    dot = graphviz.Digraph()
    dot.attr('node', shape='record')
    dot.graph_attr['rankdir'] = 'LR'

    def _visit(node, idx, visited):
        """Visit a node by depth-first search. For each algorithm node, we create
        a subgraph that encloses all its children.
        """
        idx[0] += 1
        node_index = idx[0]
        visited[node] = node_index
        label = _generate_node_label(node)
        node_records = ["<caption> " + label + f"(id={node_index})"]
        edges = []

        for field, child in node.named_children():
            if _filter_child(field, child):
                continue
            if child not in visited:
                edges += _visit(child, idx, visited)
            child_idx = visited[child]
            node_records.append(f'<{field}> ({field})')
            edge = (f'{node_index}:{field}', f'{child_idx}:caption')
            edges.append(edge)

        dot.node(
            str(node_index),
            label='|'.join(node_records),
            **_visual_style(node))

        if isinstance(node, Algorithm):
            # NOTE: the subgraph name needs to begin with 'cluster' (all lowercase)
            #       so that Graphviz recognizes it as a special cluster subgraph
            with dot.subgraph(name=f'cluster_{node_index}') as c:
                c.attr(color='green')
                if node_index != 0:
                    # Do not draw duplicate edges for subgraphs
                    c.edge_attr['style'] = 'invis'
                c.edges(edges)
                c.attr(label=label)

        return edges

    _visit(module, idx=[-1], visited={})

    return dot


class Trainer(object):
    """Base class for trainers.

    Trainer is responsible for creating algorithm and dataset/environment, setting up
    summary, checkpointing, running training iterations, and evaluating periodically.
    """

    _trainer_progress = _TrainerProgress()

    def __init__(self, config: TrainerConfig, ddp_rank: int = -1):
        """

        Args:
            config: configuration used to construct this trainer
            ddp_rank: process (and also device) ID of the process, if the
                process participates in a DDP process group to run distributed
                data parallel training. A value of -1 indicates regular single
                process training.
        """
        Trainer._trainer_progress = _TrainerProgress()
        root_dir = config.root_dir
        self._root_dir = root_dir
        self._train_dir = os.path.join(root_dir, 'train')
        self._eval_dir = os.path.join(root_dir, 'eval')

        self._algorithm_ctor = config.algorithm_ctor
        self._algorithm = None

        self._num_checkpoints = config.num_checkpoints
        self._checkpointer = None

        self._evaluate = config.evaluate
        self._eval_uncertainty = config.eval_uncertainty

        if config.num_evals is not None:
            self._eval_interval = common.compute_summary_or_eval_interval(
                config, config.num_evals)
        else:
            self._eval_interval = config.eval_interval

        if config.num_summaries is not None:
            self._summary_interval = common.compute_summary_or_eval_interval(
                config, config.num_summaries)
        else:
            self._summary_interval = config.summary_interval

        self._summaries_flush_secs = config.summaries_flush_secs
        self._summary_max_queue = config.summary_max_queue
        self._debug_summaries = config.debug_summaries
        self._summarize_grads_and_vars = config.summarize_grads_and_vars
        self._config = config
        self._random_seed = config.random_seed
        self._rank = ddp_rank
        self._conf_file_content = common.read_conf_file(root_dir)

    def train(self):
        """Perform training."""
        self._restore_checkpoint()
        alf.summary.enable_summary()

        self._checkpoint_requested = False
        signal.signal(signal.SIGUSR2, self._request_checkpoint)
        # kill -12 PID
        logging.info("Use `kill -%s %s` to request checkpoint during training."
                     % (int(signal.SIGUSR2), os.getpid()))

        self._debug_requested = False
        # kill -10 PID
        signal.signal(signal.SIGUSR1, self._request_debug)
        logging.info("Use `kill -%s %s` to request debugging." % (int(
            signal.SIGUSR1), os.getpid()))

        checkpoint_saved = False
        try:
            if self._config.profiling:
                import cProfile, pstats, io
                pr = cProfile.Profile()
                pr.enable()

            common.run_under_record_context(
                self._train,
                summary_dir=self._train_dir,
                summary_interval=self._summary_interval,
                summarize_first_interval=self._config.summarize_first_interval,
                flush_secs=self._summaries_flush_secs,
                summary_max_queue=self._summary_max_queue)

            if self._config.profiling:
                pr.disable()
                s = io.StringIO()
                ps = pstats.Stats(pr, stream=s).sort_stats('time')
                ps.print_stats()
                ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
                ps.print_stats()
                ps.print_callees()

                logging.info(s.getvalue())
            self._save_checkpoint()
            checkpoint_saved = True
        finally:
            if (self._config.confirm_checkpoint_upon_crash
                    and not checkpoint_saved and self._rank <= 0):
                # Prompts for checkpoint only when running single process
                # training (rank is -1) or master process of DDP training (rank
                # is 0).
                ans = input("Do you want to save checkpoint? (y/n): ")
                if ans.lower().startswith('y'):
                    self._save_checkpoint()
            self._close()

    @staticmethod
    def progress():
        """A static method that returns the current training progress, provided
        that only one trainer will be used for training.

        Returns:
            float: a number in :math:`[0,1]` indicating the training progress.
        """
        return Trainer._trainer_progress.progress

    @staticmethod
    def current_iterations():
        return Trainer._trainer_progress._iter_num

    @staticmethod
    def current_env_steps():
        return Trainer._trainer_progress._env_steps

    def _train(self):
        """Perform training according the the learning type. """
        pass

    def _close(self):
        """Closing operations after training. """
        pass

    def _summarize_training_setting(self):
        # We need to wait for one iteration to get the operative args
        # Right just give a fixed gin file name to store operative args
        common.write_config(self._root_dir, self._conf_file_content)
        with alf.summary.record_if(lambda: True):

            def _markdownify(paragraph):
                return "    ".join(
                    (os.linesep + paragraph).splitlines(keepends=True))

            common.summarize_config()
            alf.summary.text('commandline', ' '.join(sys.argv))
            alf.summary.text(
                'optimizers',
                _markdownify(self._algorithm.get_optimizer_info()))
            alf.summary.text(
                'unoptimized_parameters',
                _markdownify(self._algorithm.get_unoptimized_parameter_info()))
            alf.summary.text('revision', git_utils.get_revision())
            alf.summary.text('diff', _markdownify(git_utils.get_diff()))
            alf.summary.text('seed', str(self._random_seed))

            # Save a rendered directed graph of the algorithm to the root
            # directory.
            algorithm_structure_graph = _visualize_alf_tree(self._algorithm)
            if algorithm_structure_graph is not None:
                import graphviz
                try:
                    algorithm_structure_graph.render(
                        Path(self._root_dir, 'algorithm_sturcture'),
                        format='png',
                        quiet=True)
                except graphviz.backend.CalledProcessError as e:
                    # graphviz will treat any warning in the rendering as error
                    # and panic. We should just warn instead.
                    logging.warn(f'Graphviz rendering: {str(e)}')
                image_path = Path(self._root_dir, 'algorithm_sturcture.png')
                if image_path.exists():
                    img = np.array(Image.open(image_path))
                    alf.summary.images(
                        'algorithm_structure', img, dataformat='HWC', step=0)

            if self._config.code_snapshots is not None:
                for f in self._config.code_snapshots:
                    path = os.path.join(
                        os.path.abspath(os.path.dirname(__file__)), "..", f)
                    if not os.path.isfile(path):
                        common.warning_once(
                            "The code file '%s' for summary is invalid" % path)
                        continue
                    with open(path, 'r') as fin:
                        code = fin.read()
                        # adding "<pre>" will make TB show raw text instead of MD
                        alf.summary.text('code/%s' % f,
                                         "<pre>" + code + "</pre>")

    def _request_checkpoint(self, signum, frame):
        self._checkpoint_requested = True

    def _request_debug(self, signum, frame):
        self._debug_requested = True

    def _save_checkpoint(self):
        # Saving checkpoint is only enabled when running single process training
        # (rank is -1) or master process of DDP training (rank is 0).
        if self._rank <= 0:
            global_step = alf.summary.get_global_counter()
            self._checkpointer.save(global_step=global_step)

    def _restore_checkpoint(self, checkpointer):
        """Retore from saved checkpoint.

            Args:
                checkpointer (Checkpointer):
        """
        if checkpointer.has_checkpoint():
            # Some objects (e.g. ReplayBuffer) are constructed lazily in algorithm.
            # They only appear after one training iteration. So we need to run
            # train_iter() once before loading the checkpoint
            self._algorithm.train_iter()

        try:
            recovered_global_step = checkpointer.load()
            self._trainer_progress.update()
        except RuntimeError as e:
            raise RuntimeError(
                ("Checkpoint loading failed from the provided root_dir={}. "
                 "Typically this is caused by using a wrong checkpoint. \n"
                 "Please make sure the root_dir is set correctly. "
                 "Use a new value for it if "
                 "planning to train from scratch. \n"
                 "Detailed error message: {}").format(self._root_dir, e))
        if recovered_global_step != -1:
            alf.summary.set_global_counter(recovered_global_step)

        self._checkpointer = checkpointer


class RLTrainer(Trainer):
    """Trainer for reinforcement learning. """

    def __init__(self, config: TrainerConfig, ddp_rank: int = -1):
        """

        Args:
            config (TrainerConfig): configuration used to construct this trainer
            ddp_rank (int): process (and also device) ID of the process, if the
                process participates in a DDP process group to run distributed
                data parallel training. A value of -1 indicates regular single
                process training.
        """
        super().__init__(config, ddp_rank)

        self._num_env_steps = config.num_env_steps
        self._num_iterations = config.num_iterations
        assert self._num_iterations + self._num_env_steps > 0, \
            "Must provide #iterations or #env_steps for training!"
        if self._num_iterations > 0 and self._num_env_steps > 0:
            num_envs = alf.get_config_value(
                "create_environment.num_parallel_environments")
            num_iterations_with_env_interations = config.num_env_steps / (
                num_envs * config.unroll_length)
            pure_train_iters = self._num_iterations - num_iterations_with_env_interations
            assert pure_train_iters >= 0, (
                f"num_iterations={self._num_iterations} is not enough for "
                f"num_env_steps={self._num_env_steps}")
            logging.info("There is no environmental interation in the last"
                         f"{pure_train_iters} iterations")
        self._trainer_progress.set_termination_criterion(
            self._num_iterations, self._num_env_steps)

        self._num_eval_episodes = config.num_eval_episodes
        alf.summary.should_summarize_output(config.summarize_output)

        env = alf.get_env()
        logging.info(
            "observation_spec=%s" % pprint.pformat(env.observation_spec()))
        logging.info("action_spec=%s" % pprint.pformat(env.action_spec()))

        # for offline buffer construction
        untransformed_observation_spec = env.observation_spec()

        data_transformer = create_data_transformer(
            config.data_transformer_ctor, untransformed_observation_spec)
        self._config.data_transformer = data_transformer

        # keep compatibility with previous gin based config
        common.set_global_env(env)

        observation_spec = data_transformer.transformed_observation_spec
        common.set_transformed_observation_spec(observation_spec)
        logging.info("transformed_observation_spec=%s" %
                     pprint.pformat(observation_spec))

        self._algorithm = self._algorithm_ctor(
            observation_spec=observation_spec,
            action_spec=env.action_spec(),
            reward_spec=env.reward_spec(),
            env=env,
            config=self._config,
            debug_summaries=self._debug_summaries)

        # recover offline buffer
        self._algorithm.load_offline_replay_buffer(
            untransformed_observation_spec)

        self._algorithm.set_path('')
        if ddp_rank >= 0:
            # Activate the DDP training
            self._algorithm.activate_ddp(ddp_rank)

        # Create a thread env to expose subprocess gin/alf configurations
        # which otherwise will be marked as "inoperative". Only created when
        # ``TrainerConfig.no_thread_env_for_conf=False``.
        self._thread_env = None

        # See ``alf/docs/notes/knowledge_base.rst```
        # (ParallelAlfEnvironment and ThreadEnvironment) for details.
        if not config.no_thread_env_for_conf and isinstance(
                env,
                alf.environments.parallel_environment.ParallelAlfEnvironment):
            self._thread_env = create_environment(
                nonparallel=True, seed=self._random_seed)

        if self._evaluate:
            self._evaluator = Evaluator(self._config, common.get_conf_file())

    def _close_envs(self):
        """Close all envs to release their resources."""
        alf.close_env()
        if self._thread_env is not None:
            self._thread_env.close()

    def _train(self):
        env = alf.get_env()
        env.reset()
        iter_num = int(self._trainer_progress._iter_num)
        training_setting_summarized = False

        checkpoint_interval = math.ceil(
            (self._num_iterations
             or self._num_env_steps) / self._num_checkpoints)

        if self._num_iterations:
            time_to_checkpoint = self._trainer_progress._iter_num + checkpoint_interval
        else:
            time_to_checkpoint = self._trainer_progress._env_steps + checkpoint_interval

        if self._evaluate and iter_num == 0:
            self._eval()

        while True:
            t0 = time.time()
            with record_time("time/train_iter"):
                train_steps = self._algorithm.train_iter()
            t = time.time() - t0
            logging.log_every_n_seconds(
                logging.INFO,
                '%s%s -> %s: %s time=%.3f throughput=%0.2f' %
                ('' if self._rank == -1 else f'[rank {self._rank:02d}] ',
                 common.get_conf_file(),
                 os.path.basename(self._root_dir.strip('/')), iter_num, t,
                 int(train_steps) / t),
                n_seconds=1)

            just_evaluated = False
            if self._evaluate and (iter_num + 1) % self._eval_interval == 0:
                if (self._config.num_evals is None
                        or (iter_num + 1) // self._eval_interval <
                        self._config.num_evals):
                    # If num_evals is specified, the last evaluation will be
                    # performed after training finishes.
                    self._eval()
                    just_evaluated = True
            if not training_setting_summarized and train_steps > 0:
                self._summarize_training_setting()
                training_setting_summarized = True

            # check termination
            env_steps_metric = self._algorithm.get_step_metrics()[1]
            total_time_steps = env_steps_metric.result()
            iter_num += 1

            self._trainer_progress.update(iter_num, total_time_steps)

            if ((self._num_iterations and iter_num >= self._num_iterations)
                    or (not self._num_iterations
                        and total_time_steps >= self._num_env_steps)):
                # Evaluate before exiting so that the eval curve shown in TB
                # will align with the final iter/env_step.
                if self._evaluate and not just_evaluated:
                    self._eval()
                break

            if ((self._num_iterations and iter_num >= time_to_checkpoint)
                    or (not self._num_iterations and self._num_env_steps
                        and total_time_steps >= time_to_checkpoint)):
                self._save_checkpoint()
                time_to_checkpoint += checkpoint_interval
            elif self._checkpoint_requested:
                logging.info("Saving checkpoint upon request...")
                self._save_checkpoint()
                self._checkpoint_requested = False

            if self._debug_requested:
                self._debug_requested = False
                import pdb
                pdb.set_trace()

    def _close(self):
        """Closing operations after training. """
        self._algorithm.finish_train()
        self._close_envs()
        if self._evaluate:
            self._evaluator.close()

    def _restore_checkpoint(self):
        checkpointer = Checkpointer(
            ckpt_dir=os.path.join(self._train_dir, 'algorithm'),
            algorithm=self._algorithm,
            metrics=nn.ModuleList(self._algorithm.get_metrics()),
            trainer_progress=self._trainer_progress)

        super()._restore_checkpoint(checkpointer)

    def _eval(self):
        step_metrics = self._algorithm.get_step_metrics()
        step_metrics = dict((m.name, int(m.result())) for m in step_metrics)
        self._evaluator.eval(self._algorithm, step_metrics)


class SLTrainer(Trainer):
    """Trainer for supervised learning. """

    def __init__(self, config: TrainerConfig):
        """Create a SLTrainer

        Args:
            config (TrainerConfig): configuration used to construct this trainer
        """
        super().__init__(config)

        assert config.num_iterations > 0, \
            "Must provide num_iterations for training!"

        self._num_epochs = config.num_iterations
        self._trainer_progress.set_termination_criterion(self._num_epochs)
        self._algorithm = config.algorithm_ctor(config=config)
        self._algorithm.set_path('')

    def _train(self):
        begin_epoch_num = int(self._trainer_progress._iter_num)
        epoch_num = begin_epoch_num

        checkpoint_interval = math.ceil(
            self._num_epochs / self._num_checkpoints)
        time_to_checkpoint = begin_epoch_num + checkpoint_interval

        logging.info("==> Begin Training")
        while True:
            t0 = time.time()
            with record_time("time/train_iter"):
                train_steps = self._algorithm.train_iter()
                train_steps = train_steps or 1
            t = time.time() - t0
            logging.log_every_n_seconds(
                logging.INFO,
                '%s -> %s: %s time=%.3f throughput=%0.2f' %
                (common.get_conf_file(),
                 os.path.basename(self._root_dir.strip('/')), epoch_num, t,
                 int(train_steps) / t),
                n_seconds=1)

            if (epoch_num + 1) % self._eval_interval == 0:
                if self._evaluate:
                    self._algorithm.evaluate()
                if self._eval_uncertainty:
                    self._algorithm.eval_uncertainty()

            if epoch_num == begin_epoch_num:
                self._summarize_training_setting()

            # check termination
            epoch_num += 1
            self._trainer_progress.update(epoch_num)

            if (self._num_epochs and epoch_num >= self._num_epochs):
                if self._evaluate:
                    self._algorithm.evaluate()
                if self._eval_uncertainty:
                    self._algorithm.eval_uncertainty()
                break

            if self._num_epochs and epoch_num >= time_to_checkpoint:
                self._save_checkpoint()
                time_to_checkpoint += checkpoint_interval
            elif self._checkpoint_requested:
                logging.info("Saving checkpoint upon request...")
                self._save_checkpoint()
                self._checkpoint_requested = False

            if self._debug_requested:
                self._debug_requested = False
                import pdb
                pdb.set_trace()

    def _restore_checkpoint(self):
        checkpointer = Checkpointer(
            ckpt_dir=os.path.join(self._train_dir, 'algorithm'),
            algorithm=self._algorithm,
            trainer_progress=self._trainer_progress)

        super()._restore_checkpoint(checkpointer)


@torch.no_grad()
def _step(algorithm,
          env,
          time_step,
          policy_state,
          trans_state,
          metrics,
          render=False,
          recorder=None,
          sleep_time_per_step=0,
          selective_criteria_func=None):
    """Perform one step interaction using the outpupt action from ``algorithm``
    taking ``time_step`` as input. Also record the metrics.

    Note that this function is used both in ``play`` below and ``evaluate`` in
    ``evaluator.py``.

    Args:
        algorithm (RLAlgorithm): the algorithm under evaluation
        env: the environment
        time_step (TimeStep): current time step
        policy_state (nested Tensor): state of the policy
        trans_state (nested Tensor): state of the transformer(s)
        metrics (StepMetric): a list of metrics that will be updated based on
            ``time_step``.
        render (bool|False): if True, display the frames of ``env`` on a screen.
        recorder (VideoRecorder|None): recorder the frames of ``env`` and other
            additional images in prediction step info if present.
        sleep_time_per_step (int|0): The sleep time between two frames when
            ``render`` is True.
        selective_criteria_func (callable|None): a callable for determining
            whether an episode will be saved to the video file when a valid
            recorder is provided. This function takes two input arguments:
            - return (float): return of the current episode. This is useful for
                implementing return based selective criteria.
            - env_info (dict): a dictionary containing information returned by
                the environment. This is useful for implementing task specific
                selective criteria using information contained ``env_info``,
                e.g., success, infraction etc.

    Returns:
        - next time step (TimeStep): the next time step after taking an action in
            ``env``
        - policy step (AlgStep): the output from ``algorithm.predict_step``
        - new state of the transformer(s) (nested Tensor)
    """

    for metric in metrics:
        metric(time_step.cpu())

    policy_state = common.reset_state_if_necessary(
        policy_state, algorithm.get_initial_predict_state(env.batch_size),
        time_step.is_first())
    transformed_time_step, trans_state = algorithm.transform_timestep(
        time_step, trans_state)
    policy_step = algorithm.predict_step(transformed_time_step, policy_state)

    if recorder and selective_criteria_func is None:
        recorder.capture_frame(policy_step.info, time_step.is_last())

    elif recorder and selective_criteria_func is not None:
        env_frame = recorder.capture_env_frame()
        recorder.cache_frame_and_pred_info(env_frame, policy_step.info)

        if time_step.is_last():
            if selective_criteria_func(
                    map_structure(lambda x: x.cpu().numpy(),
                                  metrics[1].latest()),
                    map_structure(lambda x: x.cpu().numpy(),
                                  metrics[3].latest())):
                logging.info(
                    "+++++++++ Selective Case Discovered! +++++++++++")
                recorder.generate_video_from_cache()
            else:
                recorder.clear_cache()

    elif render:
        if env.batch_size > 1:
            env.envs[0].render(mode='human')
        else:
            env.render(mode='human')
        time.sleep(sleep_time_per_step)

    next_time_step = env.step(policy_step.output)

    return next_time_step, policy_step, trans_state


@common.mark_eval
def play(root_dir,
         env,
         algorithm,
         checkpoint_step="latest",
         num_episodes=10,
         sleep_time_per_step=0.01,
         record_file=None,
         append_blank_frames=0,
         render=True,
         selective_mode=False,
         ignored_parameter_prefixes=[]):
    """Play using the latest checkpoint under `train_dir`.

    The following example record the play of a trained model to a mp4 video:
    .. code-block:: bash

        python -m alf.bin.play \
        --root_dir=~/tmp/bullet_humanoid/ppo2/ppo2-11 \
        --num_episodes=1 \
        --record_file=ppo_bullet_humanoid.mp4

    Args:
        root_dir (str): same as the root_dir used for `train()`
        env (AlfEnvironment): the environment
        algorithm (RLAlgorithm): the training algorithm
        checkpoint_step (int|str): the number of training steps which is used to
            specify the checkpoint to be loaded. If checkpoint_step is 'latest',
            the most recent checkpoint named 'latest' will be loaded.
        num_episodes (int): number of episodes to play
        sleep_time_per_step (float): sleep so many seconds for each step
        record_file (str): if provided, video will be recorded to a file
            instead of shown on the screen.
        append_blank_frames (int): If >0, wil append such number of blank frames
            at the end of the episode in the rendered video file. A negative
            value has the same effects as 0 and no blank frames will be appended.
            This option has no effects when displaying the frames on the screen
            instead of recording to a file.
        render (bool): If False, then this function only evaluates the trained
            model without calling rendering functions. This value will be ignored
            if a ``record_file`` argument is provided.
        selective_mode (bool): whether to save the selective cases discovered
            according to a ``selective_criteria_func``.
        ignored_parameter_prefixes (list[str]): ignore the parameters whose
            name has one of these prefixes in the checkpoint.
    """
    train_dir = os.path.join(root_dir, 'train')

    ckpt_dir = os.path.join(train_dir, 'algorithm')
    checkpointer = Checkpointer(
        ckpt_dir=ckpt_dir,
        algorithm=algorithm,
        trainer_progress=Trainer._trainer_progress)
    recovered_global_step = checkpointer.load(
        checkpoint_step,
        ignored_parameter_prefixes=ignored_parameter_prefixes,
        including_optimizer=False,
        including_replay_buffer=False,
        including_data_transformers=True,
        strict=True)
    # The behavior of some algorithms is based by scheduler using training
    # progress or global step. So we need to set a valid value for progress
    # and global step
    if recovered_global_step != -1:
        alf.summary.set_global_counter(recovered_global_step)
    Trainer._trainer_progress.set_termination_criterion(
        alf.get_config_value('TrainerConfig.num_iterations'),
        alf.get_config_value('TrainerConfig.num_env_steps'))
    Trainer._trainer_progress.update()
    logging.info("global_step=%s TrainerProgress=%s" % (recovered_global_step,
                                                        Trainer.progress()))

    batch_size = env.batch_size
    recorder = None
    if record_file is not None:
        assert batch_size == 1, 'video recording is not supported for parallel play'
        # Note that ``VideoRecorder`` will import ``matplotlib`` which might have
        # some side effects on xserver (if its backend needs graphics).
        # This is incompatible with RLBench parallel envs >1 (or other
        # envs requiring xserver) for some unknown reasons, so we have a lazy import here.
        from alf.utils.video_recorder import VideoRecorder
        recorder = VideoRecorder(
            env, append_blank_frames=append_blank_frames, path=record_file)
    elif render:
        if batch_size > 1:
            env.envs[0].render(mode='human')
        else:
            # pybullet_envs need to render() before reset() to enable mode='human'
            env.render(mode='human')
    env.reset()

    time_step = common.get_initial_time_step(env)
    algorithm.eval()
    policy_state = algorithm.get_initial_predict_state(env.batch_size)
    trans_state = algorithm.get_initial_transform_state(env.batch_size)
    episodes_per_env = (num_episodes + batch_size - 1) // batch_size
    env_episodes = torch.zeros(batch_size, dtype=torch.int32)
    episode_reward = torch.zeros(batch_size)
    episode_length = torch.zeros(batch_size, dtype=torch.int32)
    episodes = 0
    metrics = [
        alf.metrics.NumberOfEpisodes(),
        alf.metrics.AverageReturnMetric(
            buffer_size=num_episodes, example_time_step=time_step),
        alf.metrics.AverageEpisodeLengthMetric(
            example_time_step=time_step, buffer_size=num_episodes),
        alf.metrics.AverageEnvInfoMetric(
            example_time_step=time_step, buffer_size=num_episodes),
        alf.metrics.AverageDiscountedReturnMetric(
            buffer_size=num_episodes, example_time_step=time_step)
    ]

    if selective_mode:
        # Below is an example selective criteria based on return.
        # This should be adjusted according to the particular task.
        selective_criteria_func = lambda return_value, env_info: return_value < 500
    else:
        selective_criteria_func = None

    while episodes < num_episodes:
        # For parallel play, we cannot naively pick the first finished `num_episodes`
        # episodes to estimate the average return (or other statitics) as it can be
        # biased. Instead, we stick to using the first episodes_per_env episodes
        # from each environment to calculate the statistics and ignore the potentially
        # extra episodes from each environment.
        invalid = env_episodes >= episodes_per_env
        # Force the step_type of the extra episodes to be StepType.FIRST so that
        # these time steps do not affect metrics as the metrics are only updated
        # at StepType.LAST. The metric computation uses cpu version of time_step.
        time_step.cpu().step_type[invalid] = StepType.FIRST

        next_time_step, policy_step, trans_state = _step(
            algorithm=algorithm,
            env=env,
            time_step=time_step,
            policy_state=policy_state,
            trans_state=trans_state,
            metrics=metrics,
            render=render,
            recorder=recorder,
            sleep_time_per_step=sleep_time_per_step,
            selective_criteria_func=selective_criteria_func)

        time_step.step_type[invalid] = StepType.FIRST
        started = time_step.step_type != StepType.FIRST
        episode_length += started
        episode_reward += started * time_step.reward

        for i in range(batch_size):
            if time_step.step_type[i] == StepType.LAST:
                logging.info(
                    "episode_length=%s episode_reward=%s" %
                    (episode_length[i].item(), episode_reward[i].item()))
                episode_reward[i] = 0.
                episode_length[i] = 0
                env_episodes[i] += 1
                episodes += 1
                common.log_metrics(metrics)

        policy_state = policy_step.state
        time_step = next_time_step

    if recorder:
        recorder.close()
    env.reset()
