# Copyright (c) 2022 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from absl import logging
from absl import flags
import torch.multiprocessing as mp
import os
import sys
import torch
import torch.nn as nn
from typing import Dict, List, Optional

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.environments.alf_environment import AlfEnvironment
from alf.utils import common
from alf.utils.checkpoint_utils import Checkpointer
from alf.utils.summary_utils import record_time
from alf.data_structures import StepType
from alf.algorithms.data_transformer import create_data_transformer
from alf.environments.utils import create_environment
from alf.trainers import policy_trainer
from collections import namedtuple

EvalJob = namedtuple(
    "EvalJob", ["type", "global_counter", "step_metrics", "state_dict"],
    defaults=[None] * 4)


class Evaluator(object):
    """Evaluator for performing evaluation on the current algorithm.

    If ``config.async_eval`` is True, the evaluation is performed asynchronously
    in a different process.

    For each round of evaluation, it will play ``config.num_eval_episodes`` using
    ``config.num_eval_environments`` parallel environments.

    Args:
        config: the training config
        conf_file: path of the config file
    """

    def __init__(self, config: TrainerConfig, conf_file: str):
        # The following line is needed for avoiding
        # "RuntimeError: unable to open shared memory object"
        # See https://github.com/facebookresearch/maskrcnn-benchmark/issues/103#issuecomment-785815218
        mp.set_sharing_strategy('file_system')
        self._async = config.async_eval
        if conf_file.endswith('.gin'):
            assert not self._async, "async_eval is not supported for gin_file"
        num_envs = config.num_eval_environments
        seed = config.random_seed
        if self._async:
            ctx = mp.get_context('spawn')
            self._job_queue = ctx.Queue()
            self._done_queue = ctx.Queue()
            pre_configs = dict(alf.get_handled_pre_configs())
            self._worker = ctx.Process(
                target=_worker,
                args=(self._job_queue, self._done_queue, conf_file,
                      pre_configs, num_envs, config.root_dir, seed))
            self._worker.start()
        else:
            self._env = create_environment(
                for_evaluation=True,
                num_parallel_environments=num_envs,
                seed=seed)
            self._evaluator = SyncEvaluator(self._env, config)

    def eval(self, algorithm: RLAlgorithm, step_metric_values: Dict[str, int]):
        """Do one round of evaluation.

        If ``config.async_eval`` is True, this function will return once the
        evaluator worker makes a copy of the state_dict of ``algorithm``.
        However, if the previous evaluation has not been finished, it will wait
        until it is finished.

        The evaluation result will be written to log file and tensorboard by the
        evaluation worker.

        Args:
            algorithm: the training algorithm
            step_metric_values: a dictionary of step metric values to generate
                the evaluation summaries against. Note that it needs to contain
                "EnvironmentSteps" at least.
        """
        with alf.summary.record_if(lambda: True):
            with record_time("time/evaluation"):
                if self._async:
                    job = EvalJob(
                        type="eval",
                        step_metrics=step_metric_values,
                        global_counter=int(alf.summary.get_global_counter()),
                        state_dict=algorithm.state_dict())
                    logging.info("Sending evaluation job...")
                    self._job_queue.put(job)
                    self._done_queue.get()
                    logging.info("Done sending evaluation job.")
                else:
                    self._evaluator.eval(algorithm, step_metric_values)

    def close(self):
        """Stop the ongoing evaluation and close the evaluator."""
        if self._async:
            job = EvalJob(type="stop")
            self._job_queue.put(job)
            self._worker.join()
        else:
            self._env.close()

    def wait_complete(self):
        """Wait until evaluation is complete."""
        if self._async:
            job = EvalJob(type="wait")
            self._job_queue.put(job)
            self._done_queue.get()


def _define_flags():
    flags.DEFINE_string('gin_file', None, 'Path to the gin-config file.')
    flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')
    flags.DEFINE_string('conf', None, 'Path to the alf config file.')
    flags.DEFINE_multi_string('conf_param', None, 'Config binding parameters.')


FLAGS = flags.FLAGS


class PeekableQueue(object):
    """A queue that supports peeking the first element without removing it.

    Note that this can only be used for a queue with one consumer.
    """

    def __init__(self, queue: mp.Queue):
        self._queue = queue
        self._elements = []  # elements that are peeked but not removed

    def peek(self):
        """Peek the first element in the queue without removing it.

        Returns:
            The first element in the queue. ``None`` if the queue is empty.
        """
        if len(self._elements) == 0:
            if not self._queue.empty():
                self._elements.append(self._queue.get())
        if len(self._elements) > 0:
            return self._elements[0]
        else:
            return None

    def get(self):
        if len(self._elements) == 0:
            return self._queue.get()
        else:
            return self._elements.pop(0)

    def empty(self):
        return len(self._elements) == 0 and self._queue.empty()


class SyncEvaluator(object):
    """Evaluator for performing evaluation on the current algorithm.

    For each round of evaluation, it will play ``config.num_eval_episodes`` using
    ``config.num_eval_environments`` parallel environments.
    """

    def __init__(self, env, config):
        self._env = env
        self._config = config
        eval_dir = os.path.join(config.root_dir, 'eval')
        self._summary_writer = alf.summary.create_summary_writer(
            eval_dir, flush_secs=config.summaries_flush_secs)

    def eval(self,
             algorithm: RLAlgorithm,
             step_metric_values: Dict[str, int],
             job_queue: Optional[PeekableQueue] = None):
        """Do one round of evaluation.

        This function will return after finishing the evaluation.

        The evaluation result will be written to log file and tensorboard by the
        evaluation worker.

        Args:
            algorithm: the training algorithm
            step_metric_values: a dictionary of step metric values to generate
                the evaluation summaries against. Note that it needs to contain
                "EnvironmentSteps" at least.
            job_queue: This is only used when `eval()` is called from a worker
                process. If during the evaluation, the worker receives a "stop"
                job from the main process, it will stop the evaluation and
                return immediately.
        """
        with alf.summary.push_summary_writer(self._summary_writer):
            logging.info("Start evaluation")
            metrics = evaluate(self._env, algorithm,
                               self._config.num_eval_episodes, job_queue)
            if metrics is None:
                return
            common.log_metrics(metrics)
            for metric in metrics:
                metric.gen_summaries(
                    train_step=alf.summary.get_global_counter(),
                    other_steps=step_metric_values)
            if (self._config.save_checkpoint_for_best_eval is not None
                    and self._config.save_checkpoint_for_best_eval(metrics)):
                logging.info("Saving the best checkpoint")
                checkpointer = Checkpointer(
                    ckpt_dir=os.path.join(self._config.root_dir, 'train',
                                          'algorithm'),
                    algorithm=algorithm,
                    metrics=nn.ModuleList(algorithm.get_metrics()),
                    trainer_progress=policy_trainer.Trainer.
                    get_trainer_progress())

                checkpointer.save(alf.summary.get_global_counter(), 'best')


class BestEvalChecker(object):
    """A checker to determine if the current evaluation is the best so far.

    It can be supplied to `TrainerConfig.save_checkpoint_for_best_eval` so that
    the best checkpoint will be saved when the evaluation is the best so far.

    Args:
        metric_type (type): the type of the metric to be compared. Default is
            `alf.metrics.AverageReturnMetric`.
        metric_name (None|str): if provided, the metric will be extracted from
            the result using this name. Default is None.
    """

    def __init__(self,
                 metric_type=alf.metrics.AverageReturnMetric,
                 metric_name=None):
        self._best_metric = -float('inf')
        self._metric_type = metric_type
        self._metric_name = metric_name

    def __call__(self, metrics: List[alf.metrics.StepMetric]) -> bool:
        if self._best_metric is None:
            return True
        else:
            for metric in metrics:
                if isinstance(metric, self._metric_type):
                    new_metric = metric.result()
                    if self._metric_name is not None:
                        new_metric = new_metric[self._metric_name]
                    if alf.summary.get_global_counter() == 0:
                        # The first evaluation is from the initial random model
                        # so we don't need to save it.
                        self._best_metric = new_metric.clone()
                        return False
                    if new_metric > self._best_metric:
                        self._best_metric = new_metric.clone()
                        return True
                    else:
                        return False
            raise ValueError("No metric of type %s found in the metrics" %
                             self._metric_type)


def _worker(job_queue: mp.Queue,
            done_queue: mp.Queue,
            conf_file: str,
            pre_configs: Dict,
            num_parallel_envs: int,
            root_dir: str,
            seed: Optional[int] = None):
    try:
        _define_flags()
        FLAGS(sys.argv, known_only=True)
        FLAGS.mark_as_parsed()
        FLAGS.alsologtostderr = True
        # TODO: redirect the log to the training process. Currently, all the logs
        # are written to a different log file.
        logging.set_verbosity(logging.INFO)
        logging.get_absl_handler().use_absl_log_file(log_dir=root_dir)
        logging.use_absl_handler()
        if torch.cuda.is_available():
            alf.set_default_device("cuda")
        if seed is not None:
            # seed the environments differently from the training
            seed = seed + 13579
        common.set_random_seed(seed)
        alf.config('TrainerConfig', mutable=False, random_seed=seed)
        alf.config(
            'create_environment',
            for_evaluation=True,
            num_parallel_environments=num_parallel_envs,
            mutable=False)
        try:
            alf.pre_config(pre_configs)
            common.parse_conf_file(conf_file)
        except Exception as e:
            alf.close_env()
            raise e

        config = policy_trainer.TrainerConfig(root_dir=root_dir)

        env = alf.get_env()
        env.reset()
        data_transformer = create_data_transformer(
            config.data_transformer_ctor, env.observation_spec())
        config.data_transformer = data_transformer
        # keep compatibility with previous gin based config
        common.set_global_env(env)
        observation_spec = data_transformer.transformed_observation_spec
        common.set_transformed_observation_spec(observation_spec)

        algorithm_ctor = config.algorithm_ctor
        algorithm = algorithm_ctor(
            observation_spec=observation_spec,
            action_spec=env.action_spec(),
            reward_spec=env.reward_spec(),
            config=config)
        algorithm.set_path('')
        policy_trainer.Trainer.get_trainer_progress(
        ).set_termination_criterion(config.num_iterations,
                                    config.num_env_steps)
        alf.summary.enable_summary()
        evaluator = SyncEvaluator(env, config)
        job_queue = PeekableQueue(job_queue)
        logging.info("Evaluator started")
        while True:
            job = job_queue.get()
            if job.type == "eval":
                # Some algorithms use scheduler depending on the global counter
                # or the training progress. So we make sure they are same as
                # the training process.
                alf.summary.set_global_counter(job.global_counter)
                env_steps = job.step_metrics["EnvironmentSteps"]
                policy_trainer.Trainer.get_trainer_progress().update(
                    job.global_counter, env_steps)
                algorithm.load_state_dict(job.state_dict)
                done_queue.put(None)
                evaluator.eval(algorithm, job.step_metrics, job_queue)
            elif job.type == "stop":
                break
            elif job.type == "wait":
                done_queue.put(None)
            else:
                raise KeyError('Received message of unknown type {}'.format(
                    job.type))

        env.close()
        done_queue.put(None)
    except KeyboardInterrupt:
        alf.get_env().close()
    except Exception as e:
        logging.exception(f'{mp.current_process().name} - {e}')


@common.mark_eval
def evaluate(env: AlfEnvironment,
             algorithm: RLAlgorithm,
             num_episodes: int,
             job_queue: Optional[PeekableQueue] = None
             ) -> List[alf.metrics.StepMetric]:
    """Perform one round of evaluation.

    Args:
        env: the environment
        algorithm: the training algorithm
        num_episodes: number of episodes to evaluate
        job_queue: This is only used when `eval()` is called from a worker
            process. If during the evaluation, the worker receives a "stop"
            job from the main process, it will stop the evaluation and
            return immediately.
    Returns:
        a list of metrics from the evaluation
    """
    batch_size = env.batch_size
    env.reset()
    env.sync_progress()
    time_step = common.get_initial_time_step(env)
    algorithm.eval()
    policy_state = algorithm.get_initial_predict_state(env.batch_size)
    trans_state = algorithm.get_initial_transform_state(env.batch_size)
    episodes_per_env = (num_episodes + batch_size - 1) // batch_size
    env_episodes = torch.zeros(batch_size, dtype=torch.int32)
    episodes = 0
    metrics = [
        alf.metrics.AverageReturnMetric(
            buffer_size=num_episodes, example_time_step=time_step),
        alf.metrics.AverageEpisodeLengthMetric(
            example_time_step=time_step, buffer_size=num_episodes),
        alf.metrics.AverageEnvInfoMetric(
            example_time_step=time_step, buffer_size=num_episodes),
        alf.metrics.AverageDiscountedReturnMetric(
            buffer_size=num_episodes, example_time_step=time_step),
        alf.metrics.EpisodicStartAverageDiscountedReturnMetric(
            example_time_step=time_step, buffer_size=num_episodes),
        alf.metrics.AverageRewardMetric(
            example_time_step=time_step, buffer_size=num_episodes),
    ]
    time_step = common.get_initial_time_step(env)
    while episodes < num_episodes:
        # For parallel play, we cannot naively pick the first finished `num_episodes`
        # episodes to estimate the average return (or other statistics) as it can be
        # biased towards short episodes. Instead, we stick to using the first
        # episodes_per_env episodes from each environment to calculate the
        # statistics and ignore the potentially extra episodes from each environment.
        invalid = env_episodes >= episodes_per_env
        # Force the step_type of the extra episodes to be StepType.FIRST so that
        # these time steps do not affect metrics as the metrics are only updated
        # at StepType.LAST. The metric computation uses cpu version of time_step.
        time_step.cpu().step_type[invalid] = StepType.FIRST

        next_time_step, policy_step, trans_state = policy_trainer._step(
            algorithm=algorithm,
            env=env,
            time_step=time_step,
            policy_state=policy_state,
            trans_state=trans_state,
            metrics=metrics)

        time_step.step_type[invalid] = StepType.FIRST

        for i in range(batch_size):
            if time_step.step_type[i] == StepType.LAST:
                env_episodes[i] += 1
                episodes += 1

        policy_state = policy_step.state
        time_step = next_time_step
        if job_queue is not None:
            job = job_queue.peek()
            if job is not None and job.type == "stop":
                logging.info("Received stop signal. Aborting evaluation.")
                return None

    env.reset()
    return metrics
