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
from queue import Empty
import torch.multiprocessing as mp
import time
import torch
from typing import Dict, List

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.data_transformer import create_data_transformer
from alf.utils import common
from collections import namedtuple

UnrollResult = namedtuple(
    "UnrollResult",
    ["time_step", "policy_step", "policy_state", "env_step_time", "step_time"])

UnrollJob = namedtuple(
    "UnrollJob", ["type", "step_metrics", "global_counter", "state_dict"],
    defaults=[None] * 4)


class AsyncUnroller(object):
    """A helper class for unroll asynchronously.

    The asynchronous unroll is performed in a different process. The unroll results
    are transmitted to the main process through a Queue. The main process should
    call ``gather_unroll_results()`` to retrieve the unroll results. Since the
    unroll process has its own algorithm parameters, the main process needs to call
    ``update_parameters()`` to update the parameters for the unroll process
    periodically. Once the main process finishes, it should call close() to
    release the resources.

    The following settings in ``TrainerConfig`` are related to the functionality
    of ``AsyncUnroller``: unroll_length, async_unroll, max_unroll_length,
    unroll_queue_size, unroll_step_interval. See algorithms.config.py for their
    documentation.

    TODO: redirect the log and summary to the training process. Currently,
    all the logs are written to a different log file and summary during
    rollout_step() is not enabled.

    Args:
        algorithm: the root RL algorithm
        unroll_queue_size: the size of the queue for transmitting the unroll results
            to the main process
        root_dir: directory for saving summary and checkpoints
        conf_file: config file name
    """

    def __init__(self, algorithm, config: TrainerConfig):
        # The following line is needed for avoiding
        # "RuntimeError: unable to open shared memory object"
        # See https://github.com/facebookresearch/maskrcnn-benchmark/issues/103#issuecomment-785815218
        mp.set_sharing_strategy('file_system')
        if config.conf_file.endswith('.gin'):
            assert not self._async, "async_unroll is not supported for gin_file"
        ctx = mp.get_context('spawn')
        self._job_queue = ctx.Queue()
        self._done_queue = ctx.Queue()
        self._result_queue = ctx.Queue(config.unroll_queue_size)
        pre_configs = dict(alf.get_handled_pre_configs())
        self._worker = ctx.Process(
            target=_worker,
            args=(self._job_queue, self._done_queue, self._result_queue,
                  config.conf_file, pre_configs, config.root_dir))
        self._worker.start()
        self.update_parameter(algorithm)
        self._closed = False

    def get_queue_size(self) -> int:
        return self._result_queue.qsize()

    def gather_unroll_results(self, unroll_length: int,
                              max_unroll_length: int) -> List[UnrollResult]:
        """Gather the unroll results:

        Args:
            unroll_length: the desired unroll length. If is 0, any length up to
                ``max_unroll_length`` is possible (including zero length) depending
                on how much data is in the queue.
            max_unroll_length: maximal length of unroll results. This is only
                used if ``unroll_length`` is 0.
        Returns:
            A list of ``UnrollResult``
        """
        unroll_results = []
        if unroll_length > 0:
            for i in range(unroll_length):
                unroll_results.append(self._result_queue.get())
        else:
            while not self._result_queue.empty() and len(
                    unroll_results) < max_unroll_length:
                unroll_results.append(self._result_queue.get())
        return unroll_results

    def update_parameter(self, algorithm):
        """Update the the model parameter for unroll.

        Args:
            algorithm (RLAlgorithm): the root RL algorithm
        """
        step_metrics = algorithm.get_step_metrics()
        step_metrics = dict((m.name, int(m.result())) for m in step_metrics)
        job = UnrollJob(
            type="update_parameter",
            step_metrics=step_metrics,
            global_counter=int(alf.summary.get_global_counter()),
            state_dict=algorithm.state_dict())
        self._job_queue.put(job)
        self._done_queue.get()

    def close(self):
        """Close the unroller and release resources."""
        if self._closed:
            return
        job = UnrollJob(type="stop")
        self._job_queue.put(job)
        self._done_queue.get()
        self._worker.join()
        self._closed = True


FLAGS = flags.FLAGS


def _worker(job_queue: mp.Queue, done_queue: mp.Queue, result_queue: mp.Queue,
            conf_file: str, pre_configs: Dict, root_dir: str):
    from alf.trainers import policy_trainer

    def _update_parameter(algorithm, job):
        # Some algorithms use scheduler depending on the global counter
        # or the training progress. So we make sure they are same as
        # the training process.
        alf.summary.set_global_counter(job.global_counter)
        env_steps = job.step_metrics["EnvironmentSteps"]
        policy_trainer.Trainer.get_trainer_progress().update(
            job.global_counter, env_steps)
        algorithm.load_state_dict(job.state_dict)
        done_queue.put(None)

    def _process_job(job):
        # return True if stop unroll
        if job.type == "update_parameter":
            _update_parameter(algorithm, job)
            return False
        elif job.type == "stop":
            return True
        else:
            raise KeyError('Received message of unknown type {}'.format(
                job.type))

    try:
        logging.set_verbosity(logging.INFO)
        logging.get_absl_handler().use_absl_log_file(log_dir=root_dir)
        logging.use_absl_handler()
        if torch.cuda.is_available():
            alf.set_default_device("cuda")
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
        observation_spec = data_transformer.transformed_observation_spec

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

        algorithm.eval()
        policy_state = algorithm.get_initial_rollout_state(env.batch_size)
        trans_state = algorithm.get_initial_transform_state(env.batch_size)
        initial_state = algorithm.get_initial_rollout_state(env.batch_size)
        time_step = common.get_initial_time_step(env)

        job = job_queue.get(block=True)
        assert job.type == "update_parameter"
        _update_parameter(algorithm, job)

        remaining = 0
        step_time = 0
        t = time.time()
        while True:
            policy_state = common.reset_state_if_necessary(
                policy_state, initial_state, time_step.is_first())
            transformed_time_step, trans_state = algorithm.transform_timestep(
                time_step, trans_state)
            policy_step = algorithm.rollout_step(transformed_time_step,
                                                 policy_state)

            policy_step = common.detach(policy_step)
            action = policy_step.output
            t0 = time.time()
            next_time_step = env.step(action)
            t1 = time.time()
            env_step_time = t1 - t0

            # note that the step_time is actually the step_time for the previous
            # step. It is used for informational purpose. When unroll_step_interval
            # is specified, it is important to monitor the actual step_time to
            # make sure it is around unroll_step_interval.
            unroll_result = UnrollResult(
                time_step=time_step,
                policy_step=policy_step,
                policy_state=policy_state,
                env_step_time=env_step_time,
                step_time=step_time)

            stopped = False
            # If result_queue is full, result_queue.put() will block, which can
            # cause deadlock if the main process is trying to update_parameter
            # or stop at the same time. So we need to periodically check job queue
            # if the result_queue is full.
            while result_queue.full():
                if not job_queue.empty():
                    job = job_queue.get()
                    stopped = _process_job(job)
                    if stopped:
                        break
                else:
                    time.sleep(0.1)
            if stopped:
                break

            result_queue.put(unroll_result)

            policy_state = policy_step.state
            time_step = next_time_step

            t1 = time.time()
            step_time = t1 - t
            remaining = config.unroll_step_interval - step_time
            try:
                if remaining > 0:
                    job = job_queue.get(block=True, timeout=remaining)
                else:
                    job = job_queue.get(block=False)
            except Empty:
                job = None
            if job is not None:
                if _process_job(job):
                    break
            t1 = time.time()
            step_time = t1 - t
            remaining = config.unroll_step_interval - step_time
            if remaining > 0:
                time.sleep(remaining)
                t = t1 + remaining
            else:
                t = t1

        env.close()
        done_queue.put(None)
        # Need this to quit the process. Otherwise, the process may wait to join
        # a background thread of the queue for ever.
        result_queue.cancel_join_thread()

    except Exception as e:
        logging.exception(f'{mp.current_process().name} - {e}')
