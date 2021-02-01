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
r"""Grid search.

To run grid search on DDPG for training gym `Pendulum`:

.. code-block:: bash

    cd ${PROJECT}/alf/examples;
    python -m alf.bin.grid_search \
    --root_dir=~/tmp/ddpg_pendulum \
    --search_config=ddpg_grid_search.json \
    --gin_file=ddpg_pendulum.gin \
    --gin_param='create_environment.num_parallel_environments=8' \
    --alsologtostderr

"""

from absl import app
from absl import flags
from absl import logging
from collections import Iterable
import gin
import itertools
import json
from multiprocessing import Queue, Manager
import os
# `pathos.multiprocessing` provides a consistent interface with std lib `multiprocessing`
# and it's more flexible
from pathos import multiprocessing
import random
import time
import torch
import traceback

import alf
from alf.bin.train import train_eval
from alf.utils import common

flags.DEFINE_string('search_config', None,
                    'Path to the grid search config file.')

FLAGS = flags.FLAGS


class GridSearchConfig(object):
    """A grid search config file should be in the json format. For example:

    .. code-block:: javascript

        {
            "desc": "desc text",
            "use_gpu": true,
            "gpus": [0, 1],
            "max_worker_num": 8,
            "repeats": 3,
            "parameters": {
                "ac/Adam.learning_rate": [1e-3, 8e-4],
                "OneStepTDLoss.gamma":"(0.995, 0.99)",
                "param_name3": param_value3,
                ...
            }
            ...
        }

    Supported keys in a json file are:

    Args:
        desc (str): a description sentence for this json file.
        use_gpu (bool): If True, then the scheduling will only put jobs on devices
            numbered ``gpus``.
        gpus (list[int]): a list of GPU device ids. If ``use_gpu`` is False,
            this list will be ignored.
        max_worker_num (int): the max number of parallel worker processes at
            any moment. ``max_worker_num`` jobs will be evenly divided among the
            devices specified by the ``gpus`` list. It's the user's responsibility
            to make sure that each device's resource is enough.
        repeats (int): each parameter combination will be repeated for so many
            times, with different random seeds.
        parameters (dict): a ``dict(param_name=param_value,)`` of the configured
            search space. Each key ``param_name`` is a gin configurable argument
            string and the paired ``param_value`` must be an iterable python object
            or a ``str`` that can be evaluated to an iterable object. When
            ``parameters`` is empty, the original gin conf won't be changed.

    See ``alf/examples/ddpg_grid_search.json`` for an example.
    """

    _all_keys_ = [
        "desc", "comment", "use_gpu", "gpus", "max_worker_num", "repeats",
        "parameters"
    ]

    def __init__(self, conf_file):
        """
        Args:
            conf_file (str): Path to the config file.
        """
        with open(conf_file) as f:
            param_keys = []
            param_values = []
            conf = json.loads(f.read())
            parameters = conf.get("parameters", dict())
            for key, value in parameters.items():
                if isinstance(value, str):
                    value = eval(value)
                param_keys.append(key)
                param_values.append(value)

        # check conf keys
        for k in conf.keys():
            assert k in self._all_keys_, "Invalid conf key: %s" % k

        self._desc = conf.get('desc', "Grid search")
        self._param_keys = param_keys
        self._param_values = param_values
        self._max_worker_num = conf.get('max_worker_num', 1)
        self._use_gpu = conf.get("use_gpu", False)
        self._gpus = conf.get("gpus", [0])
        self._repeats = conf.get("repeats", 1)

        self._check_worker_options()

    def _check_worker_options(self):
        if not isinstance(self._gpus, Iterable):
            self._gpus = [self._gpus]
        for gpu in self._gpus:
            assert isinstance(gpu, int), "gpu device must be an integer"
        assert isinstance(self._max_worker_num, int)
        assert isinstance(self._repeats, int)
        assert isinstance(self._use_gpu, bool)

    @property
    def desc(self):
        return self._desc

    @property
    def param_keys(self):
        return self._param_keys

    @property
    def param_values(self):
        return self._param_values

    @property
    def max_worker_num(self):
        return self._max_worker_num

    @property
    def use_gpu(self):
        return self._use_gpu

    @property
    def gpus(self):
        return self._gpus

    @property
    def repeats(self):
        return self._repeats


class GridSearch(object):
    """Grid Search."""

    def __init__(self, conf_file):
        """
        Args:
            conf_file (str): Path to the config file.
        """
        self._conf = GridSearchConfig(conf_file)

    def _init_device_queue(self, max_worker_num):
        m = Manager()
        device_queue = m.Queue()
        num_gpus = len(self._conf.gpus)
        for i in range(max_worker_num):
            idx = i % num_gpus
            device_queue.put(self._conf.gpus[idx])
        return device_queue

    def _generate_run_name(self,
                           parameters,
                           id,
                           repeat,
                           token_len=20,
                           max_len=255):
        """Generate a run name by writing abbr parameter key-value pairs in it,
        for an easy curve comparison between different search runs without going
        into the gin texts.

        Args:
            parameters (dict): a dictionary of parameter configurations
            id (int): an integer id of the run
            repeat (int): an integer id of the repeats of the run
            token_len (int): truncate each token for so many chars
            max_len (int): the maximal length of the generated name; make sure
                that this value won't exceed the max allowed filename length in
                the OS

        Returns:
            str: a string with parameters abbr encoded
        """

        def _abbr_single(x, l):
            def _initials(t):
                words = [w for w in t.split('_') if w]
                len_per_word = max(l // len(words), 1)
                return '_'.join([w[:len_per_word] for w in words])

            if isinstance(x, str):
                tokens = x.replace("/", "_").split(".")
                tokens = [_initials(t) for t in tokens]
                return ".".join(tokens)
            else:
                return str(x)

        def _abbr(x, l):
            if isinstance(x, Iterable) and not isinstance(x, str):
                strs = []
                for key in x:
                    try:
                        val = x.get(key)
                        strs.append("%s=%s" % (_abbr(key, l), _abbr(val, l)))
                    except:
                        strs.append("%s" % _abbr(key, l))
                return "+".join(strs)
            else:
                return _abbr_single(x, l)

        def _generate_name(max_token_len):
            name = "%04dr%d" % (id, repeat)
            abbr = _abbr(parameters, max_token_len)
            if abbr:
                name += "+" + abbr
            return name

        # first try not truncating words
        name = _generate_name(max_token_len=max_len)
        if len(name) > max_len:
            # If this regenerated name is still over ``max_len``, it will get
            # hard truncated
            name = _generate_name(max_token_len=token_len)
        return name[:max_len]

    def run(self):
        """Run trainings with all possible parameter combinations in
        the configured space.
        """

        # parsing gin configuration here to make all jobs have same copy
        #   of base configuration (gin file may be changed occasionally)
        gin_file = common.get_gin_file()
        gin.parse_config_files_and_bindings(gin_file, FLAGS.gin_param)

        param_keys = self._conf.param_keys
        param_values = self._conf.param_values
        max_worker_num = self._conf.max_worker_num

        process_pool = multiprocessing.Pool(
            processes=max_worker_num, maxtasksperchild=1)
        device_queue = self._init_device_queue(max_worker_num)

        for repeat in range(self._conf.repeats):
            for task_count, values in enumerate(
                    itertools.product(*param_values)):
                parameters = dict(zip(param_keys, values))
                root_dir = "%s/%s" % (FLAGS.root_dir,
                                      self._generate_run_name(
                                          parameters, task_count, repeat))
                process_pool.apply_async(
                    func=self._worker,
                    args=[root_dir, parameters, device_queue],
                    error_callback=lambda e: logging.error(e))

        process_pool.close()
        process_pool.join()

    def _worker(self, root_dir, parameters, device_queue):
        # sleep for random seconds to avoid crowded launching
        try:
            time.sleep(random.uniform(0, 3))

            device = device_queue.get()
            if self._conf.use_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""  # run on cpu

            if torch.cuda.is_available():
                alf.set_default_device("cuda")
            logging.set_verbosity(logging.INFO)

            logging.info("Search parameters %s" % parameters)
            with gin.unlock_config():
                gin.parse_config(
                    ['%s=%s' % (k, v) for k, v in parameters.items()])
                gin.parse_config(
                    "TrainerConfig.confirm_checkpoint_upon_crash=False")
            train_eval(FLAGS.ml_type, root_dir)

            device_queue.put(device)
        except Exception as e:
            logging.info(traceback.format_exc())
            raise e


def main(_):
    FLAGS.alsologtostderr = True
    GridSearch(FLAGS.search_config).run()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    flags.mark_flag_as_required('root_dir')
    flags.mark_flag_as_required('search_config')
    app.run(main)
