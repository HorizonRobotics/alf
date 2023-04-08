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

For using ALF conf, replace "--gin_file" with "--conf" and "--gin_param" with
"--conf_param".
"""

from absl import app
from absl import flags
from absl import logging
import copy
import gin
import itertools
import json
from multiprocessing import Queue, Manager
import os
# `pathos.multiprocessing` provides a consistent interface with std lib `multiprocessing`
# and it's more flexible
from pathos import multiprocessing
import pathlib
import random
import re
import subprocess
import sys
import time
import torch
import traceback
from typing import Iterable
import unicodedata

import alf
from alf.bin.train import _define_flags as _train_define_flags
from alf.bin.train import _train
from alf.utils import common


def _slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Also strip leading and trailing whitespace, dashes,
    and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode(
            'ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value)
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def _define_flags():
    _train_define_flags()
    flags.DEFINE_string('search_config', None,
                        'Path to the grid search config file.')
    flags.DEFINE_bool(
        'snapshot_gridsearch_activated', False,
        'Whether a snapshot has been generated for grid search. (ONLY '
        'change this flag manually if you know what you are doing!')


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
            search space. Each key ``param_name`` is a gin/alf configurable argument
            string and the paired ``param_value`` must be an iterable python object
            or a ``str`` that can be evaluated to an iterable object. When
            ``parameters`` is empty, the original conf file won't be changed.

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
                           max_len=50):
        """Generate a run name by writing abbr parameter key-value pairs in it,
        for an easy comparison between different search runs without going
        into Tensorboard 'text' for run details.

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
                return _slugify('_'.join([w[:len_per_word] for w in words]))

            if isinstance(x, str):
                tokens = x.replace("/", "_").split(".")
                tokens = [_initials(t) for t in tokens]
                return ".".join(tokens)
            else:
                return _abbr_single(str(x), l)

        def _abbr(x, l):
            if isinstance(x, Iterable) and not isinstance(x, str):
                strs = []
                for key, val in x.items():
                    strs.append("%s=%s" % (_abbr(key, l), _abbr(val, l)))
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

        # This ``conf_file`` will be retrieved from ``root_dir``
        assert FLAGS.conf is None and FLAGS.gin_file is None

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
                root_dir = common.abs_path(root_dir)
                process_pool.apply_async(
                    func=self._worker,
                    args=[root_dir, parameters, device_queue],
                    error_callback=lambda e: logging.error(e))

        process_pool.close()
        process_pool.join()

        # Remove the alf snapshot so that it won't waste disk space (we only
        # need snapshots under each search run dir).
        alf_repo = common.abs_path(os.path.join(FLAGS.root_dir, "alf"))
        os.system("rm -rf %s*" % alf_repo)

    def _worker(self, root_dir, parameters, device_queue):
        # sleep for random seconds to avoid crowded launching
        try:
            time.sleep(random.uniform(0, 3))

            conf_file = common.get_conf_file()

            # We still need to keep a snapshot of ALF repo at ``<root_dir>``
            # for playing individual searching job later
            os.system(f"mkdir -p {root_dir}; "
                      f"cp {FLAGS.root_dir}/*.tar.gz {root_dir}/")

            device = device_queue.get()
            if self._conf.use_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""  # run on cpu

            if torch.cuda.is_available():
                alf.set_default_device("cuda")
            logging.set_verbosity(logging.INFO)

            logging.info("Search parameters %s" % parameters)

            if conf_file.endswith('.gin'):
                common.parse_conf_file(conf_file)
                # re-bind gin conf params
                with gin.unlock_config():
                    gin.parse_config(
                        ['%s=%s' % (k, v) for k, v in parameters.items()])
                    gin.parse_config(
                        "TrainerConfig.confirm_checkpoint_upon_crash=False")
            else:
                # need to first pre_config before parsing the conf file
                confs = copy.copy(parameters)
                confs.update({
                    'TrainerConfig.confirm_checkpoint_upon_crash': False
                })
                alf.pre_config(confs)
                common.parse_conf_file(conf_file)

            # init env random seed differently for each worker
            alf.get_env()
            # current only support non-distributed training for each individual
            # job of grid search
            _train(root_dir)

            device_queue.put(device)
        except Exception as e:
            logging.info(traceback.format_exc())
            raise e
        finally:
            alf.close_env()


def search():
    FLAGS.alsologtostderr = True
    logging.get_absl_handler().use_absl_log_file(
        log_dir=os.path.expanduser(FLAGS.root_dir))
    GridSearch(FLAGS.search_config).run()


def launch_snapshot_gridsearch():
    """This gridsearch function uses a cached ALF snapshot to generate grid-search
    runs. Because some search jobs might stay in the queue until resources are
    available, the cache is used to make sure that when a search job is launched,
    it's actually using the right ALF version.
    """
    root_dir = common.abs_path(FLAGS.root_dir)

    # write the current conf file as
    # ``<root_dir>/alf_config.py`` or ``<root_dir>/configured.gin``
    conf_file = common.get_conf_file()
    if conf_file.endswith('.gin'):
        # for gin, we need to parse it first. Otherwise, configured.gin will be
        # empty
        common.parse_conf_file(conf_file)
    common.write_config(root_dir)

    # generate a snapshot of ALF repo as ``<root_dir>/alf``
    common.generate_alf_snapshot(common.alf_root(), conf_file, root_dir)

    # point the grid search to the snapshot paths, in case the code has been
    # changed when launching a job in the queue
    env_vars = common.get_alf_snapshot_env_vars(root_dir)

    # remove the conf file option since we will retrieve it from ``root_dir``
    flags = []
    skip_flag = False
    for f in sys.argv[1:]:
        if not skip_flag:
            if f in ('--conf', '--gin_file'):
                skip_flag = True  # skip the next flag which is the file path
            elif f.startswith(('--conf=', '--gin_file=')):
                pass
            else:
                flags.append(f)
        else:
            skip_flag = False
    flags.append('--snapshot_gridsearch_activated')

    args = ['python', '-m', 'alf.bin.grid_search'] + flags

    try:
        subprocess.check_call(
            " ".join(args),
            env=env_vars,
            stdout=sys.stdout,
            stderr=sys.stdout,
            shell=True)
    except subprocess.CalledProcessError:
        # No need to output anything
        pass


def main(_):
    if FLAGS.snapshot_gridsearch_activated:
        search()
    else:
        launch_snapshot_gridsearch()


if __name__ == '__main__':
    _define_flags()
    logging.set_verbosity(logging.INFO)
    flags.mark_flag_as_required('root_dir')
    flags.mark_flag_as_required('search_config')
    app.run(main)
