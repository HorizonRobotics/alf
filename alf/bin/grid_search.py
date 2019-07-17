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

import itertools
import json

from absl import app
from absl import flags
from absl import logging
import gin

# `pathos.multiprocessing` provides a consistent interface with std lib `multiprocessing`
# and it's more  flexible
from pathos import multiprocessing
from alf.bin.main import train_eval
from alf.utils import common

flags.DEFINE_string('search_config', None,
                    'Path to the grid search config file.')
FLAGS = flags.FLAGS
r"""Grid search.

To run grid search on ddpg for gym Pendulum:
```bash
cd ${PROJECT}/alf/examples;
python -m alf.bin.grid_search \
  --root_dir=~/tmp/ddpg_pendulum \
  --search_config=ddpg_grid_search.json \
  --gin_file=ddpg_pendulum.gin \
  --gin_param='create_environment.num_parallel_environments=8' \
  --alsologtostderr
```
"""


class GridSearchConfig(object):
    """ Grid Search Config

    Grid search config file should be json format
    For example:
    {
        "desc": "desc text",
        "max_worker_num": 8,
        "parameters": {
            "create_ddpg_algorithm.actor_learning_rate": [1e-3, 8e-4],
            "OneStepTDLoss.gamma":"(0.995, 0.99)",
            "param3_name": param3_value,
              ...
        }
        ...
    }
    `max_worker_num` is max number of parallel search worker processes
    `parameters` is a dict(param_name=param_value,) of configured search space .
    param_name is a gin configurable argument str and param_value must be an
    iterable python object or a str that can be evaluated to an iterable object.

    See `ddpg_grid_search.json` for an example.
    """

    def __init__(self, conf_file):
        """
        Args:
            conf_file (str): Path to the config file.
        """
        with open(conf_file) as f:
            param_keys = []
            param_values = []
            conf = json.loads(f.read())
            for key, value in conf['parameters'].items():
                if isinstance(value, str):
                    value = eval(value)
                param_keys.append(key)
                param_values.append(value)

        self._desc = conf['desc']
        self._param_keys = param_keys
        self._param_values = param_values
        self._max_worker_num = conf['max_worker_num']

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


class GridSearch(object):
    """Grid Search"""

    def __init__(self, conf_file):
        """Create GridSearch instance
        Args:
            conf_file (str): Path to the config file.
        """
        self._conf = GridSearchConfig(conf_file)

    def run(self):
        """Run trainings with all possible parameter combinations in configured space
        """

        processes = []
        param_keys = self._conf.param_keys
        param_values = self._conf.param_values
        max_worker_num = self._conf.max_worker_num

        process_pool = multiprocessing.Pool(
            processes=max_worker_num, maxtasksperchild=1)

        for values in itertools.product(*param_values):
            parameters = dict(zip(param_keys, values))
            root_dir = "%s/%d" % (FLAGS.root_dir, len(processes))
            process_pool.apply_async(
                func=self._worker,
                args=[root_dir, parameters],
                error_callback=lambda e: logging.error(e))

        process_pool.close()
        process_pool.join()

    def _worker(self, root_dir, parameters):
        logging.set_verbosity(logging.INFO)
        gin_file = common.get_gin_file()
        if FLAGS.gin_file:
            common.copy_gin_configs(root_dir, gin_file)
        gin.parse_config_files_and_bindings(gin_file, FLAGS.gin_param)
        with gin.unlock_config():
            gin.parse_config(['%s=%s' % (k, v) for k, v in parameters.items()])
        train_eval(root_dir)


def main(_):
    GridSearch(FLAGS.search_config).run()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    flags.mark_flag_as_required('root_dir')
    flags.mark_flag_as_required('search_config')
    app.run(main)
