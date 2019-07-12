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

import multiprocessing
import itertools
import json

from absl import app
from absl import flags
from absl import logging
import gin
from alf.examples.main import train_eval
from alf.utils import common

flags.DEFINE_string('search_config', None,
                    'Path to the grid search config file.')
flags.DEFINE_integer('max_worker_num', 0,
                     'Max number of parallel search processes.')
FLAGS = flags.FLAGS
r"""Grid search.

To run grid search on ddpg for gym Pendulum:
```bash
python grid_search.py \
  --root_dir=~/tmp/ddpg_pendulum \
  --search_config=ddpg_grid_search.json \
  --max_worker_num=8 \
  --gin_file=ddpg_pendulum.gin \
  --gin_param='create_environment.num_parallel_environments=8' \
  --alsologtostderr
```
"""


class GridSearch(object):
    """Grid Search"""

    def __init__(self, conf_file):
        """Create GridSearch instance

        Grid search config file must be json format, and element `parameters` is required
        `parameters` is a dict(key=value,) of configured search space .
        key must be gin configurable argument str and value must be an iterable python object
        or a str that can be evaluated to an iterable object .
        see `ddpg_grid_search.json` for a example.

        Args:
            conf_file (str): Path to the config file.
        """
        self._conf_file = conf_file

    @staticmethod
    def parse_config(conf_file):
        with open(conf_file) as f:
            keys = []
            values = []
            conf = json.loads(f.read())
            for key, value in conf['parameters'].items():
                if isinstance(value, str):
                    value = eval(value)
                keys.append(key)
                values.append(value)
        return keys, values

    def run(self, max_worker_num=0):
        """
        Args:
            max_worker_num (int): Max number of parallel search worker processes,
                0 for unlimited.
        """
        processes = []
        para_keys, para_values = GridSearch.parse_config(self._conf_file)
        # `worker_queue` is just a status queue to control the number for working process.
        # And to simplify, process pool is not used here, just spawn process for new
        # search job.
        worker_queue = multiprocessing.Queue(maxsize=max_worker_num)
        for values in itertools.product(*para_values):
            worker_queue.put(None, block=True)
            parameters = dict(zip(para_keys, values))
            root_dir = "%s/%d" % (FLAGS.root_dir, len(processes))
            process = multiprocessing.Process(
                target=self._worker, args=(root_dir, parameters, worker_queue))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()

    def _worker(self, root_dir, parameters, worker_queue):
        logging.set_verbosity(logging.INFO)
        gin_file = common.get_gin_file()
        if FLAGS.gin_file:
            common.copy_gin_configs(root_dir, gin_file)
        gin.parse_config_files_and_bindings(gin_file, FLAGS.gin_param)
        with gin.unlock_config():
            gin.parse_config(['%s=%s' % (k, v) for k, v in parameters.items()])
        train_eval(root_dir)
        worker_queue.get()


def main(_):
    GridSearch(FLAGS.search_config).run(max_worker_num=FLAGS.max_worker_num)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    flags.mark_flag_as_required('root_dir')
    flags.mark_flag_as_required('search_config')
    app.run(main)
