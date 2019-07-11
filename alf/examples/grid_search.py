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

flags.DEFINE_string('search_file', None,
                    'Path to the grid search config file.')
FLAGS = flags.FLAGS
r"""Grid search.

To run grid search on ddpg for gym Pendulum:
```bash
python grid_search.py \
  --root_dir=~/tmp/ddpg_pendulum \
  --search_file=ddpg_grid_search.json \
  --gin_file=ddpg_pendulum.gin \
  --gin_param='create_environment.num_parallel_environments=8' \
  --alsologtostderr
```
"""


class GridSearch(object):
    """Grid Search"""

    def __init__(self, conf_file):
        """
        Args:
            conf_file (str): Path to the config file. Config file must be json format,
                and `parameters` is required. Parameters consist of key and iterable values.
                see `ddpg_grid_search.json` for a good example
                for more details.
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

    def run(self):
        processes = []
        para_keys, para_values = GridSearch.parse_config(self._conf_file)
        for values in itertools.product(*para_values):
            parameters = dict(zip(para_keys, values))
            root_dir = "%s_%d" % (FLAGS.root_dir, len(processes))
            process = multiprocessing.Process(
                target=self._worker, args=(root_dir, parameters))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()

    def _worker(self, root_dir, parameters):
        logging.set_verbosity(logging.INFO)
        gin_file = common.get_gin_file()
        if FLAGS.gin_file:
            common.copy_gin_configs(root_dir, gin_file)
        gin.parse_config_files_and_bindings(gin_file, FLAGS.gin_param)
        with gin.unlock_config():
            gin.parse_config(['%s=%s' % (k, v) for k, v in parameters.items()])
        train_eval(root_dir + "/train")


def main(_):
    GridSearch(FLAGS.search_file).run()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    flags.mark_flag_as_required('root_dir')
    flags.mark_flag_as_required('search_file')
    app.run(main)
