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

# TODO: These file will be filled with actual stuff
import torch


def text(name, data, step=None):
    pass


def scalar(name, data, step=None):
    pass


def histogram(name, data, step=None):
    pass


def should_record_summaries():
    return True


_global_counter = None


def get_global_counter():
    """Get the global counter.

    Returns:
        the global int64 Tensor counter
    """
    global _global_counter
    if _global_counter is None:
        _global_counter = torch.tensor(0, dtype=torch.int64)
    return _global_counter


class record_if(object):
    def __init__(self, cond):
        self._cond = cond

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass


def create_file_writer(summary_dir, flush_millis, max_queue):
    pass


def set_default_writer(write):
    pass
