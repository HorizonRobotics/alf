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
"""Summary related functions."""

import functools
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Callable

_summary_enabled = False

_default_writer: SummaryWriter = None

_global_counter = torch.tensor(0, dtype=torch.int64)

_scope_stack = ['']

_record_if_stack = [
    lambda: True,
]

_summary_writer_stack = [None]

# The default number of bins for histogram
_default_bins = 30


class scope(object):
    """A context manager for prefixing summary names.

    Example:
    ```
    with alf.summary.scope("root"):
        alf.summary.scalar("val", 1)    # tag is "root/val"
        with alf.summary.scope("train"):
            alf.summary.scalar("val", 1)    # tag is "root/train/val"
    ```
    """

    def __init__(self, name: str):
        """Create the context manager.

        Args:
            name (str): name of the scope
        """
        name.strip('/')
        self._name = name

    @property
    def name(self):
        """Get the name of the scope."""
        return self._name

    def __enter__(self):
        scope_name = _scope_stack[-1] + self._name + '/'
        _scope_stack.append(scope_name)
        return scope_name

    def __exit__(self, type, value, traceback):
        _scope_stack.pop()


def _summary_wrapper(summary_func):
    """Summary wrapper

    Wrapper summary function to reduce cost for data computation
    """

    @functools.wraps(summary_func)
    def wrapper(name, data, step=None, **kwargs):
        if should_record_summaries():
            if step is None:
                step = _global_counter
            name = _scope_stack[-1] + name
            summary_func(name, data, step, **kwargs)

    return wrapper


@_summary_wrapper
def text(name, data, step=None, walltime=None):
    """Add text data to summary.

    Note that the actual tag will be `name + "/text_summary"` because torch
    adds "/text_summary to tag. See
    https://github.com/pytorch/pytorch/blob/877ab3afe33eeaa797296d2794317b59e5ac90f4/torch/utils/tensorboard/summary.py#L477

    Args:
        name (str): Data identifier
        data (str): String to save
        step (int): Global step value to record. None for using get_global_counter()
        walltime (float): Optional override default walltime (time.time())
            seconds after epoch of event
    """
    _summary_writer_stack[-1].add_text(name, data, step, walltime=walltime)


@_summary_wrapper
def scalar(name, data, step=None, walltime=None):
    """Addd scalar data to summary.

    Note that data will be changed to float value (i.e. possible loss of
    precision). See
    https://github.com/pytorch/pytorch/blob/877ab3afe33eeaa797296d2794317b59e5ac90f4/torch/utils/tensorboard/summary.py#L175

    Args:
        name (str): Data identifier
        data (float): Value to save
        step (int): Global step value to record. None for using get_global_counter()
        walltime (float): Optional override default walltime (time.time())
            seconds after epoch of event
    """
    _summary_writer_stack[-1].add_scalar(name, data, step, walltime=walltime)


@_summary_wrapper
def histogram(name, data, step=None, bins=None, walltime=None, max_bins=None):
    """Add histogram to summary.

    Args:
        name (str): Data identifier
        data (Tensor | numpy.array | str/blobname): Values to build histogram
        step (int): Global step value to record. None for using get_global_counter()
        bins (int|str): Number of buckets or one of {‘tensorflow’,’auto’, ‘fd’, …}.
            This determines how the bins are made. You can find other options in:
            https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
        walltime (float): Optional override default walltime (time.time())
            seconds after epoch of event
    """
    if bins is None:
        bins = _default_bins
    _summary_writer_stack[-1].add_histogram(
        name, data, step, bins=bins, walltime=walltime, max_bins=max_bins)


def should_record_summaries():
    """Whether summary should be recorded.

    Returns:
        bool: False means that all calls to scalar(), text(), histogram() etc
            are not recorded.

    """
    return (_summary_writer_stack[-1] and is_summary_enabled()
            and _record_if_stack[-1]())


def get_global_counter():
    """Get the global counter

    Returns:
        the global int64 Tensor counter
    """
    return _global_counter


def reset_global_counter():
    """Reset the global counter to zero
    """
    _global_counter.data.fill_(0)


class record_if(object):
    """Context manager to set summary recording on or off according to `cond`."""

    def __init__(self, cond: Callable):
        """Create the context manager.

        Args:
            cond (Callable): a function which returns whether summary should be
                recorded.
        """
        self._cond = cond

    def __enter__(self):
        _record_if_stack.append(self._cond)

    def __exit__(self, type, value, traceback):
        _record_if_stack.pop()


def create_summary_writer(summary_dir, flush_secs=10, max_queue=10):
    """Ceates a SummaryWriter that will write out events to the event file.

    Args:
        summary_dir (str) – Save directory location.
        max_queue (int) – Size of the queue for pending events and summaries
            before one of the ‘add’ calls forces a flush to disk.
            Default is ten items.
        flush_secs (int) – How often, in seconds, to flush the pending events
            and summaries to disk. Default is every 10 seconds.
    Returns:
        SummaryWriter
    """
    return SummaryWriter(
        log_dir=summary_dir, flush_secs=flush_secs, max_queue=max_queue)


def set_default_writer(writer):
    """Set the default summary writer."""
    _summary_writer_stack[0] = writer


def enable_summary(flag=True):
    """Enable summary.

    Args:
        flag (bool): True to enable, False to disable
    """
    global _summary_enabled
    _summary_enabled = flag


def disable_summary():
    """Disable summary."""
    global _summary_enabled
    _summary_enabled = False


def is_summary_enabled():
    """Return whether summary is enabled."""
    return _summary_enabled


class push_summary_writer(object):
    def __init__(self, writer):
        self._writer = writer

    def __enter__(self):
        _summary_writer_stack.append(self._writer)

    def __exit__(self, type, value, traceback):
        _summary_writer_stack.pop()
