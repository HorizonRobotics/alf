# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Union
from alf.utils.schedulers import update_progress

try:
    # If tensorflow has been installed, pytorch might use tensorflow's
    # tensorboard. In this case, gfile needs to be redirected if embedding
    # projector is to be used.
    # https://github.com/pytorch/pytorch/issues/30966#issuecomment-582747929
    import tensorflow as tf
    import tensorboard as tb
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
except:
    pass

_summary_enabled = False

_summarize_output = False

_default_writer: SummaryWriter = None

_global_counter = np.array(0, dtype=np.int64)

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


_SUMMARY_DATA_BUFFER = {}


def _summary_wrapper(summary_func):
    """Summary wrapper

    Wrapper summary function to reduce cost for data computation
    """

    @functools.wraps(summary_func)
    def wrapper(name,
                data,
                average_over_summary_interval=False,
                step=None,
                **kwargs):
        """
        Args:
            average_over_summary_interval: if True, the average value of data during a
                summary interval will be written to summary. If data is None,
                it will be ignored for calculating the average. Note that providing
                a "None" value for data is different from not calling the summary
                function at all. A "None" value for data will cause the summary
                to be generated if ``should_record_summaries()`` returns True
                at the moment.
        """
        if average_over_summary_interval:
            if isinstance(data, torch.Tensor):
                data = data.detach()
            if name.startswith('/'):
                name = name[1:]
            else:
                name = _scope_stack[-1] + name
            if data is not None:
                if name in _SUMMARY_DATA_BUFFER:
                    data_sum, counter = _SUMMARY_DATA_BUFFER[name]
                    _SUMMARY_DATA_BUFFER[name] = data_sum + data, counter + 1
                else:
                    _SUMMARY_DATA_BUFFER[name] = data, 1
            if should_record_summaries() and name in _SUMMARY_DATA_BUFFER:
                data_sum, counter = _SUMMARY_DATA_BUFFER[name]
                del _SUMMARY_DATA_BUFFER[name]
                data = data_sum / counter
                if step is None:
                    step = _global_counter
                summary_func(name, data, step, **kwargs)
        else:
            if should_record_summaries():
                if isinstance(data, torch.Tensor):
                    data = data.detach()
                if step is None:
                    step = _global_counter
                if name.startswith('/'):
                    name = name[1:]
                else:
                    name = _scope_stack[-1] + name
                summary_func(name, data, step, **kwargs)

    return wrapper


def scope_name():
    """Get the full name of the current summary scope."""
    return _scope_stack[-1]


@_summary_wrapper
def images(name, data, step=None, dataformat='NCHW', walltime=None):
    """Add image data to summary.

    Args:
        name (str): Data identifier
        data (Tensor | numpy.array): image data
        step (int): Global step value to record. None for using ``get_global_counter()``
        dataformat (str): one of ('NCHW', 'NHWC', 'CHW', 'HWC', 'HW', 'WH')
        walltime (float): Optional override default walltime (time.time())
            seconds after epoch of event
    """
    _summary_writer_stack[-1].add_images(
        name, data, step, walltime=walltime, dataformats=dataformat)


@_summary_wrapper
def video(name: str,
          data: Union[np.ndarray, torch.Tensor],
          step: int = None,
          fps: int = 4,
          walltime: float = None):
    """Add video data to summary.

    Args:
        name: data identifier
        data: a video tensor or array of shape ``[B,T,C,H,W]``. The values should
            lie in [0, 255] for type uint8 or [0, 1] for type float.
        step: global step value to record
        fps: frames per second
        walltime: Optional override default walltime (time.time())
            seconds after epoch of event
    """
    _summary_writer_stack[-1].add_video(
        name, data, step, fps=fps, walltime=walltime)


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
    """Add scalar data to summary.

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


@_summary_wrapper
def embedding(name, data, step=None, class_labels=None, label_imgs=None):
    """Add embeddings to summary. The potentially high-dimensional embeddings
    will be projected down to either 2D or 3D for visualization, with several
    projection techniques to choose from in Tensorboard.

    The visualized embeddings can be seen in the "PROJECTOR" page of Tensorboard.

    Note: if this function is called multiple times, on the page there will be
    multiple visualizations, each for every call.

    Args:
        name (str): data identifier
        data (Tensor | numpy.array): a matrix of shape ``[N, D]``, where ``D``
            is the dimensionality of the embedding.
        step (int): global step value to record. None for using
            ``get_global_counter()``.
        class_labels (list[str]): an optional list of class labels of length
            ``N`` can be provided, where each label corresponds to an embedding.
        label_imgs (Tensor): an optional tensor of shape ``[N, C, H, W]``. Each
            label img corresponds to an embedding. Use this if you want to
            associate each embedding with an image for visualization.
    """
    _summary_writer_stack[-1].add_embedding(
        tag=name,
        mat=data,
        metadata=class_labels,
        label_img=label_imgs,
        global_step=step)


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
    """Reset the global counter to zero."""
    _global_counter.fill(0)
    update_progress("global_counter", 0)


def increment_global_counter():
    global _global_counter
    _global_counter += 1
    update_progress("global_counter", _global_counter)


def set_global_counter(counter):
    global _global_counter
    _global_counter.fill(counter)
    update_progress("global_counter", counter)


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
    """Creates a SummaryWriter that will write out events to the event file.

    Args:
        summary_dir (str) – Save directory location.
        flush_secs (int) – How often, in seconds, to flush the pending events
            and summaries to disk. Default is every 10 seconds.
        max_queue (int) – Size of the queue for pending events and summaries
            before one of the ‘add’ calls forces a flush to disk.
            Default is ten items.
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


def should_summarize_output(flag=None):
    """Get or set summarize output flag.

    Args:
        flag (bool or None): when provided, sets the flag, otherwise, return
            the stored _summarize_output flag.
    Returns:
        bool for getter or None for setter.
    """
    global _summarize_output
    if flag is None:
        return _summarize_output and should_record_summaries()
    else:
        _summarize_output = bool(flag)


class push_summary_writer(object):
    def __init__(self, writer):
        self._writer = writer

    def __enter__(self):
        _summary_writer_stack.append(self._writer)

    def __exit__(self, type, value, traceback):
        _summary_writer_stack.pop()


def enter_summary_scope(method):
    """A decorator to run the wrapped method in a new summary scope.

    The class the method belongs to must have attribute '_name' and it
    will be used as the name of the summary scope.

    Instead of using ``with alf.summary.scope(self._name):`` inside a class method,
    we can use ``@alf.summary.enter_summary_scope`` to decorate the method to
    have the benefit of cleaner code.
    """

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        # The first argument to the method is going to be ``self``, i.e. the
        # instance that the method belongs to.
        assert hasattr(self,
                       '_name'), "self is expected to have attribute '_name'"
        scope_name = _scope_stack[-1] + self._name + '/'
        _scope_stack.append(scope_name)
        ret = method(self, *args, **kwargs)
        _scope_stack.pop()
        return ret

    return wrapped


class EnsureSummary(object):
    """Ensure summaries are generated in an infrequent code block.

    Sometime, a code block runs infrequently or with different frequency compared
    to the summary_interval. This can lead to the problem that the summaries in
    this code block are not generated or generated rarely. This class is a helper
    to solve this problem.

    .. code-block:: python

        # initialization. For example, in __init__
        self.ensure_summary = EnsureSummary()

        # Add the following line at somewhere where it can be reached at very global step
        self.ensure_summary.tick()

        # Run the infrequent code block in the ensure_summary context:
        with self.ensure_summary:
            # the infrequent code block

    """

    def __init__(self):
        self._need_to_summarize = False

    def tick(self):
        if should_record_summaries():
            self._need_to_summarize = True

    def __enter__(self):
        _record_if_stack.append(lambda: self._need_to_summarize)

    def __exit__(self, type, value, traceback):
        _record_if_stack.pop()
        self._need_to_summarize = False
