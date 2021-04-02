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
"""Coordinate asynchronous training process termination on request."""

from absl import logging
import contextlib
import ctypes
import multiprocessing as mp
from multiprocessing import Event, Lock, Value
import six
import sys
import time

# Adapted from tensorflow/python/training/coordinator.py
# to use python multiprocessing.
#
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


class Coordinator(object):
    """A coordinator for processes.

    This class implements a simple mechanism to coordinate the termination of a
    set of processes.

    .. code-block:: python

        with coord.stop_on_exception():
            while not coord.should_stop():
                ...do some work...

    """

    def __init__(self):
        """Create a new Coordinator.
        """
        # Protects all attributes.
        self._lock = Lock()
        # Event set when processes must stop.
        self._stop_event = Event()
        # Python exc_info to report.
        # If not None, it should hold the returned value of sys.exc_info(), which is
        # a tuple containing exception (type, value, traceback).
        self._exc_info_to_raise = None
        # True if we have called join() already.
        self._joined = Value(ctypes.c_int, 0, lock=False)
        # Set of processes registered for joining when join() is called.  These
        # will be joined in addition to the processes passed to the join()
        # call.  It's ok if processes are both registered and passed to the join()
        # call.
        self._registered_processes = set()

    def request_stop(self, ex=None):
        """Request that the processes stop.

        After this is called, calls to ``should_stop()`` will return ``True``.
        Note: If an exception is being passed in, in must be in the context of
        handling the exception (i.e. ``try: ... except Exception as ex: ...``) and not
        a newly created one.

        Args:
            ex (Exception or exc_info tuple): Optional `Exception`, or
            Python `exc_info` tuple as returned by `sys.exc_info()`.
            If this is the first call to `request_stop()` the
            corresponding exception is recorded and re-raised from `join()`.
        """
        with self._lock:
            # If we have already joined the coordinator the exception will not have a
            # chance to be reported, so just raise it normally.  This can happen if
            # you continue to use a session have having stopped and joined the
            # coordinator process.
            if self.joined:
                if isinstance(ex, tuple):
                    six.reraise(*ex)
                elif ex is not None:
                    # NOTE(touts): This is bogus if request_stop() is not called
                    # from the exception handler that raised ex.
                    six.reraise(*sys.exc_info())
            if not self._stop_event.is_set():
                if ex and self._exc_info_to_raise is None:
                    if isinstance(ex, tuple):
                        logging.info(
                            "Error reported to Coordinator: %s",
                            str(ex[1]),
                            exc_info=ex)
                        self._exc_info_to_raise = ex
                    else:
                        logging.info("Error reported to Coordinator: %s, %s",
                                     type(ex), str(ex))
                        self._exc_info_to_raise = sys.exc_info()
                    # self._exc_info_to_raise should contain a tuple containing exception
                    # (type, value, traceback)
                    if (len(self._exc_info_to_raise) != 3
                            or not self._exc_info_to_raise[0]
                            or not self._exc_info_to_raise[1]):
                        # Raise, catch and record the exception here so that error happens
                        # where expected.
                        try:
                            raise ValueError(
                                "ex must be a tuple or sys.exc_info must "
                                "return the current exception: %s" %
                                self._exc_info_to_raise)
                        except ValueError:
                            # Record this error so it kills the coordinator properly.
                            # NOTE(touts): As above, this is bogus if request_stop() is not
                            # called from the exception handler that raised ex.
                            self._exc_info_to_raise = sys.exc_info()

            self._stop_event.set()

    def clear_stop(self):
        """Clears the stop flag.
        After this is called, calls to ``should_stop()`` will return ``False``.
        """
        with self._lock:
            self._joined.value = 0
            self._exc_info_to_raise = None
            if self._stop_event.is_set():
                self._stop_event.clear()

    def should_stop(self):
        """Check if stop was requested.
        Returns:
            True if a stop was requested.
        """
        return self._stop_event.is_set()

    @contextlib.contextmanager
    def stop_on_exception(self):
        """Context manager to request stop when an Exception is raised.
        Code that uses a coordinator must catch exceptions and pass
        them to the ``request_stop()`` method to stop the other processes
        managed by the coordinator.
        This context handler simplifies the exception handling.
        Use it as follows:

        .. code-block:: python

            with coord.stop_on_exception():
                # Any exception raised in the body of the with
                # clause is reported to the coordinator before terminating
                # the execution of the body.
                ...body...

        This is completely equivalent to the slightly longer code:

        .. code-block:: python

            try:
                ...body...
            except:
                coord.request_stop(sys.exc_info())

        Yields:
            nothing.
        """
        try:
            yield
        except:  # pylint: disable=bare-except
            self.request_stop(ex=sys.exc_info())

    def wait_for_stop(self, timeout=None):
        """Wait till the Coordinator is told to stop.
        Args:
            timeout: Float.  Sleep for up to that many seconds waiting for
                should_stop() to become True.
        Returns:
            True if the Coordinator is told stop, False if the timeout expired.
        """
        return self._stop_event.wait(timeout)

    def register_process(self, process):
        """Register a process to join.
        Args:
            process: A python.multiprocessing.Process to join.
        """
        with self._lock:
            self._registered_processes.add(process)

    def join(self,
             processes=None,
             stop_grace_period_secs=120,
             ignore_live_processes=False):
        """Wait for processes to terminate.
        This call blocks until a set of processes have terminated.  The set of process
        is the union of the processes passed in the `processes` argument and the list
        of processes that registered with the coordinator by calling
        `Coordinator.register_process()`.
        After the processes stop, if an `exc_info` was passed to `request_stop`, that
        exception is re-raised.
        Grace period handling: When `request_stop()` is called, processes are given
        'stop_grace_period_secs' seconds to terminate.  If any of them is still
        alive after that period expires, a `RuntimeError` is raised.  Note that if
        an `exc_info` was passed to `request_stop()` then it is raised instead of
        that `RuntimeError`.
        Args:
            processes (list of `Processes`): The started processes to join in
                addition to the registered processes.
            stop_grace_period_secs: Number of seconds given to processes to stop after
                `request_stop()` has been called.
            ignore_live_processes: If `False`, raises an error if any of the processes are
                still alive after `stop_grace_period_secs`.
        Raises:
            RuntimeError: If any process is still alive after `request_stop()`
                is called and the grace period expires.
        """
        # processes registered after this call will not be joined.
        with self._lock:
            if processes is None:
                processes = self._registered_processes
            else:
                processes = self._registered_processes.union(set(processes))
            # Copy the set into a list to avoid race conditions where a new process
            # is added while we are waiting.
            processes = list(processes)

        # Wait for all processes to stop or for request_stop() to be called.
        while any(t.is_alive()
                  for t in processes) and not self.wait_for_stop(1.0):
            pass

        # If any process is still alive, wait for the grace period to expire.
        # By the time this check is executed, processes may still be shutting down,
        # so we add a sleep of increasing duration to give them a chance to shut
        # down without losing too many cycles.
        # The sleep duration is limited to the remaining grace duration.
        stop_wait_secs = 0.001
        while any(t.is_alive()
                  for t in processes) and stop_grace_period_secs >= 0.0:
            time.sleep(stop_wait_secs)
            stop_grace_period_secs -= stop_wait_secs
            stop_wait_secs = 2 * stop_wait_secs
            # Keep the waiting period within sane bounds.
            # The minimum value is to avoid decreasing stop_wait_secs to a value
            # that could cause stop_grace_period_secs to remain unchanged.
            stop_wait_secs = max(
                min(stop_wait_secs, stop_grace_period_secs), 0.001)

        # List the processes still alive after the grace period.
        stragglers = [t.name for t in processes if t.is_alive()]

        # Terminate with an exception if appropriate.
        with self._lock:
            self._joined.value = 1
            self._registered_processes = set()
            if self._exc_info_to_raise:
                six.reraise(*self._exc_info_to_raise)
            elif stragglers:
                if ignore_live_processes:
                    logging.info(
                        "Coordinator stopped with processes still running: %s",
                        " ".join(stragglers))
                else:
                    raise RuntimeError(
                        "Coordinator stopped with processes still running: %s"
                        % " ".join(stragglers))

    @property
    def joined(self):
        return self._joined.value

    def raise_requested_exception(self):
        """If an exception has been passed to `request_stop`, this raises it."""
        with self._lock:
            if self._exc_info_to_raise:
                six.reraise(*self._exc_info_to_raise)


class Process(mp.Process):
    """A coordinated process class to execute acting loops.
    """

    def __init__(self, coord, target=None, args=(), kwargs={}):
        """Creates a process, running target in a loop, managed by coordinator.

        Args:
            coord (Coordinator): coordinator used to manage this new process.
            target (callable): to be invoked by run() in a loop, until
                coordinator tells the process to stop.
            args (list): optional arguments for target callable.
            kwargs (dict): optional keyword arguments for target callable.
        """
        if not isinstance(coord, Coordinator):
            raise ValueError(
                "'coord' argument must be a Coordinator: %s" % coord)
        super().__init__()
        self._coord = coord
        # allow pass in target or overriding body
        self._target = target or self.body
        self._args = args
        self._kwargs = kwargs
        self._coord.register_process(self)

    def body(self, args=(), kwargs={}):
        raise NotImplementedError

    def run(self):
        with self._coord.stop_on_exception():
            self.start_loop()
            while not self._coord.should_stop():
                self.run_loop()
            self.stop_loop()

    def start_loop(self):
        """Called when the process starts."""
        pass

    def stop_loop(self):
        """Called when the process stops."""
        pass

    def run_loop(self):
        """Called in a back to back loop."""
        if self._target:
            self._target(*self._args, **self._kwargs)
