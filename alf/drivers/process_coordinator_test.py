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
"""Tests for Process Coordinator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import current_process, Event, Process
import sys
import time

from alf.drivers import process_coordinator as coordinator
from alf import test


def StopOnEvent(coord, wait_for_stop, set_when_stopped):
    wait_for_stop.wait()
    coord.request_stop()
    set_when_stopped.set()


def RaiseOnEvent(coord, wait_for_stop, set_when_stopped, ex, report_exception):
    try:
        wait_for_stop.wait()
        raise ex
    except RuntimeError as e:
        if report_exception:
            coord.request_stop(e)
        else:
            coord.request_stop(sys.exc_info())
    finally:
        if set_when_stopped:
            set_when_stopped.set()


def RaiseOnEventUsingContextHandler(coord, wait_for_stop, set_when_stopped,
                                    ex):
    with coord.stop_on_exception():
        wait_for_stop.wait()
        raise ex
    if set_when_stopped:
        set_when_stopped.set()


def SleepABit(n_secs):
    time.sleep(n_secs)


def WaitForProcessesToRegister(coord, num_processes):
    while True:
        with coord._lock:
            if len(coord._registered_processes) == num_processes:
                break
        time.sleep(0.001)


class CoordinatorTest(test.TestCase):
    def testStopAPI(self):
        coord = coordinator.Coordinator()
        self.assertFalse(coord.should_stop())
        self.assertFalse(coord.wait_for_stop(0.01))
        coord.request_stop()
        self.assertTrue(coord.should_stop())
        self.assertTrue(coord.wait_for_stop(0.01))

    def testStopAsync(self):
        coord = coordinator.Coordinator()
        self.assertFalse(coord.should_stop())
        self.assertFalse(coord.wait_for_stop(0.1))
        wait_for_stop_ev = Event()
        has_stopped_ev = Event()
        t = Process(
            target=StopOnEvent, args=(coord, wait_for_stop_ev, has_stopped_ev))
        t.start()
        self.assertFalse(coord.should_stop())
        self.assertFalse(coord.wait_for_stop(0.01))
        wait_for_stop_ev.set()
        has_stopped_ev.wait()
        self.assertTrue(coord.wait_for_stop(0.05))
        self.assertTrue(coord.should_stop())

    def testJoin(self):
        coord = coordinator.Coordinator()
        processes = [
            Process(target=SleepABit, args=(0.01, )),
            Process(target=SleepABit, args=(0.02, )),
            Process(target=SleepABit, args=(0.01, ))
        ]
        for t in processes:
            t.start()
        coord.join(processes)
        for t in processes:
            self.assertFalse(t.is_alive())

    def testJoinAllRegistered(self):
        coord = coordinator.Coordinator()
        processes = [
            Process(target=SleepABit, args=(0.01, )),
            Process(target=SleepABit, args=(0.02, )),
            Process(target=SleepABit, args=(0.01, ))
        ]
        for t in processes:
            t.start()
        for p in processes:
            coord.register_process(p)
        WaitForProcessesToRegister(coord, len(processes))
        coord.join()
        for t in processes:
            self.assertFalse(t.is_alive())

    def testJoinSomeRegistered(self):
        coord = coordinator.Coordinator()
        processes = [
            Process(target=SleepABit, args=(0.01, )),
            Process(target=SleepABit, args=(0.02, )),
            Process(target=SleepABit, args=(0.01, ))
        ]
        for t in processes:
            t.start()
        coord.register_process(processes[0])
        coord.register_process(processes[2])
        WaitForProcessesToRegister(coord, 2)
        # processes[1] is not registered we must pass it in.
        coord.join([processes[1]])
        for t in processes:
            self.assertFalse(t.is_alive())

    def testJoinGraceExpires(self):
        def TestWithGracePeriod(stop_grace_period):
            coord = coordinator.Coordinator()
            wait_for_stop_ev = Event()
            has_stopped_ev = Event()
            processes = [
                Process(
                    target=StopOnEvent,
                    args=(coord, wait_for_stop_ev, has_stopped_ev)),
                Process(target=SleepABit, args=(10.0, ))
            ]
            for t in processes:
                t.daemon = True
                t.start()
            wait_for_stop_ev.set()
            has_stopped_ev.wait()
            with self.assertRaisesRegex(RuntimeError,
                                        "processes still running"):
                coord.join(processes, stop_grace_period_secs=stop_grace_period)

        TestWithGracePeriod(1e-10)
        TestWithGracePeriod(0.002)
        TestWithGracePeriod(1.0)

    def testJoinWithoutGraceExpires(self):
        coord = coordinator.Coordinator()
        wait_for_stop_ev = Event()
        has_stopped_ev = Event()
        processes = [
            Process(
                target=StopOnEvent,
                args=(coord, wait_for_stop_ev, has_stopped_ev)),
            Process(target=SleepABit, args=(10.0, ))
        ]
        for t in processes:
            t.daemon = True
            t.start()
        wait_for_stop_ev.set()
        has_stopped_ev.wait()
        coord.join(
            processes, stop_grace_period_secs=1., ignore_live_processes=True)

    def testJoinRaiseReportExcInfo(self):
        coord = coordinator.Coordinator()
        ev_1 = Event()
        ev_2 = Event()
        processes = [
            Process(
                target=RaiseOnEvent,
                args=(coord, ev_1, ev_2, RuntimeError("First"), False)),
            Process(
                target=RaiseOnEvent,
                args=(coord, ev_2, None, RuntimeError("Too late"), False))
        ]
        for t in processes:
            t.start()

        ev_1.set()

        # Being converted from threads, we don't raise exceptions from
        # sub processes, but we do stop all processing via the stop event.
        #
        # If we need to print sub process traceback in the future, we can use
        # something like this: https://stackoverflow.com/questions/19924104/python-multiprocessing-handling-child-errors-in-parent

        # not raising: with self.assertRaisesRegex(RuntimeError, "First"):
        coord.join(processes)

    def testJoinRaiseReportException(self):
        coord = coordinator.Coordinator()
        ev_1 = Event()
        ev_2 = Event()
        processes = [
            Process(
                target=RaiseOnEvent,
                args=(coord, ev_1, ev_2, RuntimeError("First"), True)),
            Process(
                target=RaiseOnEvent,
                args=(coord, ev_2, None, RuntimeError("Too late"), True))
        ]
        for t in processes:
            t.start()

        ev_1.set()
        # not raising with self.assertRaisesRegex(RuntimeError, "First"):
        coord.join(processes)

    def testJoinRaiseReportExceptionUsingHandler(self):
        coord = coordinator.Coordinator()
        ev_1 = Event()
        ev_2 = Event()
        processes = [
            Process(
                target=RaiseOnEventUsingContextHandler,
                args=(coord, ev_1, ev_2, RuntimeError("First"))),
            Process(
                target=RaiseOnEventUsingContextHandler,
                args=(coord, ev_2, None, RuntimeError("Too late")))
        ]
        for t in processes:
            t.start()

        ev_1.set()
        # not raising with self.assertRaisesRegex(RuntimeError, "First"):
        coord.join(processes)

    def testClearStopClearsExceptionToo(self):
        coord = coordinator.Coordinator()
        ev_1 = Event()
        processes = [
            Process(
                target=RaiseOnEvent,
                args=(coord, ev_1, None, RuntimeError("First"), True)),
        ]
        for t in processes:
            t.start()

        # not raising with self.assertRaisesRegex(RuntimeError, "First"):
        ev_1.set()
        coord.join(processes)

        coord.clear_stop()
        processes = [
            Process(
                target=RaiseOnEvent,
                args=(coord, ev_1, None, RuntimeError("Second"), True)),
        ]
        for t in processes:
            t.start()
        # not raising with self.assertRaisesRegex(RuntimeError, "Second"):
        ev_1.set()
        coord.join(processes)

    def testRequestStopRaisesIfJoined(self):
        coord = coordinator.Coordinator()
        # Join the coordinator right away.
        coord.join([])
        reported = False
        with self.assertRaisesRegex(RuntimeError, "Too late"):
            try:
                raise RuntimeError("Too late")
            except RuntimeError as e:
                reported = True
                coord.request_stop(e)
        self.assertTrue(reported)
        # If we clear_stop the exceptions are handled normally.
        coord.clear_stop()
        try:
            raise RuntimeError("After clear")
        except RuntimeError as e:
            coord.request_stop(e)
        with self.assertRaisesRegex(RuntimeError, "After clear"):
            coord.join([])

    def testRequestStopRaisesIfJoined_ExcInfo(self):
        # Same as testRequestStopRaisesIfJoined but using syc.exc_info().
        coord = coordinator.Coordinator()
        # Join the coordinator right away.
        coord.join([])
        reported = False
        with self.assertRaisesRegex(RuntimeError, "Too late"):
            try:
                raise RuntimeError("Too late")
            except RuntimeError:
                reported = True
                coord.request_stop(sys.exc_info())
        self.assertTrue(reported)
        # If we clear_stop the exceptions are handled normally.
        coord.clear_stop()
        try:
            raise RuntimeError("After clear")
        except RuntimeError:
            coord.request_stop(sys.exc_info())
        with self.assertRaisesRegex(RuntimeError, "After clear"):
            coord.join([])


def _StopAt0(coord, n):
    if n[0] == 0:
        coord.request_stop()
    else:
        n[0] -= 1


if __name__ == "__main__":
    test.main()
