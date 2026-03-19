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
import os
import tempfile
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.util import tensor_util
import torch

import alf


def _find_event_file(root_dir):
    event_file = None
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            if "events" in file_name and 'profile' not in file_name:
                event_file = os.path.join(root, file_name)
                break
    return event_file


class SummaryTest(alf.test.TestCase):

    def test_summary(self):
        with tempfile.TemporaryDirectory() as root_dir:
            writer = alf.summary.create_summary_writer(root_dir,
                                                       flush_secs=10,
                                                       max_queue=10)
            alf.summary.set_default_writer(writer)
            alf.summary.enable_summary()
            with alf.summary.scope("root") as scope_name:
                self.assertEqual(scope_name, "root/")
                alf.summary.scalar("scalar", 2020)
                with alf.summary.scope("a") as scope_name:
                    self.assertEqual(scope_name, "root/a/")
                    alf.summary.text("text", "sample text")
                with alf.summary.record_if(lambda: False):
                    alf.summary.text("test", "this should not appear")
                alf.summary.disable_summary()
                alf.summary.text("testa", "this should not appear")
                alf.summary.enable_summary()
                with alf.summary.scope("b") as scope_name:
                    self.assertEqual(scope_name, "root/b/")
                    alf.summary.histogram("histogram",
                                          torch.arange(100).numpy())
            writer.close()

            event_file = _find_event_file(root_dir)

            self.assertIsNotNone(event_file)

            tag2val = {
                'root/scalar': None,
                'root/a/text/text_summary': None,
                'root/b/histogram': None
            }

            for event_str in event_file_loader.EventFileLoader(
                    event_file).Load():
                if event_str.summary.value:
                    for item in event_str.summary.value:
                        self.assertTrue(item.tag in tag2val)
                        tag2val[item.tag] = tensor_util.make_ndarray(
                            item.tensor)

            self.assertEqual(tag2val['root/scalar'], 2020)
            self.assertEqual(tag2val['root/a/text/text_summary'][0],
                             b'sample text')
            self.assertEqual(tag2val['root/b/histogram'].min(), 0)
            self.assertEqual(tag2val['root/b/histogram'].max(), 99)
            self.assertEqual(len(tag2val['root/b/histogram']), 30)

    def test_average_all_summaries(self):
        with tempfile.TemporaryDirectory() as root_dir:
            writer = alf.summary.create_summary_writer(root_dir,
                                                       flush_secs=10,
                                                       max_queue=10)
            alf.summary.set_default_writer(writer)
            alf.summary.enable_summary()
            event_file = _find_event_file(root_dir)
            self.assertIsNotNone(event_file)

            tag2val = {
                'scalar1': None,
                'scalar2': None,
                'scalar3': None,
                'scalar4': None,
            }

            def load_summaries():
                writer.flush()
                for event_str in event_file_loader.EventFileLoader(
                        event_file).Load():
                    if event_str.summary.value:
                        for item in event_str.summary.value:
                            self.assertTrue(item.tag in tag2val)
                            tag2val[item.tag] = tensor_util.make_ndarray(
                                item.tensor)

            load_summaries()

            with alf.summary.record_if(lambda: True):
                # average_over_summary_interval is False by default
                alf.summary.scalar("scalar1", 101)
                alf.summary.scalar("scalar1", 102)
                alf.summary.scalar("scalar2", 103)
                alf.summary.scalar("scalar3", 105)
                alf.summary.scalar("scalar4", 106)

            load_summaries()
            self.assertEqual(tag2val['scalar1'], 102)
            self.assertEqual(tag2val['scalar2'], 103)
            self.assertEqual(tag2val['scalar3'], 105)
            self.assertEqual(tag2val['scalar4'], 106)

            # Test that average_all_summaries uses its own record boundary and
            # ignores nested record_if settings.
            num_iters = 4
            counter = 1
            cond = lambda: counter == num_iters
            target_names = ["scalar1", "scalar3"]
            with alf.summary.average_all_summaries(cond, target_names):
                for i in range(num_iters):
                    # This record_if should be overwritten for scalar1
                    with alf.summary.record_if(lambda: True):
                        # This scalar should be averaged
                        alf.summary.scalar("scalar1", 100 + i)
                        # This scalar should not be averaged
                        alf.summary.scalar("scalar2", 100 + i)

                    # This record_if should be overwritten for scalar3
                    with alf.summary.record_if(lambda: counter == 2):
                        # This scalar should be averaged
                        alf.summary.scalar("scalar3", 100 + i)
                        # This scalar should not be averaged
                        alf.summary.scalar("scalar4", 100 + i)
                    counter += 1

            load_summaries()
            self.assertEqual(tag2val['scalar1'], 101.5)
            self.assertEqual(tag2val['scalar2'], 103)
            self.assertEqual(tag2val['scalar3'], 101.5)
            self.assertEqual(tag2val['scalar4'], 101)

            writer.close()


if __name__ == "__main__":
    alf.test.main()
