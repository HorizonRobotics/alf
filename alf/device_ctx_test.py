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

import torch
import alf


class DeviceCtxTest(alf.test.TestCase):
    def test_device_ctx(self):
        with alf.device("cpu"):
            self.assertEqual(alf.get_default_device(), "cpu")
            self.assertEqual(torch.tensor([1]).device.type, "cpu")
            if torch.cuda.is_available():
                with alf.device("cuda"):
                    self.assertEqual(alf.get_default_device(), "cuda")
                    self.assertEqual(torch.tensor([1]).device.type, "cuda")
            self.assertEqual(alf.get_default_device(), "cpu")
            self.assertEqual(torch.tensor([1]).device.type, "cpu")


if __name__ == "__main__":
    alf.test.main()
