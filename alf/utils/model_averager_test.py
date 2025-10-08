# Copyright (c) 2025 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from functools import partial

from torch import nn
import alf
from alf.utils.model_averager import AveragedModel, ema_avg_fn


class ModelAveragerTest(alf.test.TestCase):

    def test_averaged_model(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))

        averaged_model = AveragedModel(model,
                                       avg_fn=partial(ema_avg_fn,
                                                      ema_rate=0.0))

        n = 16
        for i in range(n):
            model[0].weight.data.copy_(i)
            model[0].bias.data.copy_(i)
            model[1].weight.data.copy_(i)
            model[1].bias.data.copy_(i)
            model[1].running_mean.copy_(i)
            model[1].running_var.copy_(i)
            averaged_model.update_parameters(model)

        module = averaged_model.module

        self.assertTrue((module[0].weight == (n - 1) / 2).all())
        self.assertTrue((module[0].bias == (n - 1) / 2).all())
        self.assertTrue((module[1].weight == (n - 1) / 2).all())
        self.assertTrue((module[1].bias == (n - 1) / 2).all())
        self.assertTrue((module[1].running_mean == n - 1).all())
        self.assertTrue((module[1].running_var == n - 1).all())


if __name__ == '__main__':
    alf.test.main()
