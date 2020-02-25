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

# TODO: implement metrics


class Metric(object):
    def summarize(self, train_step, step_metrics):
        pass

    def __call__(self, experience):
        pass


class NumberOfEpisodes(Metric):
    pass


class EnvironmentSteps(Metric):
    pass


class AverageReturnMetric(Metric):
    def __init__(self, batch_size, buffer_size):
        pass


class AverageEpisodeLengthMetric(Metric):
    def __init__(self, batch_size, buffer_size):
        pass
