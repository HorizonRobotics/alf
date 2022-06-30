# Copyright (c) 2022 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from absl.testing import parameterized

import alf
from alf.algorithms.her_algorithms import HerSacAlgorithm, HerDdpgAlgorithm
from alf.algorithms.sac_algorithm import SacAlgorithm, SacInfo
from alf.algorithms.ddpg_algorithm import DdpgAlgorithm, DdpgInfo


class HerAlgorithmsTest(parameterized.TestCase, alf.test.TestCase):
    def test_her_algo_name(self):
        self.assertEqual("HerSacAlgorithm", HerSacAlgorithm.__name__)
        self.assertEqual("HerDdpgAlgorithm", HerDdpgAlgorithm.__name__)

    @parameterized.parameters([
        (SacInfo, ),
        (DdpgInfo, ),
    ])
    def test_her_info(self, Info):
        info = Info(reward=1)
        self.assertEqual(1, info.reward)
        # HerAlgInfo assumes default field value to be (), need to be consistent with AlgInfo
        self.assertEqual((), info.action)
        self.assertEqual({}, info.get_derived())
        ret = info.set_derived({"a": 1, "b": 2})
        # info is immutable
        self.assertEqual({}, info.get_derived())
        # ret is the new instance with field "derived" replaced
        self.assertEqual(1, ret.get_derived_field("a"))
        self.assertEqual(2, ret.get_derived_field("b"))
        # get nonexistent field with and without default
        self.assertEqual("none", ret.get_derived_field("x", default="none"))
        self.assertRaises(AssertionError, ret.get_derived_field, "x")
