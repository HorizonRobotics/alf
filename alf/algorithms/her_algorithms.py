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
"""HER Algorithms (Wrappers)."""
"""Classes defined here are used to transfer relevant info about the
sampled/replayed experience from HindsightDataTransformer all the way to
algorithm.calc_loss and the loss class.

Actual hindsight relabeling happens in HindsightDataTransformer.

For usage, see alf/examples/her_fetchpush_conf.py.
"""

import alf
from alf.algorithms.sac_algorithm import SacAlgorithm, SacInfo
from alf.algorithms.ddpg_algorithm import DdpgAlgorithm, DdpgInfo
from alf.data_structures import TimeStep
from alf.utils import common


def her_wrapper(alg_cls, alg_info):
    """A helper function to construct HerAlgo based on the base (off-policy) algorithm.

    We mainly do two things here:
        1. Create the new HerInfo namedtuple, containing a ``derived`` field together
        with the existing fields of AlgInfo.  The ``derived`` field is a dict, to be
        populated with information derived from the Hindsight relabeling process.
        This HerInfo structure stores training information collected from replay and
        processed by the algorithm's train_step.

        2. Create a new HerAlgo child class of the input base algorithm.
        The new class additionally handles passing derived fields along the pipeline
        for the loss function (e.g. LowerboundedTDLoss) to access.
    """
    HerClsName = "Her" + alg_cls.__name__
    # HerAlgo class inherits the base RL algorithm class
    HerCls = type(HerClsName, (alg_cls, ), {})
    HerCls.counter = 0

    HerInfoName = "Her" + alg_info.__name__
    # Unfortunately, the user has to ensure that the default_value of HerAlgInfo has to be
    # exactly the same as the AlgInfo, otherwise there could be bugs.
    HerInfoCls = alf.data_structures.namedtuple(
        HerInfoName, alg_info._fields + ("derived", ), default_value=())
    alg_info.__name__ = HerInfoName

    # NOTE: replay_buffer.py has similar functions for handling BatchInfo namedtuple.

    # New __new__ for AlgInfo, so every time AlgInfo is called to create an instance,
    # an HerAlgInfo instance (with the additional ``derived`` dict) is created and
    # returned instead.  This allows us to wrap an algorithm's AlgInfo class without
    # changing any code in the original AlgInfo class, keeping HER code separate.
    @common.add_method(alg_info)
    def __new__(info_cls, **kwargs):
        assert info_cls == alg_info
        her_info = HerInfoCls(**kwargs)
        # Set default value, later code will check for this
        her_info = her_info._replace(derived={})
        return her_info

    # New accessor methods for HerAlgInfo to access the ``derived`` dict.
    @common.add_method(HerInfoCls)
    def get_derived_field(self, field):
        assert field in self.derived, f"field {field} not in BatchInfo.derived"
        return self.derived[field]

    @common.add_method(HerInfoCls)
    def get_derived(self):
        return self.derived

    @common.add_method(HerInfoCls)
    def set_derived(self, new_dict):
        assert self.derived == {}
        return self._replace(derived=new_dict)

    # New methods for HerAlg
    @common.add_method(HerCls)
    def __init__(self, **kwargs):
        """
        Args:
            kwargs: arguments passed to the constructor of the underlying algorithm.
        """
        assert HerCls.counter == 0, f"HerCls {HerCls} already defined"
        super(HerCls, self).__init__(**kwargs)
        HerCls.counter += 1

    @common.add_method(HerCls)
    def preprocess_experience(self, inputs: TimeStep, rollout_info: alg_info,
                              batch_info):
        """Pass derived fields from batch_info into rollout_info"""
        time_step, rollout_info = super(HerCls, self).preprocess_experience(
            inputs, rollout_info, batch_info)
        if hasattr(rollout_info, "derived") and batch_info.derived:
            # Expand to the proper dimensions consistent with other experience fields
            derived = alf.nest.map_structure(
                lambda x: x.unsqueeze(1).expand(time_step.reward.shape[:2]),
                batch_info.get_derived())
            rollout_info = rollout_info.set_derived(derived)
        return time_step, rollout_info

    @common.add_method(HerCls)
    def train_step(self, inputs: TimeStep, state, rollout_info: alg_info):
        """Pass derived fields from rollout_info into alg_step.info"""
        alg_step = super(HerCls, self).train_step(inputs, state, rollout_info)
        return alg_step._replace(
            info=alg_step.info.set_derived(rollout_info.get_derived()))

    return HerCls  # End of her_wrapper function


# Create the actual wrapped HerAlgorithms
HerSacAlgorithm = her_wrapper(SacAlgorithm, SacInfo)
HerDdpgAlgorithm = her_wrapper(DdpgAlgorithm, DdpgInfo)
"""To help understand what's going on, here is the detailed data flow:

1. Replayer samples the experience with batch_info from replay_buffer.

2. HindsightDataTransformer samples and relabels the experience, stores the derived info containing
her: whether the experience has been relabeled, future_distance: the number of time steps to
the future achieved goal used to relabel the experience.
HindsightDataTransformer finally returns experience with experience.batch_info.derived
containing the derived information.

(NOTE: we cannot put HindsightDataTransformer into HerAlgo.preprocess_experience, as preprocessing
happens after data_transformations, but Hindsight relabeling has to happen before other data
transformations like observation normalization, because hindsight accesses replay_buffer data directly,
which has not gone through the data transformers.
Maybe we could invoke HindsightDataTransformer automatically, e.g. by preprending it to
``TrainConfig.data_transformer_ctr`` in this file.  Maybe that's too magical, and should be avoided.)

3. HerAlgo.preprocess_experience copies ``batch_info.derived`` over to ``rollout_info.derived``.
NOTE: We cannot copy from exp to rollout_info because the input to preprocess_experience is time_step,
not exp in algorithm.py:

.. code-block:: python

   time_step, rollout_info = self.preprocess_experience(
       experience.time_step, experience.rollout_info, batch_info)

4. HerAlgo.train_step copies ``exp.rollout_info.derived`` over to ``policy_step.info.derived``.
NOTE: we cannot just copy derived from exp into AlgInfo in train_step, because train_step accepts
time_step instead of exp as input:

.. code-block:: python

    policy_step = self.train_step(exp.time_step, policy_state,
                                  exp.rollout_info)

5. BaseAlgo.calc_loss will call LowerboundedTDLoss with HerBaseAlgoInfo.
"""
