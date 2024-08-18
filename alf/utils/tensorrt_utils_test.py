# Copyright (c) 2024 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
from functools import partial
import time
import torch
import os

import alf
from alf.data_structures import restart
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.utils.tensorrt_utils import TensorRTEngine, tensorrtify_method, is_available


def create_sac_and_inputs():
    # Create algorithm
    observation_spec = alf.TensorSpec((10, ))
    action_spec = alf.BoundedTensorSpec((2, ))
    sac = SacAlgorithm(
        observation_spec,
        action_spec,
        actor_network_cls=partial(
            alf.networks.ActorDistributionNetwork,
            fc_layer_params=(1024, ) * 10),
        critic_network_cls=partial(
            alf.networks.CriticNetwork, joint_fc_layer_params=(64, 64)))

    # Create dummy timestep and state
    obs = alf.utils.spec_utils.zeros_from_spec(observation_spec, batch_size=1)
    dummy_timestep = restart(
        observation=obs, action_spec=action_spec, batched=True)
    state = sac.get_initial_predict_state(batch_size=1)

    # randomize agent parameters
    for param in sac.parameters():
        param.data.uniform_(-0.01, 0.01)

    return sac, dummy_timestep, state


class TensorRTUtilsTest(parameterized.TestCase, alf.test.TestCase):
    def setUp(self):
        super().setUp()
        if not is_available():
            self.skipTest('onnxruntime or tensorrt is not installed.')

    @property
    def has_gpu(self):
        return torch.cuda.is_available()

    @property
    def has_tensorrt(self):
        try:
            import tensorrt
            return True
        except ImportError:
            return False

    def test_tensorrt_available(self):
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        expected_providers = ['CPUExecutionProvider']
        # On CI server, we don't have GPUs or tensorrt (GPU only)
        if self.has_gpu:
            self.assertTrue('CUDAExecutionProvider' in providers,
                            "Need to install onnxruntime-gpu!")
            expected_providers.insert(0, 'CUDAExecutionProvider')
        if self.has_tensorrt:
            self.assertTrue('TensorrtExecutionProvider' in providers,
                            "tensorrt installation error!")
            expected_providers.insert(0, 'TensorrtExecutionProvider')
        self.assertEqual(providers, expected_providers)

    def test_tensorrt_engine(self):
        alg, timestep, state = create_sac_and_inputs()
        trt_engine = TensorRTEngine(
            alg,
            SacAlgorithm.predict_step,
            example_args=(timestep, ),
            example_kwargs={'state': state})
        alg.eval()

        start_time = time.time()
        for _ in range(100):
            alg_step = alg.predict_step(timestep, state)
        print("Eager-mode predict step time: ",
              (time.time() - start_time) / 100)

        start_time = time.time()
        for _ in range(100):
            trt_alg_step = trt_engine(timestep, state=state)
        print(f"Graph-mode predict step time: ",
              (time.time() - start_time) / 100)

        torch.testing.assert_close(trt_alg_step.output, alg_step.output)

    @parameterized.parameters(True, False)
    def test_tensorrt_decorator(self, tensorrt_backend):
        alg, timestep, state = create_sac_and_inputs()
        alg.eval()

        start_time = time.time()
        for _ in range(100):
            alg_step = alg.predict_step(timestep, state)
        print("Eager-mode predict step time: ",
              (time.time() - start_time) / 100)

        if not tensorrt_backend:
            # This will use CUDA or CPU backend to execute the onnx model
            os.environ[
                'ORT_ONNX_BACKEND_EXCLUDE_PROVIDERS'] = 'TensorrtExecutionProvider'

        tensorrtify_method(alg, 'predict_step')
        alg.predict_step(timestep, state=state)  # build engine first
        start_time = time.time()
        for _ in range(100):
            trt_alg_step = alg.predict_step(timestep, state=state)
        print(f"Gragh-mode predict step time: ",
              (time.time() - start_time) / 100)

        torch.testing.assert_close(trt_alg_step.output, alg_step.output)


if __name__ == "__main__":
    alf.test.main()
