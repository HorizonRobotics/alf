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
import tempfile
import torch
import torchvision.models as models
import unittest
import os

import alf
from alf.data_structures import restart
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.utils.tensorrt_utils import (
    OnnxRuntimeEngine, compile_method, get_tensorrt_engine_class,
    is_onnxruntime_available, is_tensorrt_available)


def create_sac_and_inputs():
    # Create algorithm
    observation_spec = alf.TensorSpec((10, ))
    action_spec = alf.BoundedTensorSpec((2, ))
    sac = SacAlgorithm(
        observation_spec,
        action_spec,
        actor_network_cls=partial(
            alf.networks.ActorDistributionNetwork,
            fc_layer_params=(1024, ) * 5),
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
    @unittest.skipIf(not is_onnxruntime_available(),
                     "onnxruntime not installed")
    def test_onnxruntime_backends(self):
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        expected_providers = ['CPUExecutionProvider']
        # On CI server, we don't have GPUs or tensorrt (GPU only)
        if torch.cuda.is_available():
            self.assertTrue('CUDAExecutionProvider' in providers,
                            "Need to install onnxruntime-gpu!")
            expected_providers.insert(0, 'CUDAExecutionProvider')
        if is_tensorrt_available():
            self.assertTrue('TensorrtExecutionProvider' in providers,
                            "tensorrt installation error!")
            expected_providers.insert(0, 'TensorrtExecutionProvider')
        self.assertEqual(providers, expected_providers)

    @unittest.skipIf(not is_onnxruntime_available(),
                     "onnxruntime not installed")
    def test_onnxruntime_engine(self):
        alg, timestep, state = create_sac_and_inputs()
        engine = OnnxRuntimeEngine(
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
            engine_alg_step = engine(timestep, state=state)
        print(f"Onnxruntime predict step time: ",
              (time.time() - start_time) / 100)

        self.assertTensorClose(engine_alg_step.output, alg_step.output)

    @unittest.skipIf(not is_onnxruntime_available(),
                     "onnxruntime not installed")
    @parameterized.parameters(True, False)
    def test_compile_method(self, tensorrt_backend):
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

        compile_method(alg, 'predict_step', OnnxRuntimeEngine)
        alg.predict_step(timestep, state=state)  # build engine first
        start_time = time.time()
        for _ in range(100):
            engine_alg_step = alg.predict_step(timestep, state=state)
        print(f"Onnxruntime predict step time: ",
              (time.time() - start_time) / 100)

        self.assertTensorClose(engine_alg_step.output, alg_step.output)

    @unittest.skipIf(not is_tensorrt_available(), "tensorrt is unavailable")
    def test_tensorrt_engine(self):
        alg, timestep, state = create_sac_and_inputs()
        alg.eval()

        start_time = time.time()
        for _ in range(100):
            alg_step = alg.predict_step(timestep, state)
        print("Eager-mode predict step time: ",
              (time.time() - start_time) / 100)

        compile_method(alg, 'predict_step',
                       get_tensorrt_engine_class(validate_args=True))
        alg.predict_step(timestep, state=state)  # build engine
        start_time = time.time()
        for _ in range(100):
            trt_alg_step = alg.predict_step(timestep, state=state)
        print(f"TensorRT predict step time: ",
              (time.time() - start_time) / 100)

        self.assertTensorClose(trt_alg_step.output, alg_step.output)

    @unittest.skipIf(not is_tensorrt_available(), "tensorrt is unavailable")
    def test_tensorrt_engine_cache(self):
        engine_file = tempfile.mktemp(suffix='.trt')
        alg, timestep, state = create_sac_and_inputs()
        compile_method(
            alg, 'predict_step',
            get_tensorrt_engine_class(
                validate_args=True, engine_file=engine_file))
        start_time = time.time()
        alg.predict_step(timestep, state=state)  # build engine
        self.assertGreater(time.time() - start_time,
                           1)  # takes more than 1 second

        alg, timestep, state = create_sac_and_inputs()
        # Now if we compile again with engine file, the engine should be directly
        # loaded from disk, even though the alg has been recreated
        compile_method(
            alg, 'predict_step',
            get_tensorrt_engine_class(
                validate_args=True, engine_file=engine_file))
        start_time = time.time()
        alg.predict_step(timestep, state=state)  # load engine
        self.assertLess(time.time() - start_time, 0.1)

        alg, timestep, state = create_sac_and_inputs()
        # Now we compile again and force building the engine
        compile_method(
            alg, 'predict_step',
            get_tensorrt_engine_class(
                validate_args=True,
                engine_file=engine_file,
                force_build_engine=True))
        start_time = time.time()
        alg.predict_step(timestep, state=state)  # build engine
        self.assertGreater(time.time() - start_time, 1)

        alg, timestep, state = create_sac_and_inputs()
        compile_method(alg, 'predict_step')
        start_time = time.time()
        alg.predict_step(timestep, state=state)  # build engine
        self.assertGreater(time.time() - start_time, 1)

    @unittest.skipIf(not is_tensorrt_available()
                     and not is_onnxruntime_available(),
                     "tensorrt and onnxruntime are unavailable")
    def test_tensorrt_resnet50(self):
        model = models.resnet50(pretrained=True)
        model.eval()
        dummy_img = torch.randn(1, 3, 224, 224)

        for _ in range(10):
            eager_output = model(dummy_img)
        start_time = time.time()
        for _ in range(100):
            eager_output = model(dummy_img)
        print("Eager-mode predict step time: ",
              (time.time() - start_time) / 100)

        if is_tensorrt_available():
            for fp16 in [True, False]:
                model = models.resnet50(pretrained=True)
                model.eval()
                compile_method(
                    model, 'forward',
                    get_tensorrt_engine_class(fp16=fp16, validate_args=True))
                model(dummy_img)  # build engine
                for _ in range(10):
                    output = model(dummy_img)
                start_time = time.time()
                for _ in range(100):
                    output = model(dummy_img)
                fp_str = "16" if fp16 else "32"
                print(f"TensorRT FP{fp_str} predict step time: ",
                      (time.time() - start_time) / 100)
                eps = 0.03 if fp16 else 0.01
                self.assertTensorClose(eager_output, output, epsilon=eps)

        if is_onnxruntime_available():
            model1 = models.resnet50(pretrained=True)
            model1.eval()
            # Use onnxruntime API
            compile_method(model1, 'forward', OnnxRuntimeEngine)
            model1(dummy_img)  # build engine
            for _ in range(10):
                output = model1(dummy_img)
            start_time = time.time()
            for _ in range(100):
                output = model1(dummy_img)
            print(f"ONNX-runtime predict step time: ",
                  (time.time() - start_time) / 100)
            self.assertTensorClose(eager_output, output, epsilon=0.01)


if __name__ == "__main__":
    alf.test.main()
