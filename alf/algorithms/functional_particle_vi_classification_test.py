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


import absl
from absl.testing import parameterized
import numpy as np
import torch
import torch.nn.functional as F

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.functional_particle_vi_algorithm import FuncParVIAlgorithm
from alf.tensor_specs import TensorSpec
from alf.utils import math_ops
from alf.utils.datagen import load_nclass_test


class FuncParVIClassificationTest(parameterized.TestCase, alf.test.TestCase):
    """ 
    ParVI Classification Test
        A 4 class classification problem, where the classes are distributed
        as 4 symmetric Normal distributions with non overlapping support. 
        The hypernetwork is trained to sample classification functions that
        fit the data, for the purpose of observing the predictive
        distributions of sampled funcitons on data outside the training
        distribution. 
    """
    
    @parameterized.parameters(
        ('svgd', False),
        ('gfsf', False),
        ('svgd', True),
        ('gfsf', True),
        ('minmax', False),
    )
    def test_classification_func_par_vi(self,
                                        par_vi='svgd',
                                        function_vi=False,
                                        num_classes=2,
                                        num_particles=100):
        """
        Symmetric N-class classification problem. The training data are drawn
        from a mixture of normal distributions, the class index of a data point
        is given by the index of the component it was drawn from. 
        The particle ensemble is trained to achieve low loss / high accuracy
        on this data. The test shows the epistemic uncertainty of the ensemble
        up to 2 std. in the regions near and far from the data. 
        """

        input_size = 2
        output_dim = num_classes
        input_spec = TensorSpec((input_size, ), torch.float64)
        train_size = 100
        test_size = 100
        batch_size = train_size
        train_batch_size = 10
        train_loader, test_loader = load_nclass_test(
            num_classes,
            train_size=train_size,
            test_size=test_size,
            train_bs=train_batch_size,
            test_bs=train_batch_size)
        test_inputs = test_loader.dataset.get_features()
        test_targets = test_loader.dataset.get_targets()

        algorithm = FuncParVIAlgorithm(
            input_tensor_spec=input_spec,
            fc_layer_params=((10, True), (10, True)),
            last_layer_param=(output_dim, True),
            last_activation=math_ops.identity,
            num_particles=num_particles,
            loss_type='classification',
            par_vi=par_vi,
            function_vi=function_vi,
            function_bs=train_batch_size,
            function_extra_bs_ratio=0.1,
            critic_hidden_layers=(3,),
            critic_iter_num=2,
            critic_l2_weight=10.,
            critic_optimizer=alf.optimizers.Adam(lr=1e-2),
            optimizer=alf.optimizers.Adam(lr=1e-2))

        algorithm.set_data_loader(
            train_loader,
            test_loader,
            entropy_regularization=batch_size/train_size)
        absl.logging.info('Functional Particle VI: Fitting {} Classes'.format(
            num_classes))
        absl.logging.info('{} - {} particles'.format(par_vi, num_particles))
        
        def _test(i):
            params = algorithm.particles
            outputs = algorithm.predict_step(test_inputs).output

            _params = params.detach().cpu().numpy()

            probs = F.softmax(outputs, dim=-1)
            preds = probs.mean(1).cpu().argmax(-1)
            mean_acc = preds.eq(test_targets.cpu().view_as(preds)).float()
            mean_acc = mean_acc.sum() / len(test_targets)
            
            sample_preds = probs.cpu().argmax(-1).reshape(-1, 1)
            targets_unrolled = test_targets.unsqueeze(1).repeat(
                1, num_particles).reshape(-1, 1)
            
            sample_acc = sample_preds.eq(
                targets_unrolled.cpu().view_as(sample_preds)).float()
            sample_acc = sample_acc.sum()/len(targets_unrolled)

            absl.logging.info('-'*86)
            absl.logging.info('iter {}'.format(i))
            absl.logging.info('mean particle acc: {}'.format(mean_acc.item()))
            absl.logging.info('all particles acc: {}'.format(
                sample_acc.item()))
        
        train_iter = 1500
        for i in range(train_iter):
            algorithm.train_iter()
            if i % 500 == 0:
                _test(i)
        
        algorithm.evaluate()

        params = algorithm.particles
        outputs, _ = algorithm._param_net(test_inputs)
        _params = params.detach().cpu().numpy()

        probs = F.softmax(outputs, dim=-1)
        preds = probs.mean(1).cpu().argmax(-1)
        mean_acc = preds.eq(test_targets.cpu().view_as(preds)).float()
        mean_acc = mean_acc.sum() / len(test_targets)
        
        sample_preds = probs.cpu().argmax(-1).reshape(-1, 1)
        targets_unrolled = test_targets.unsqueeze(1).repeat(
            1, num_particles).reshape(-1, 1)
        
        sample_acc = sample_preds.eq(
            targets_unrolled.cpu().view_as(sample_preds)).float()
        sample_acc = sample_acc.sum()/len(targets_unrolled)

        absl.logging.info('-'*86)
        absl.logging.info('iter {}'.format(i))
        absl.logging.info('mean particle acc: {}'.format(mean_acc.item()))
        absl.logging.info('all particles acc: {}'.format(
            sample_acc.item()))
        self.assertGreater(mean_acc, 0.95)
        self.assertGreater(sample_acc, 0.95)


if __name__ == "__main__":
    alf.test.main()
