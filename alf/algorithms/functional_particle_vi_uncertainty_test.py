# Copyright (c) 2021 Horizon Robotics. All Rights Reserved.
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
import torch
import torch.nn.functional as F

import alf
from alf.algorithms.functional_particle_vi_algorithm import FuncParVIAlgorithm
from alf.tensor_specs import TensorSpec
from alf.utils import math_ops

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rcParams.update({'font.size': 18})
import os


class FuncParVIUncertaintyTest(parameterized.TestCase, alf.test.TestCase):
    """ 
    HyperNetwork Sample Test
        Two tests given in order of increasing difficulty. 
        1. A 3-dimensional multivariate Gaussian distribution 
        2. A 4 class classification problem, where the classes are distributed
            as 4 symmetric Normal distributions with non overlapping support. 
            The hypernetwork is trained to sample classification functions that
            fit the data, for the purpose of observing the predictive
            distributions of sampled funcitons on data outside the training
            distribution. 

        We do not directly compute the posterior for any of these distributions
        We instead determine closeness qualitatively by sampling predictors from
        our hypernetwork, and comparing the resulting prediction statistics to 
        samples drawn from an HMC-based neural network. 
    """

    def generate_class_data(self, n_samples=200, num_classes=4):

        if num_classes == 4:
            means = [(2., 2.), (-2., 2.), (2., -2.), (-2., -2.)]
        else:
            means = [(2., 2.), (-2., -2.)]

        data = torch.zeros(n_samples, 2)
        labels = torch.zeros(n_samples)
        size = n_samples // len(means)
        for i, (x, y) in enumerate(means):
            dist = torch.distributions.Normal(torch.tensor([x, y]), .3)
            samples = dist.sample([size])
            data[size * i:size * (i + 1)] = samples
            labels[size * i:size * (i + 1)] = torch.ones(len(samples)) * i

        return data, labels.long()

    def plot_classification(self, i, algorithm, tag, n_classes):
        basedir = '/data/jerry/results/alf/gtk_ensemble/4_class/{}'.format(tag)
        os.makedirs(basedir, exist_ok=True)
        x = torch.linspace(-12, 12, 100)
        y = torch.linspace(-12, 12, 100)
        gridx, gridy = torch.meshgrid(x, y)
        grid = torch.stack((gridx.reshape(-1), gridy.reshape(-1)), -1)
        outputs = algorithm.predict_step(grid).output.cpu()
        if i >= 45000 and i <= 50000:
            print('saving')
            torch.save(outputs, basedir + '_final_outputs_{}.pt'.format(i))
        outputs = F.softmax(outputs, -1).detach()  # [B, D]

        mean_outputs = outputs.mean(1).cpu()  # [B, D]
        std_outputs = outputs.std(1).cpu()
        conf_mean = mean_outputs.mean(-1)
        conf_std = std_outputs.max(-1)[0] * 1.94

        labels = mean_outputs.argmax(-1)
        data, _ = self.generate_class_data(
            n_samples=400, num_classes=n_classes)

        p1 = plt.scatter(
            grid[:, 0].cpu(), grid[:, 1].cpu(), c=conf_std, cmap='rainbow')
        p2 = plt.scatter(
            data[:, 0].cpu(), data[:, 1].cpu(), c='black', alpha=0.1)
        cbar = plt.colorbar(p1)
        cbar.set_label("confidance (std)")
        plt.savefig(basedir + '_conf_map-std_{}.png'.format(i))
        plt.close('all')

        p1 = plt.scatter(
            grid[:, 0].cpu(), grid[:, 1].cpu(), c=labels, cmap='rainbow')
        p2 = plt.scatter(
            data[:, 0].cpu(), data[:, 1].cpu(), c='black', alpha=0.1)
        cbar = plt.colorbar(p1)
        cbar.set_label("predicted labels")
        plt.savefig(basedir + '_conf_map-labels_{}.png'.format(i))
        print('saved figure: ', basedir)
        plt.close('all')

    @parameterized.parameters(
        # (None, 0.1, False),
        (None, 0.5, True),
        (None, 0.2, True),
        # ('svgd', True),
        # ('gfsf', False),
        # ('gfsf', True),
        # ('minmax', True),
        # (None, False),
        # ('minmax', False)
    )
    def test_classification_func_par_vi(self,
                                        par_vi='svgd',
                                        capacity_ratio=1.,
                                        full_rank_update=False,
                                        function_vi=False,
                                        n_classes=4,
                                        num_particles=100):
        """
        Symmetric N-class classification problem. The training data are drawn
        from a mixture of normal distributions, the class index of a data point
        is given by the index of the component it was drawn from. 
        The particle ensemble is trained to achieve low loss / high accuracy
        on this data. The test shows the epistemic uncertainty of the ensemble
        up to 2 std. in the regions near and far from the data. 
        """

        print('{} - {} particles'.format(par_vi, num_particles))
        print('Functional Particle VI: Fitting {} Classes'.format(n_classes))
        print("Function VI: {}".format(function_vi))
        input_size = 2
        output_dim = 4
        input_spec = TensorSpec((input_size, ), torch.float64)
        test_nsamples = 200
        batch_size = 100
        train_batch_size = 100
        inputs, targets = self.generate_class_data(batch_size, n_classes)
        test_inputs, test_targets = self.generate_class_data(
            test_nsamples, n_classes)
        if n_classes == 4:
            param_dim = 184
        else:
            param_dim = 162

        algorithm = FuncParVIAlgorithm(
            input_tensor_spec=input_spec,
            output_dim=output_dim,
            fc_layer_params=(10, 10),
            use_fc_bias=True,
            last_activation=math_ops.identity,
            last_use_bias=True,
            num_particles=num_particles,
            loss_type='classification',
            par_vi=par_vi,
            function_vi=function_vi,
            function_bs=train_batch_size,
            function_extra_bs_ratio=0.1,
            critic_hidden_layers=(10, 10),
            critic_iter_num=5,
            critic_optimizer=alf.optimizers.Adam(lr=1e-3),
            optimizer=alf.optimizers.Adam(
                lr=1e-3,
                capacity_ratio=capacity_ratio,
                full_rank_update=full_rank_update))

        def _train(i, entropy_regularization=None):
            perm = torch.randperm(batch_size)
            idx = perm[:train_batch_size]
            train_inputs = inputs[idx]
            train_targets = targets[idx]
            if entropy_regularization is None:
                entropy_regularization = train_batch_size / batch_size

            alg_step = algorithm.train_step(
                inputs=(train_inputs, train_targets),
                entropy_regularization=entropy_regularization)

            loss_info, params = algorithm.update_with_gradient(alg_step.info)

        def _test(i):
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
            sample_acc = sample_acc.sum() / len(targets_unrolled)

            print('-' * 86)
            print('iter ', i)
            print('mean particle acc: ', mean_acc.item())
            print('all particles acc: ', sample_acc.item())

            # with torch.no_grad():
            #     if par_vi is None:
            #         tag = 'mle'
            #     else:
            #         tag = par_vi
            #     if function_vi:
            #         tag += 'f-vi'
            #     tag += '/{}_cls'.format(n_classes)
            #     self.plot_classification(i, algorithm, tag, n_classes)

        train_iter = 50000
        for i in range(train_iter):
            _train(i)
            if i % 5000 == 0:
                _test(i)

        with torch.no_grad():
            if par_vi is None:
                tag = 'mle'
            else:
                tag = par_vi
            if function_vi:
                tag += 'f-vi'
            if capacity_ratio < 1:
                tag += f'_r{capacity_ratio}'
            if full_rank_update:
                tag += '_fullrank'
            # tag += '/{}_cls'.format(n_classes)
            self.plot_classification(i, algorithm, tag, n_classes)

    def generate_regression_data(self, n_train, n_test):
        x_train1 = torch.linspace(-6, -2, n_train // 2).view(-1, 1)
        x_train2 = torch.linspace(2, 6, n_train // 2).view(-1, 1)
        x_train3 = torch.linspace(-2, 2, 4).view(-1, 1)
        x_train = torch.cat((x_train1, x_train2, x_train3), dim=0)
        y_train = -(1 + x_train) * torch.sin(1.2 * x_train)
        y_train = y_train + torch.ones_like(y_train).normal_(0, 0.04)

        x_test = torch.linspace(-6, 6, n_test).view(-1, 1)
        y_test = -(1 + x_test) * torch.sin(1.2 * x_test)
        y_test = y_test + torch.ones_like(y_test).normal_(0, 0.04)
        return (x_train, y_train), (x_test, y_test)

    def plot_bnn_regression(self, i, algorithm, data, tag):
        basedir = 'plots/regression/functional_par_vi/{}/'.format(tag)
        os.makedirs(basedir, exist_ok=True)
        sns.set_style('darkgrid')
        gt_x = torch.linspace(-6, 6, 500).view(-1, 1).cpu()
        gt_y = -(1 + gt_x) * torch.sin(1.2 * gt_x)
        (x_train, y_train), (x_test, y_test) = data
        outputs = algorithm.predict_step(x_test).output.cpu()
        mean = outputs.mean(1).squeeze()
        std = outputs.std(1).squeeze()
        x_test = x_test.cpu().numpy()
        x_train = x_train.cpu().numpy()

        plt.fill_between(
            x_test.squeeze(),
            mean.T + 2 * std.T,
            mean.T - 2 * std.T,
            alpha=0.5)
        plt.plot(gt_x, gt_y, color='red', label='ground truth')
        plt.plot(x_test, mean.T, label='posterior mean', alpha=0.9)
        plt.scatter(
            x_train,
            y_train.cpu().numpy(),
            color='r',
            marker='+',
            label='train pts',
            alpha=1.0,
            s=50)
        plt.legend(fontsize=18, loc='best')
        plt.ylim([-6, 8])
        plt.savefig(basedir + 'iter_{}.png'.format(i))
        plt.close('all')

    # @parameterized.parameters(
    #     ('svgd', False),
    #     # ('svgd', True),
    #     # ('gfsf', False),
    #     # ('gfsf', True),
    #     # (None, False),
    # )
    # def test_regression_func_par_vi(self,
    #                                 par_vi='svgd',
    #                                 function_vi=False,
    #                                 num_particles=100):
    #     n_train = 80
    #     n_test = 200
    #     input_size = 1
    #     output_dim = 1
    #     input_spec = TensorSpec((input_size, ), torch.float64)
    #     train_batch_size = n_train
    #     batch_size = n_train
    #
    #     train_samples, test_samples = self.generate_regression_data(
    #         n_train, n_test)
    #     inputs, targets = train_samples
    #     test_inputs, test_targets = test_samples
    #     print('{} - {} particles'.format(par_vi, num_particles))
    #     print('Functional Particle VI: Fitting Regressors')
    #     print("Function VI: {}".format(function_vi))
    #
    #     algorithm = FuncParVIAlgorithm(
    #         input_tensor_spec=input_spec,
    #         fc_layer_params=((50, True), ),
    #         last_layer_param=(output_dim, True),
    #         last_activation=math_ops.identity,
    #         num_particles=num_particles,
    #         loss_type='regression',
    #         par_vi=par_vi,
    #         function_vi=function_vi,
    #         function_bs=train_batch_size,
    #         optimizer=alf.optimizers.Adam(lr=1e-2))
    #
    #     def _train(entropy_regularization=None):
    #         train_inputs = inputs
    #         train_targets = targets
    #         if entropy_regularization is None:
    #             entropy_regularization = train_batch_size / batch_size
    #
    #         alg_step = algorithm.train_step(
    #             inputs=(train_inputs, train_targets),
    #             entropy_regularization=entropy_regularization)
    #
    #         loss_info, params = algorithm.update_with_gradient(alg_step.info)
    #
    #     def _test(i):
    #         outputs, _ = algorithm._param_net(test_inputs)
    #         mse_err = (outputs.mean(1) - test_targets).pow(2).mean()
    #         print('Expected MSE: {}'.format(mse_err))
    #
    #     for i in range(10000):
    #         _train()
    #         if i % 200 == 0:
    #             _test(i)
    #             with torch.no_grad():
    #                 data = (train_samples, test_samples)
    #                 tag = par_vi
    #                 if function_vi:
    #                     tag += '_fvi'
    #                 self.plot_bnn_regression(i, algorithm, data, tag)


if __name__ == "__main__":
    alf.test.main()
