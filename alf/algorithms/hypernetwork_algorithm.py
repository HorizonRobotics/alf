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

from absl import logging
import functools
import gin
import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable

import alf
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.config import TrainerConfig
from alf.data_structures import AlgStep, LossInfo, namedtuple
from alf.algorithms.generator import Generator
from alf.algorithms.hypernetwork_networks import ParamNetwork
from alf.networks import EncodingNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import common, math_ops, summary_utils
from alf.utils.summary_utils import record_time

HyperNetworkLossInfo = namedtuple("HyperNetworkLossInfo", ["loss", "extra"])


def classification_loss(output, target):
    pred = output.max(-1)[1]
    acc = pred.eq(target).float().mean(0)
    avg_acc = acc.mean()
    loss = F.cross_entropy(output.transpose(1, 2), target)
    return HyperNetworkLossInfo(loss=loss, extra=avg_acc)


def regression_loss(output, target):
    out_shape = output.shape[-1]
    assert (target.shape[-1] == out_shape), (
        "feature dimension of output and target does not match.")
    loss = 0.5 * F.mse_loss(
        output.reshape(-1, out_shape),
        target.reshape(-1, out_shape),
        reduction='sum')
    return HyperNetworkLossInfo(loss=loss, extra=())


@gin.configurable
class HyperNetwork(Algorithm):
    """HyperNetwork 

    HyperNetwork algorithm maintains a generator that generates a set of 
    parameters for a predefined neural network from a random noise input. 
    It is based on the following work:

    https://github.com/neale/HyperGAN

    Ratzlaff and Fuxin. "HyperGAN: A Generative Model for Diverse, 
    Performant Neural Networks." International Conference on Machine Learning. 2019.

    Major differences versus the original paper are:

    * A single generator that generates parameters for all network layers.

    * Remove the mixer and the discriminator.

    * The generator is trained with Amortized particle-based variational 
      inference (ParVI) methods, please refer to generator.py for details.

    """

    def __init__(self,
                 input_tensor_spec,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu_,
                 last_layer_param=None,
                 last_activation=None,
                 noise_dim=32,
                 hidden_layers=(64, 64),
                 use_fc_bn=False,
                 num_particles=10,
                 entropy_regularization=1.,
                 loss_type="classification",
                 voting="soft",
                 par_vi="svgd",
                 optimizer=None,
                 logging_network=False,
                 logging_training=False,
                 logging_evaluate=False,
                 config: TrainerConfig = None,
                 name="HyperNetwork"):
        """
        Args:
            Args for the generated parametric network
            ====================================================================
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input. If nested, then ``preprocessing_combiner`` must not be
                None.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format 
                ``(filters, kernel_size, strides, padding, pooling_kernel)``,
                where ``padding`` and ``pooling_kernel`` are optional.
            fc_layer_params (tuple[tuple]): a tuple of tuples where each tuple
                takes a format ``(FC layer sizes. use_bias)``, where 
                ``use_bias`` is optional.
            activation (nn.functional): activation used for all the layers but
                the last layer.
            last_layer_param (tuple): an optional tuple of the format
                ``(size, use_bias)``, where ``use_bias`` is optional,
                it appends an additional layer at the very end. 
                Note that if ``last_activation`` is specified, 
                ``last_layer_param`` has to be specified explicitly.
            last_activation (nn.functional): activation function of the
                additional layer specified by ``last_layer_param``. Note that if
                ``last_layer_param`` is not None, ``last_activation`` has to be
                specified explicitly.

            Args for the generator
            ====================================================================
            noise_dim (int): dimension of noise
            hidden_layers (tuple): size of hidden layers.
            use_fc_bn (bool): whether use batnch normalization for fc layers.
            num_particles (int): number of sampling particles
            entropy_regularization (float): weight of entropy regularization

            Args for training and testing
            ====================================================================
            loss_type (str): loglikelihood type for the generated functions,
                types are [``classification``, ``regression``]
            voting (str): types of voting results from sampled functions,
                types are [``soft``, ``hard``]
            par_vi (str): types of particle-based methods for variational inference,
                types are [``svgd``, ``svgd2``, ``svgd3``, ``gfsf``]
            optimizer (torch.optim.Optimizer): The optimizer for training.
            logging_network (bool): whether logging the archetectures of networks.
            logging_training (bool): whether logging loss and acc during training.
            logging_evaluate (bool): whether logging loss and acc of evaluate.
            config (TrainerConfig): configuration for training
            name (str):
        """
        super().__init__(train_state_spec=(), optimizer=optimizer, name=name)

        param_net = ParamNetwork(
            input_tensor_spec=input_tensor_spec,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            activation=activation,
            last_layer_param=last_layer_param,
            last_activation=last_activation)

        gen_output_dim = param_net.param_length
        noise_spec = TensorSpec(shape=(noise_dim, ))
        net = EncodingNetwork(
            noise_spec,
            fc_layer_params=hidden_layers,
            use_fc_bn=use_fc_bn,
            last_layer_size=gen_output_dim,
            last_activation=math_ops.identity,
            name="Generator")

        if logging_network:
            logging.info("Generated network")
            logging.info("-" * 68)
            logging.info(param_net)

            logging.info("Generator network")
            logging.info("-" * 68)
            logging.info(net)

        if par_vi == 'svgd':
            par_vi = 'svgd3'

        self._generator = Generator(
            gen_output_dim,
            noise_dim=noise_dim,
            net=net,
            entropy_regularization=entropy_regularization,
            par_vi=par_vi,
            optimizer=None,
            name=name)

        self._param_net = param_net
        self._num_particles = num_particles
        self._entropy_regularization = entropy_regularization
        self._train_loader = None
        self._test_loader = None
        self._use_fc_bn = use_fc_bn
        self._loss_type = loss_type
        self._logging_training = logging_training
        self._logging_evaluate = logging_evaluate
        self._config = config
        assert (voting in ['soft',
                           'hard']), ('voting only supports "soft" and "hard"')
        self._voting = voting
        if loss_type == 'classification':
            self._loss_func = classification_loss
            self._vote = self._classification_vote
        elif loss_type == 'regression':
            self._loss_func = regression_loss
            self._vote = self._regression_vote
        else:
            raise ValueError("Unsupported loss_type: %s" % loss_type)

    def set_data_loader(self, train_loader, test_loader=None):
        """Set data loadder for training and testing.

        Args:
            train_loader (torch.utils.data.DataLoader): training data loader
            test_loader (torch.utils.data.DataLoader): testing data loader
        """
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._entropy_regularization = 1 / len(train_loader)

    def set_num_particles(self, num_particles):
        """Set the number of particles to sample through one forward
        pass of the hypernetwork. """
        self._num_particles = num_particles

    @property
    def num_particles(self):
        """number of sampled particles. """
        return self._num_particles

    def sample_parameters(self, noise=None, num_particles=None, training=True):
        """Sample parameters for an ensemble of networks. 
        
        Args:
            noise (Tensor): input noise to self._generator. Default is None.
            num_particles (int): number of sampled particles. Default is None.
                If both noise and num_particles are None, num_particles
                provided to the constructor will be used as batch_size for 
                self._generator.
            training (bool): whether or not training self._generator

        Returns:
            AlgStep.output from predict_step of self._generator
        """
        if noise is None and num_particles is None:
            num_particles = self.num_particles
        generator_step = self._generator.predict_step(
            noise=noise, batch_size=num_particles, training=training)
        return generator_step.output

    def predict_step(self, inputs, params=None, num_particles=None,
                     state=None):
        """Predict ensemble outputs for inputs using the hypernetwork model.

        Args:
            inputs (Tensor): inputs to the ensemble of networks.
            params (Tensor): parameters of the ensemble of networks,
                if None, will resample.
            num_particles (int): size of sampled ensemble. Default is None.
            state: not used.

        Returns:
            AlgStep: outputs with shape (batch_size, self._param_net._output_spec.shape[0])
        """
        if params is None:
            params = self.sample_parameters(num_particles=num_particles)
        self._param_net.set_parameters(params)
        outputs, _ = self._param_net(inputs)
        return AlgStep(output=outputs, state=(), info=())

    def train_iter(self, num_particles=None, state=None):
        """Perform one epoch (iteration) of training.
        
        Args:
            num_particles (int): number of sampled particles. Default is None.
            state: not used

        Return:
            mini_batch number
        """

        assert self._train_loader is not None, "Must set data_loader first."
        alf.summary.increment_global_counter()
        with record_time("time/train"):
            loss = 0.
            if self._loss_type == 'classification':
                avg_acc = []
            for batch_idx, (data, target) in enumerate(self._train_loader):
                data = data.to(alf.get_default_device())
                target = target.to(alf.get_default_device())
                alg_step = self.train_step((data, target),
                                           num_particles=num_particles,
                                           state=state)
                loss_info, params = self.update_with_gradient(alg_step.info)
                loss += loss_info.extra.generator.loss
                if self._loss_type == 'classification':
                    avg_acc.append(alg_step.info.extra.generator.extra)
        acc = None
        if self._loss_type == 'classification':
            acc = torch.as_tensor(avg_acc).mean() * 100
        if self._logging_training:
            if self._loss_type == 'classification':
                logging.info("Avg acc: {}".format(acc))
            logging.info("Cum loss: {}".format(loss))
        self.summarize_train(loss_info, params, cum_loss=loss, avg_acc=acc)

        return batch_idx + 1

    def train_step(self,
                   inputs,
                   num_particles=None,
                   entropy_regularization=None,
                   state=None):
        """Perform one batch of training computation.

        Args:
            inputs (nested Tensor): input training data. 
            num_particles (int): number of sampled particles. Default is None,
                in which case self._num_particles will be used for batch_size
                of self._generator.
            state: not used

        Returns:
            AlgStep:
                outputs: Tensor with shape (batch_size, dim)
                info: LossInfo
        """
        if num_particles is None:
            num_particles = self._num_particles
        if entropy_regularization is None:
            entropy_regularization = self._entropy_regularization

        return self._generator.train_step(
            inputs=None,
            loss_func=functools.partial(self._neglogprob, inputs),
            batch_size=num_particles,
            entropy_regularization=entropy_regularization,
            state=())

    def evaluate(self, num_particles=None):
        """Evaluate on a randomly drawn ensemble. 

        Args:
            num_particles (int): number of sampled particles. Default is None.
        """

        assert self._test_loader is not None, "Must set test_loader first."
        logging.info("==> Begin testing")
        if self._use_fc_bn:
            self._generator.eval()
        params = self.sample_parameters(num_particles=num_particles)
        self._param_net.set_parameters(params)
        if self._use_fc_bn:
            self._generator.train()
        with record_time("time/test"):
            if self._loss_type == 'classification':
                test_acc = 0.
            test_loss = 0.
            for i, (data, target) in enumerate(self._test_loader):
                data = data.to(alf.get_default_device())
                target = target.to(alf.get_default_device())
                output, _ = self._param_net(data)  # [B, N, D]
                loss, extra = self._vote(output, target)
                if self._loss_type == 'classification':
                    test_acc += extra.item()
                test_loss += loss.loss.item()

        if self._loss_type == 'classification':
            test_acc /= len(self._test_loader.dataset)
            alf.summary.scalar(name='eval/test_acc', data=test_acc * 100)
        if self._logging_evaluate:
            if self._loss_type == 'classification':
                logging.info("Test acc: {}".format(test_acc * 100))
            logging.info("Test loss: {}".format(test_loss))
        alf.summary.scalar(name='eval/test_loss', data=test_loss)

    def _neglogprob(self, inputs, params):
        self._param_net.set_parameters(params)
        num_particles = params.shape[0]
        data, target = inputs
        output, _ = self._param_net(data)  # [B, N, D]
        target = target.unsqueeze(1).expand(*target.shape[:1], num_particles,
                                            *target.shape[1:])
        return self._loss_func(output, target)

    def _classification_vote(self, output, target):
        """ensmeble the ooutputs from sampled classifiers."""
        num_particles = output.shape[1]
        probs = F.softmax(output, dim=-1)  # [B, N, D]
        if self._voting == 'soft':
            pred = probs.mean(1).cpu()  # [B, D]
            vote = pred.argmax(-1)
        elif self._voting == 'hard':
            pred = probs.argmax(-1).cpu()  # [B, N, 1]
            vote = []
            for i in range(pred.shape[0]):
                values, counts = torch.unique(
                    pred[i], sorted=False, return_counts=True)
                modes = (counts == counts.max()).nonzero()
                label = values[torch.randint(len(modes), (1, ))]
                vote.append(label)
            vote = torch.as_tensor(vote, device='cpu')
        correct = vote.eq(target.cpu().view_as(vote)).float().cpu().sum()
        target = target.unsqueeze(1).expand(*target.shape[:1], num_particles,
                                            *target.shape[1:])
        loss = classification_loss(output, target)
        return loss, correct

    def _regression_vote(self, output, target):
        """ensemble the outputs for sampled regressors."""
        num_particles = output.shape[1]
        pred = output.mean(1)  # [B, D]
        loss = regression_loss(pred, target)
        target = target.unsqueeze(1).expand(*target.shape[:1], num_particles,
                                            *target.shape[1:])
        total_loss = regression_loss(output, target)
        return loss, total_loss

    def summarize_train(self, loss_info, params, cum_loss=None, avg_acc=None):
        """Generate summaries for training & loss info after each gradient update.
        The default implementation of this function only summarizes params
        (with grads) and the loss. An algorithm can override this for additional
        summaries. See ``RLAlgorithm.summarize_train()`` for an example.

        Args:
            experience (nested Tensor): samples used for the most recent
                ``update_with_gradient()``. By default it's not summarized.
            train_info (nested Tensor): ``AlgStep.info`` returned by either
                ``rollout_step()`` (on-policy training) or ``train_step()``
                (off-policy training). By default it's not summarized.
            loss_info (LossInfo): loss
            params (list[Parameter]): list of parameters with gradients
        """
        if self._config.summarize_grads_and_vars:
            summary_utils.summarize_variables(params)
            summary_utils.summarize_gradients(params)
        if self._config.debug_summaries:
            summary_utils.summarize_loss(loss_info)
        if cum_loss is not None:
            alf.summary.scalar(name='train_epoch/neglogprob', data=cum_loss)
        if avg_acc is not None:
            alf.summary.scalar(name='train_epoch/avg_acc', data=avg_acc)
