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
import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable

import alf
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.config import TrainerConfig
from alf.algorithms.particle_vi_algorithm import ParVIAlgorithm
from alf.data_structures import AlgStep, LossInfo, namedtuple
from alf.networks import EncodingNetwork, ParamNetwork
from alf.tensor_specs import TensorSpec
from alf.nest.utils import get_outer_rank
from alf.utils import common, math_ops, summary_utils
from alf.utils.summary_utils import record_time

FuncParVILossInfo = namedtuple("FuncParVILossInfo", ["loss", "extra"])


def classification_loss(output, target):
    pred = output.max(-1)[1]
    acc = pred.eq(target).float().mean(0)
    avg_acc = acc.mean()
    loss = F.cross_entropy(output.transpose(1, 2), target)
    return FuncParVILossInfo(loss=loss, extra=avg_acc)


def regression_loss(output, target):
    out_shape = output.shape[-1]
    assert (target.shape[-1] == out_shape), (
        "feature dimension of output and target does not match.")
    loss = 0.5 * F.mse_loss(
        output.reshape(-1, out_shape),
        target.reshape(-1, out_shape),
        reduction='sum')
    return FuncParVILossInfo(loss=loss, extra=())


def _expand_to_replica(inputs, replicas, spec):
    """Expand the inputs of shape [B, ...] to [B, n, ...] if n > 1,
        where n is the number of replicas. When n = 1, the unexpanded
        inputs will be returned.
    Args:
        inputs (Tensor): the input tensor to be expanded
        spec (TensorSpec): the spec of the unexpanded inputs. It is used to
            determine whether the inputs is already an expanded one. If it
            is already expanded, inputs will be returned without any
            further processing.
    Returns:
        Tensor: the expaneded inputs or the original inputs.
    """
    outer_rank = get_outer_rank(inputs, spec)
    if outer_rank == 1 and replicas > 1:
        return inputs.unsqueeze(1).expand(-1, replicas, *inputs.shape[1:])
    else:
        return inputs


@gin.configurable
class FuncParVIAlgorithm(ParVIAlgorithm):
    """Functional ParVI Algorithm 

    Functional ParVI algorithm maintains a set of functional particles, 
    where each particle is a neural network. All particles are updated
    using particle-based VI approaches.

    There are two ways of treating a neural network as a particle: 

    * All the weights of the neural network as a particle.

    * Outputs of the neural network for an input mini-batch as a particle.  

    """

    def __init__(self,
                 input_tensor_spec=None,
                 param_net: ParamNetwork = None,
                 train_state_spec=(),
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu_,
                 last_layer_param=None,
                 last_activation=None,
                 num_particles=10,
                 entropy_regularization=1.,
                 loss_type="classification",
                 voting="soft",
                 par_vi="svgd",
                 function_vi=False,
                 function_bs=None,
                 function_extra_bs_ratio=0.1,
                 function_extra_bs_sampler='uniform',
                 function_extra_bs_std=1.,
                 optimizer=None,
                 logging_network=False,
                 logging_training=False,
                 logging_evaluate=False,
                 config: TrainerConfig = None,
                 debug_summaries=False,
                 name="FuncParVIAlgorithm"):
        """
        Args:
            Args for the each parametric network
            ====================================================================
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input. If nested, then ``preprocessing_combiner`` must not be
                None.
            param_net (ParamNetwork): input parametric network.
            train_state_spec (nested TensorSpec): for the network state of
                ``train_step()``.
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

            Args for the ensemble of particles
            ====================================================================
            num_particles (int): number of sampling particles
            entropy_regularization (float): weight of the repulsive term in par_vi. 

            Args for function_vi
            ====================================================================
            function_vi (bool): whether to use funciton value based par_vi, current
                supported by [``svgd2``, ``svgd3``, ``gfsf``].
            function_bs (int): mini batch size for par_vi training. 
                Needed for critic initialization when function_vi is True. 
            function_extra_bs_ratio (float): ratio of extra sampled batch size 
                w.r.t. the function_bs.
            function_extra_bs_sampler (str): type of sampling method for extra
                training batch, types are [``uniform``, ``normal``].
            function_extra_bs_std (float): std of the normal distribution for
                sampling extra training batch when using normal sampler.

            Args for training and testing
            ====================================================================
            loss_type (str): loglikelihood type for the generated functions,
                types are [``classification``, ``regression``]
            voting (str): types of voting results from sampled functions,
                types are [``soft``, ``hard``]
            par_vi (str): types of particle-based methods for variational inference,
                types are [``svgd``, ``gfsf``]
                * svgd: empirical expectation of SVGD is evaluated by reusing
                    the same batch of particles.   
                * gfsf: wasserstein gradient flow with smoothed functions. It 
                    involves a kernel matrix inversion, so computationally more
                    expensive, but in some cases the convergence seems faster 
                    than svgd approaches.
            function_vi (bool): whether to use function value based par_vi.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            logging_network (bool): whether logging the archetectures of networks.
            logging_training (bool): whether logging loss and acc during training.
            logging_evaluate (bool): whether logging loss and acc of evaluate.
            config (TrainerConfig): configuration for training
            name (str):
        """
        if param_net is None:
            assert input_tensor_spec is not None
            param_net = ParamNetwork(
                input_tensor_spec=input_tensor_spec,
                conv_layer_params=conv_layer_params,
                fc_layer_params=fc_layer_params,
                activation=activation,
                last_layer_param=last_layer_param,
                last_activation=last_activation)

        particle_dim = param_net.param_length

        if logging_network:
            logging.info("Each network")
            logging.info("-" * 68)
            logging.info(param_net)

        super().__init__(
            particle_dim,
            train_state_spec=train_state_spec,
            num_particles=num_particles,
            entropy_regularization=entropy_regularization,
            par_vi=par_vi,
            optimizer=optimizer,
            debug_summaries=debug_summaries,
            name=name)

        self._param_net = param_net
        self._param_net.set_parameters(self.particles.data, reinitialize=True)

        self._train_loader = None
        self._test_loader = None
        self._loss_type = loss_type
        self._logging_training = logging_training
        self._logging_evaluate = logging_evaluate
        self._config = config
        self._function_vi = function_vi

        if function_vi:
            assert function_bs is not None, (
                "need to specify batch_size of function outputs.")
            self._function_extra_bs = math.ceil(
                function_bs * function_extra_bs_ratio)
            self._function_extra_bs_sampler = function_extra_bs_sampler
            self._function_extra_bs_std = function_extra_bs_std

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

    def predict_step(self, inputs, params=None, state=None):
        """Predict ensemble outputs for inputs using the hypernetwork model.

        Args:
            inputs (Tensor): inputs to the ensemble of networks.
            params (Tensor): parameters of the ensemble of networks,
                if None, use self.particles.
            state: not used.

        Returns:
            AlgStep: outputs with shape (batch_size, self._param_net._output_spec.shape[0])
        """
        if params is None:
            params = self.particles
        self._param_net.set_parameters(params)
        outputs, _ = self._param_net(inputs)
        return AlgStep(output=outputs, state=(), info=())

    def train_iter(self, state=None):
        """Perform one epoch (iteration) of training.
        
        Args:
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
                alg_step = self.train_step((data, target), state=state)
                loss_info, params = self.update_with_gradient(alg_step.info)
                loss += loss_info.extra.loss
                if self._loss_type == 'classification':
                    avg_acc.append(alg_step.info.extra.extra)
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
                   entropy_regularization=None,
                   loss_mask=None,
                   state=None):
        """Perform one batch of training computation.

        Args:
            inputs (nested Tensor): input training data. 
            entropy_regularization (float): weight of the repulsive term in par_vi. 
                If None, use self._entropy_regularization.
            loss_mask (Tensor): mask indicating which samples are valid for 
                loss propagation.
            state: not used

        Returns:
            AlgStep:
                outputs: Tensor with shape (batch_size, dim)
                info: LossInfo
        """
        if entropy_regularization is None:
            entropy_regularization = self._entropy_regularization

        if self._function_vi:
            data, target = inputs
            return super().train_step(
                loss_func=functools.partial(self._function_neglogprob, target),
                transform_func=functools.partial(self._function_transform,
                                                 data),
                entropy_regularization=entropy_regularization,
                loss_mask=loss_mask,
                state=())
        else:
            return super().train_step(
                loss_func=functools.partial(self._neglogprob, inputs),
                entropy_regularization=entropy_regularization,
                state=())

    def _function_transform(self, data, params):
        """
        Transform the particles to its corresponding function values
        evaluated on the training batch. Used when function_vi is True.

        Args:
            data (torch.Tensor): training batch input.
            params (torch.Tensor): parameter tensor for param_net.

        Returns:
            outputs (torch.Tensor): outputs of param_net under params
                evaluated on data.
            density_outputs (torch.Tensor): outputs of param_net under
                params evaluated on sampled extra data.
        """
        # sample extra data
        if isinstance(params, tuple):
            params, extra_samples = params
        else:
            sample = data[-self._function_extra_bs:]
            noise = torch.zeros_like(sample)
            if self._function_extra_bs_sampler == 'uniform':
                noise.uniform_(0., 1.)
            else:
                noise.normal_(mean=0., std=self._function_extra_bs_std)
            extra_samples = sample + noise

        num_particles = params.shape[0]
        self._param_net.set_parameters(params)
        aug_data = torch.cat([data, extra_samples], dim=0)
        aug_outputs, _ = self._param_net(aug_data)  # [B+b, P, D]

        outputs = aug_outputs[:data.shape[0]]  # [B, P, D]
        outputs = outputs.transpose(0, 1)
        outputs = outputs.view(num_particles, -1)  # [P, B * D]

        density_outputs = aug_outputs[-extra_samples.shape[0]:]  # [b, P, D]
        density_outputs = density_outputs.transpose(0, 1)
        density_outputs = density_outputs.view(num_particles, -1)

        return outputs, density_outputs

    def _function_neglogprob(self, targets, outputs):
        """
        Function computing negative log_prob loss for function outputs.
        Used when function_vi is True.

        Args:
            targets (torch.Tensor): target values of the training batch.
            outputs (torch.Tensor): function outputs to evaluate the loss.

        Returns:
            negative log_prob for outputs evaluated on current training batch.
        """
        num_particles = outputs.shape[0]
        # [B, D] -> [B, N, D]
        targets = _expand_to_replica(targets, num_particles,
                                     self._param_net.output_spec)
        # [B, N, D] -> [N, B, D]
        targets = targets.permute(1, 0, 2)
        # [N, B, D] -> [N, -1]
        targets = targets.view(num_particles, -1)

        return self._loss_func(outputs, targets)

    def _neglogprob(self, inputs, params):
        """
        Function computing negative log_prob loss for generator outputs.
        Used when function_vi is False.

        Args:
            inputs (torch.Tensor): (data, target) of training batch.
            params (torch.Tensor): generator outputs to evaluate the loss.

        Returns:
            negative log_prob for params evaluated on current training batch.
        """
        self._param_net.set_parameters(params)
        num_particles = params.shape[0]
        data, target = inputs
        output, _ = self._param_net(data)  # [B, N, D]
        # [B, d] -> [B, N, d]
        target = _expand_to_replica(target, num_particles,
                                    self._param_net.output_spec)
        return self._loss_func(output, target)

    def evaluate(self):
        """Evaluate on a randomly drawn ensemble. """

        assert self._test_loader is not None, "Must set test_loader first."
        logging.info("==> Begin testing")
        self._param_net.set_parameters(self.particles)
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
