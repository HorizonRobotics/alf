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
"""ParticleVI algorithm on parameterized functions."""

from absl import logging
import functools
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
from alf.utils.sl_utils import classification_loss, regression_loss, auc_score
from alf.utils.sl_utils import predict_dataset


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
        Tensor: the expanded inputs or the original inputs.
    """
    outer_rank = get_outer_rank(inputs, spec)
    if outer_rank == 1 and replicas > 1:
        return inputs.unsqueeze(1).expand(-1, replicas, *inputs.shape[1:])
    else:
        return inputs


@alf.configurable
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
                 data_creator=None,
                 data_creator_outlier=None,
                 input_tensor_spec=None,
                 output_dim=None,
                 param_net: ParamNetwork = None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 use_conv_bias=False,
                 use_conv_ln=False,
                 use_fc_bias=True,
                 use_fc_ln=False,
                 activation=torch.relu_,
                 last_activation=math_ops.identity,
                 last_use_bias=True,
                 last_use_ln=False,
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
                 critic_hidden_layers=(100, 100),
                 critic_iter_num=2,
                 critic_l2_weight=10.,
                 critic_use_bn=True,
                 num_train_classes=10,
                 optimizer=None,
                 critic_optimizer=None,
                 logging_network=False,
                 logging_training=False,
                 logging_evaluate=False,
                 config: TrainerConfig = None,
                 debug_summaries=False,
                 name="FuncParVIAlgorithm"):
        """
        Args:
            data_creator (Callable): called as ``data_creator()`` to get a tuple
                of ``(train_dataloader, test_dataloader)``
            data_creator_outlier (Callable): called as ``data_creator()`` to get
                a tuple of ``(outlier_train_dataloader, outlier_test_dataloader)``
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input. If nested, then ``preprocessing_combiner`` must not be
                None. It must be provided if ``data_creator`` is not provided.
            output_dim (int): dimension of the output of the generated network.
                It must be provided if ``data_creator`` is not provided.
            param_net (ParamNetwork): input parametric network.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format
                ``(filters, kernel_size, strides, padding, pooling_kernel)``,
                where ``padding`` and ``pooling_kernel`` are optional.
            fc_layer_params (tuple[tuple]): a tuple of tuples where each tuple
                takes a format ``(FC layer sizes. use_bias)``, where
                ``use_bias`` is optional.
            use_conv_bias (bool|None): whether use bias for conv layers. If None, 
                will use ``not use_bn`` for conv layers.
            use_conv_ln (bool): whether use layer normalization for conv layers.
            use_fc_bias (bool): whether use bias for fc layers.
            use_fc_ln (bool): whether use layer normalization for fc layers.
            activation (Callable): activation used for all the layers but
                the last layer.
            last_activation (Callable): activation function of the
                additional layer specified by ``last_layer_param``. Note that if
                ``last_layer_param`` is not None, ``last_activation`` has to be
                specified explicitly.
            last_use_bias (bool): whether use bias for the last layer
            last_use_ln (bool): whether use normalization for the last layer.

            num_particles (int): number of sampling particles
            entropy_regularization (float): weight of the repulsive term in par_vi.

            function_vi (bool): whether to use function value based par_vi, current
                supported by [``svgd2``, ``svgd3``, ``gfsf``].
            function_bs (int): mini batch size for par_vi training.
                Needed for critic initialization when function_vi is True.
            function_extra_bs_ratio (float): ratio of extra sampled batch size
                w.r.t. the function_bs.
            function_extra_bs_sampler (str): type of sampling method for extra
                training batch, types are [``uniform``, ``normal``].
            function_extra_bs_std (float): std of the normal distribution for
                sampling extra training batch when using normal sampler.

            critic_hidden_layers (tuple): sizes of hidden layers of the critic,
                used for ``minmax``.
            critic_l2_weight (float): weight of L2 regularization in training
                the critic, used for ``minmax``.
            critic_iter_num (int): number of critic updates for each generator
                train_step, used for ``minmax``.
            critic_use_bn (book): whether use batch norm for each layers of the
                critic, used for ``minmax``.
            critic_optimizer (torch.optim.Optimizer): Optimizer for training the
                critic, used for ``minmax``.

            loss_type (str): loglikelihood type for the generated functions,
                types are [``classification``, ``regression``]
            voting (str): types of voting results from sampled functions,
                types are [``soft``, ``hard``]
            par_vi (str): types of particle-based methods for variational inference,
                types are [``svgd``, ``gfsf``, ``minmax``]

                * svgd: empirical expectation of SVGD is evaluated by reusing
                  the same batch of particles.
                * gfsf: wasserstein gradient flow with smoothed functions. It
                  involves a kernel matrix inversion, so computationally more
                  expensive, but in some cases the convergence seems faster
                  than svgd approaches.
            function_vi (bool): whether to use function value based par_vi.
            num_train_classes (int): number of classes in training set.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            logging_network (bool): whether logging the architectures of networks.
            logging_training (bool): whether logging loss and acc during training.
            logging_evaluate (bool): whether logging loss and acc of evaluate.
            config (TrainerConfig): configuration for training
            name (str):
        """
        if data_creator is not None:
            trainset, testset = data_creator()
            if data_creator_outlier is not None:
                outlier_dataloaders = data_creator_outlier()
            else:
                outlier_dataloaders = None
            self.set_data_loader(trainset, testset, outlier_dataloaders)
            input_tensor_spec = TensorSpec(shape=trainset.dataset[0][0].shape)
            if hasattr(trainset.dataset, 'classes'):
                output_dim = len(trainset.dataset.classes)
            else:
                output_dim = num_train_classes
            input_tensor_spec = input_tensor_spec
        else:
            assert input_tensor_spec is not None and output_dim is not None, (
                "input_tensor_spec and output_dim need to be provided if "
                "data_creator is not provided")
            self._train_loader = None
            self._test_loader = None

        last_layer_size = output_dim

        if param_net is None:
            assert input_tensor_spec is not None
            param_net = ParamNetwork(
                input_tensor_spec=input_tensor_spec,
                conv_layer_params=conv_layer_params,
                fc_layer_params=fc_layer_params,
                use_conv_bias=use_conv_bias,
                use_conv_ln=use_conv_ln,
                use_fc_bias=use_fc_bias,
                use_fc_ln=use_fc_ln,
                n_groups=num_particles,
                activation=activation,
                last_layer_size=last_layer_size,
                last_activation=last_activation,
                last_use_bias=last_use_bias,
                last_use_ln=last_use_ln)

        particle_dim = param_net.param_length

        if logging_network:
            logging.info("Each network")
            logging.info("-" * 68)
            logging.info(param_net)

        super().__init__(
            particle_dim,
            num_particles=num_particles,
            entropy_regularization=entropy_regularization,
            par_vi=par_vi,
            critic_hidden_layers=critic_hidden_layers,
            critic_l2_weight=critic_l2_weight,
            critic_iter_num=critic_iter_num,
            critic_use_bn=critic_use_bn,
            critic_optimizer=critic_optimizer,
            optimizer=optimizer,
            debug_summaries=debug_summaries,
            name=name)

        self._param_net = param_net
        self._param_net.set_parameters(self.particles.data, reinitialize=True)

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

    def set_data_loader(self,
                        train_loader,
                        test_loader=None,
                        outlier_data_loaders=None,
                        entropy_regularization=None):
        """Set data loadder for training and testing.

        Args:
            train_loader (torch.utils.data.DataLoader): training data loader
            test_loader (torch.utils.data.DataLoader): testing data loader
            outlier_data_loaders (tuple[torch.utils.data.DataLoader):
                (trainloader, testloader) for outlier datasets
            entropy_regularization (float): weight of particle VI repulsive
                term.
        """
        self._train_loader = train_loader
        self._test_loader = test_loader
        if entropy_regularization is not None:
            self._entropy_regularization = entropy_regularization

        if outlier_data_loaders is not None:
            assert isinstance(outlier_data_loaders, tuple), "outlier dataset "\
                "must be provided in the format (outlier_train, outlier_test)"
            self._outlier_train_loader = outlier_data_loaders[0]
            self._outlier_test_loader = outlier_data_loaders[1]
        else:
            self._outlier_train_loader = self._outlier_test_loader = None

    def predict_step(self, inputs, params=None, state=None):
        """Predict ensemble outputs for inputs using the hypernetwork model.

        Args:
            inputs (Tensor): inputs to the ensemble of networks.
            params (Tensor): parameters of the ensemble of networks,
                if None, use self.particles.
            state (None): not used.

        Returns:
            AlgStep:
            - output (Tensor): predictions with shape
                ``[batch_size, self._param_net._output_spec.shape[0]]``
            - state (None): not used
        """
        if params is None:
            params = self.particles
        self._param_net.set_parameters(params)
        outputs, _ = self._param_net(inputs)
        return AlgStep(output=outputs, state=(), info=())

    def train_iter(self, state=None):
        """Perform one epoch (iteration) of training.

        Args:
            state (None): not used

        Returns:
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
            state (None): not used

        Returns:
            AlgStep:
            - output(Tensor): shape is ``[batch_size, dim]``
            - state: not used
            - info (LossInfo): loss
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
            data (Tensor): training batch input.
            params (Tensor): parameter tensor for param_net.

        Returns:
            outputs (Tensor): outputs of param_net under params
                evaluated on data.
            density_outputs (Tensor): outputs of param_net under
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
        density_outputs = density_outputs.transpose(0, 1)  # [P, b, D]
        density_outputs = density_outputs.view(num_particles, -1)  # [P, b * D]

        return outputs, density_outputs

    def _function_neglogprob(self, targets, outputs):
        """Function computing negative log_prob loss for function outputs.
        Used when function_vi is True.

        Args:
            targets (Tensor): target values of the training batch.
            outputs (Tensor): function outputs to evaluate the loss.

        Returns:
            Negative log_prob for outputs evaluated on current training batch.
        """
        num_particles = outputs.shape[0]
        if self._loss_type == 'regression':
            # [B, D] -> [B, N, D]
            targets = _expand_to_replica(targets, num_particles,
                                         self._param_net.output_spec)
            # [B, N, D] -> [N, B, D]
            targets = targets.permute(1, 0, 2)
            # [N, B, D] -> [N, -1]
            targets = targets.view(num_particles, -1)
        else:
            # [B] -> [B, 1]
            targets = targets.unsqueeze(1)
            # [B, 1] -> [N, B, 1]
            targets = targets.unsqueeze(0).expand(num_particles,
                                                  *targets.shape)

        return self._loss_func(outputs, targets)

    def _neglogprob(self, inputs, params):
        """Function computing negative log_prob loss for generator outputs.
        Used when function_vi is False.

        Args:
            inputs (Tensor): (data, target) of training batch.
            params (Tensor): generator outputs to evaluate the loss.

        Returns:
            Negative log_prob for params evaluated on current training batch.
        """
        self._param_net.set_parameters(params)
        num_particles = params.shape[0]
        data, target = inputs
        output, _ = self._param_net(data)  # [B, N, D]
        if self._loss_type == 'regression':
            # [B, d] -> [B, N, d]
            target = _expand_to_replica(target, num_particles,
                                        self._param_net.output_spec)
        else:
            # [B] -> [B, N]
            target = target.unsqueeze(1).expand(*target.shape[:1],
                                                num_particles)

        return self._loss_func(output, target)

    def evaluate(self):
        """Evaluatation of the ParVI ensemble on a test dataset."""

        assert self._test_loader is not None, "Must set test_loader first."
        logging.info("==> Begin evaluating")
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
        """Ensemble the outputs from sampled classifiers."""
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
        """Ensemble the outputs for sampled regressors."""
        num_particles = output.shape[1]
        pred = output.mean(1)  # [B, D]
        loss = regression_loss(pred, target)
        target = target.unsqueeze(1).expand(*target.shape[:1], num_particles,
                                            *target.shape[1:])
        total_loss = regression_loss(output, target)
        return loss, total_loss

    def eval_uncertainty(self):
        """Function to evaluate the epistemic uncertainty of the ensemble.
        This method computes the following metrics:

        * AUROC (AUC) evaluates the separability of model predictions with
          respect to the training data and a prespecified outlier dataset.
          AUC is computed with respect to the entropy in the averaged
          softmax probabilities, as well as the sum of the variance of
          the softmax probabilities over the ensemble.
        """

        with torch.no_grad():
            outputs = predict_dataset(self._param_net, self._test_loader)
            outputs_outlier = predict_dataset(self._param_net,
                                              self._outlier_test_loader)
        probs = F.softmax(outputs, -1)
        probs_outlier = F.softmax(outputs_outlier, -1)

        mean_probs = probs.mean(0)
        mean_probs_outlier = probs_outlier.mean(0)

        entropy = torch.distributions.Categorical(mean_probs).entropy()
        entropy_outlier = torch.distributions.Categorical(
            mean_probs_outlier).entropy()

        variance = F.softmax(outputs, -1).var(0).sum(-1)
        variance_outlier = F.softmax(outputs_outlier, -1).var(0).sum(-1)

        auroc_entropy = auc_score(entropy, entropy_outlier)
        auroc_variance = auc_score(variance, variance_outlier)
        logging.info("AUROC score (entropy): {}".format(auroc_entropy))
        logging.info("AUROC score (variance): {}".format(auroc_variance))
        alf.summary.scalar(name='eval/auroc_entropy', data=auroc_entropy)
        alf.summary.scalar(name='eval/auroc_variance', data=auroc_variance)
        return auroc_entropy, auroc_variance

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
        if self._config is not None:
            if self._config.summarize_grads_and_vars:
                summary_utils.summarize_variables(params)
                summary_utils.summarize_gradients(params)
            if self._config.debug_summaries:
                summary_utils.summarize_loss(loss_info)
        if cum_loss is not None:
            alf.summary.scalar(name='train_epoch/neglogprob', data=cum_loss)
        if avg_acc is not None:
            alf.summary.scalar(name='train_epoch/avg_acc', data=avg_acc)
