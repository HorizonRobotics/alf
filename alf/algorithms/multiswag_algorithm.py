# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""MultiSwag algorithm implemented based on FuncParVIAlgorithm."""

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
from alf.algorithms.multiswag_subspace import Subspace
from alf.algorithms.functional_particle_vi_algorithm import FuncParVIAlgorithm
from alf.algorithms.functional_particle_vi_algorithm import _expand_to_replica
from alf.data_structures import AlgStep, LossInfo, namedtuple
from alf.networks import Network, EncodingNetwork, ParamNetwork
from alf.tensor_specs import TensorSpec
from alf.nest.utils import get_outer_rank
from alf.utils import common, math_ops, summary_utils
from alf.utils.summary_utils import record_time, safe_mean_hist_summary
from alf.utils.sl_utils import classification_loss, regression_loss, auc_score
from alf.utils.sl_utils import predict_dataset


@alf.configurable
class MultiSwagAlgorithm(FuncParVIAlgorithm):
    """MultiSwagAlgorithm, described in:

    ::

        Willson and Izmailov. "Bayesian Deep Learning and a Probabilistic 
        Perspective of Generalization", arXiv:2002.08791

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
                 num_samples_per_model=2,
                 subspace_after_update_steps=10000,
                 subspace="covariance",
                 subspace_max_rank=20,
                 loss_type="classification",
                 voting="soft",
                 num_train_classes=10,
                 optimizer=None,
                 logging_network=False,
                 logging_training=False,
                 logging_evaluate=False,
                 config: TrainerConfig = None,
                 debug_summaries=False,
                 name="MultiSwagAlgorithm"):
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
            param_net (Network): input param network.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format
                ``(filters, kernel_size, strides, padding, pooling_kernel)``,
                where ``padding`` and ``pooling_kernel`` are optional.
            fc_layer_params (tuple[tuple]): a tuple of tuples where each tuple
                takes a format ``(FC layer sizes. use_bias)``, where
                ``use_bias`` is optional.
            use_conv_bias (bool|None): whether use bias for conv layers. If None, 
                will use ``not use_bn`` for conv layers.
            use_conv_bn (bool): whether use batch normalization for conv layers.
            use_fc_bias (bool): whether use bias for fc layers.
            use_fc_bn (bool): whether use batch normalization for fc layers.
            activation (Callable): activation used for all the layers but
                the last layer.
            last_activation (Callable): activation function of the
                additional layer specified by ``last_layer_param``. Note that if
                ``last_layer_param`` is not None, ``last_activation`` has to be
                specified explicitly.
            last_use_bias (bool): whether use bias for the last layer
            last_use_bn (bool): whether use batch normalization for the last layer.

            num_particles (int): number of SWAG models.
            num_samples_per_model (int): number of samples for each SWAG model.
            subspace_after_update_steps (int): SWAG subspaces start only after
                so many number of update steps (according to ``global_counter``).
            var_clamp (float): clamp threshold of variance.
            subspace (str): types of subspace construction methods,
                types are [``random``, ``covariance``, ``pca``, ``freq_dir``]

                * random: empirical expectation of SVGD is evaluated by reusing
                * covariance: wasserstein gradient flow with smoothed functions. It
                * pca:
                * freq_dir:
            subspace_max_rank (int): max rank of SWAG subspace.

            loss_type (str): loglikelihood type for the generated functions,
                types are [``classification``, ``regression``]
            voting (str): types of voting results from sampled functions,
                types are [``soft``, ``hard``]
            num_train_classes (int): number of classes in training set.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            logging_network (bool): whether logging the archetectures of networks.
            logging_training (bool): whether logging loss and acc during training.
            logging_evaluate (bool): whether logging loss and acc of evaluate.
            config (TrainerConfig): configuration for training
            name (str):
        """
        super().__init__(
            data_creator=data_creator,
            data_creator_outlier=data_creator_outlier,
            input_tensor_spec=input_tensor_spec,
            output_dim=output_dim,
            param_net=param_net,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            use_conv_bias=use_conv_bias,
            use_conv_ln=use_conv_ln,
            use_fc_bias=use_fc_bias,
            use_fc_ln=use_fc_ln,
            activation=activation,
            last_activation=last_activation,
            last_use_bias=last_use_bias,
            last_use_ln=last_use_ln,
            num_particles=num_particles,
            loss_type=loss_type,
            par_vi=None,
            function_vi=False,
            num_train_classes=num_train_classes,
            optimizer=optimizer,
            config=config,
            logging_network=logging_network,
            logging_training=logging_training,
            logging_evaluate=logging_evaluate,
            debug_summaries=debug_summaries,
            name=name)

        self._num_samples_per_model = num_samples_per_model
        self._subspace_after_update_steps = subspace_after_update_steps
        self._subspace_max_rank = subspace_max_rank

        self._subspaces = []
        for _ in range(num_particles):
            self._subspaces.append(
                Subspace.create(
                    subspace,
                    num_parameters=self.particle_dim,
                    max_rank=subspace_max_rank))

    @property
    def num_models(self):
        return self._num_particles

    @property
    def num_basins(self):
        return self._num_particles

    def get_particles(self):
        if self._subspaces[0].rank > 0:
            return self._sample_subspace(1, use_subspace_mean=True)
        else:
            return self.particles

    def _sample_subspace(self,
                         sample_size,
                         scale=0.5,
                         diag_noise=True,
                         use_subspace_mean=None):
        samples = []
        for i in range(self.num_models):
            # append a tensor of [n_sample, n_params]
            samples.append(self._subspaces[i].sample(
                sample_size, use_subspace_mean=use_subspace_mean))

        return torch.cat(samples, dim=0)  # [n_models * n_sample, n_params]

    def predict_step(self,
                     inputs,
                     training=False,
                     sample_size=None,
                     state=None):
        """Predict base_model or ensemble outputs for inputs.

        Args:
            inputs (Tensor): inputs to the ensemble of networks.
            training (bool): whether the prediction is used for training.
            sample_size (int|None): size of sampled ensemble for prediction.
                If None (default), use the base models. Otherwise, sample
                ``sample_size`` many models from each swag model.
            state (None): not used.

        Returns:
            AlgStep:
            - output (Tensor): if ``sample_size`` is None, with shape
                ``[B, D]``, otherwise, with shape ``[B, D]`` (n=1) or 
                ``[B, n, D]``, where the meanings of symbols are:

                - B: batch size
                - n: sample_size
                - D: output dimension

            - state (None): not used
        """
        if training or self._subspaces[0].rank < self._subspace_max_rank:
            return super().predict_step(inputs)
        else:
            if sample_size is None:
                sample_size = self._num_samples_per_model
            params = self._sample_subspace(
                sample_size)  # [n_model * n, n_params]
            self.param_net.set_parameters(params)
            # [bs, n_model * n, n_out] or [bs, n_model * n]
            outputs, _ = self.param_net(inputs)

            return AlgStep(output=outputs, state=(), info=())

    def train_iter(self, update_subspace=False, state=None):
        train_steps = super().train_iter(state=state)
        if update_subspace:
            self._update_subspace()
        return train_steps

    def _update_subspace(self):
        cur_weights = self.particles.detach()
        for i in range(self.num_models):
            self._subspaces[i].update(cur_weights[i])
            if self._debug_summaries and alf.summary.should_record_summaries():
                with alf.summary.scope(self._name):
                    safe_mean_hist_summary('subspace_var/' + str(i),
                                           self._subspaces[i].variance)

    def update_with_gradient(self,
                             loss_info,
                             valid_masks=None,
                             weight=1.0,
                             batch_info=None):
        loss_info, all_params = super().update_with_gradient(
            loss_info,
            valid_masks=valid_masks,
            weight=weight,
            batch_info=batch_info)
        if alf.summary.get_global_counter(
        ) > self._subspace_after_update_steps:
            self._update_subspace()

        return loss_info, all_params
