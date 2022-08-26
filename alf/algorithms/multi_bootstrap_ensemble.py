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
"""MultiBootstrap ensemble implemented based on FuncParVIAlgorithm."""

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
from alf.utils.summary_utils import record_time
from alf.utils.sl_utils import classification_loss, regression_loss, auc_score
from alf.utils.sl_utils import predict_dataset

MbeInfo = namedtuple("MbeInfo",
                     ["total_std", "opt_std"],
                     default_value=())

@alf.configurable 
class MultiBootstrapEnsemble(FuncParVIAlgorithm):
    """MultiBootstrapEnsemble

    It maintains an ensemble of functional particles of size num_basins times
    num_particles_per_basin. All functions share the same network structure.
    Two caveats for initializaing and training the ensemble:

    * Functional particles of the same basin are initialized with a same set 
      of parameters, but trained to follow different SGD paths by feeding in 
      different stochastic training batches at each SGD step.

    * Functional particles of different basins are initialized differently.

    The key insigts are as follows:

    * Predictive variance among the whole ensemble captures epistemic plus
      optimization uncertainty of the network model.

    * Predictive variance among members within a basin captures mainly
      optimization uncertainty of the corresponding basin of the network 
      model.

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
                 num_basins=5,
                 num_particles_per_basin=4,
                 loss_type="classification",
                 voting="soft",
                 num_train_classes=10,
                 optimizer=None,
                 initial_train_steps=0,
                 logging_network=False,
                 logging_training=False,
                 logging_evaluate=False,
                 config: TrainerConfig = None,
                 debug_summaries=False,
                 name="MultiBootstrapEnsemble"):
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

            num_basins (int): number basins (different particle initializations)
                to explore for the function space.
            num_particles_per_basin (int): number of particles to explore within 
                each basin.

            loss_type (str): loglikelihood type for the generated functions,
                types are [``classification``, ``regression``]
            voting (str): types of voting results from sampled functions,
                types are [``soft``, ``hard``]
            num_train_classes (int): number of classes in training set.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            initial_train_steps (int): if positive, number of steps that the 
                algorithm is trained with preprocessed inputs before regular 
                train_step.
            logging_network (bool): whether logging the archetectures of networks.
            logging_training (bool): whether logging loss and acc during training.
            logging_evaluate (bool): whether logging loss and acc of evaluate.
            config (TrainerConfig): configuration for training
            name (str):
        """
        if param_net is None:
            assert input_tensor_spec is not None and output_dim is not None, (
                "input_tensor_spec and output_dim need to be provided if "
                "both data_creator and param_net are not provided")
            last_layer_size = output_dim
            param_net = ParamNetwork(
                input_tensor_spec=input_tensor_spec,
                conv_layer_params=conv_layer_params,
                fc_layer_params=fc_layer_params,
                use_conv_bias=use_conv_bias,
                use_conv_ln=use_conv_ln,
                use_fc_bias=use_fc_bias,
                use_fc_ln=use_fc_ln,
                activation=activation,
                last_layer_size=last_layer_size,
                last_activation=last_activation,
                last_use_bias=last_use_bias,
                last_use_ln=last_use_ln)

        particle_dim = param_net.param_length

        init_particles = torch.randn(
            num_basins, particle_dim, requires_grad=True)  # [nb, D]
        all_particles = torch.repeat_interleave(
            init_particles, num_particles_per_basin, dim=0)  # [nb*np, D]
        particles = torch.nn.Parameter(all_particles)

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
            particles=particles,
            num_particles=num_basins * num_particles_per_basin,
            loss_type=loss_type,
            par_vi=None,
            function_vi=False,
            num_train_classes=num_train_classes,
            optimizer=optimizer,
            initial_train_steps=initial_train_steps,
            config=config,
            logging_network=logging_network,
            logging_training=logging_training,
            logging_evaluate=logging_evaluate,
            debug_summaries=debug_summaries,
            name=name)

        self._num_basins = num_basins
        self._num_particles_per_basin = num_particles_per_basin

    @property
    def num_basins(self):
        return self._num_basins

    @property
    def num_particles_per_basin(self):
        return self._num_particles_per_basin

    def predict_step(self, inputs, training=False, state=None):
        """Predict ensemble outputs for inputs using the hypernetwork model.

        Args:
            inputs (Tensor): inputs to the ensemble of networks.
            state (None): not used.

        Returns:
            AlgStep:
            - output (Tensor): predictions with shape
                ``[batch_size, n_sample, self._param_net._output_spec.shape[0]]``
            - state (None): not used
            - info (MbeInfo)
        """
        self._param_net.set_parameters(self.particles)
        outputs, _ = self._param_net(inputs)  # [bs, n_particles, d_out]
        # [bs, n_particles, d_out] or [bs, n_particles]
        outputs_mean = outputs.mean  
        total_std = outputs_mean.std(1)  # [bs, d_out] or [bs]
        outputs_mean = outputs_mean.reshape(
            outputs_mean.shape[0],
            self.num_basins, 
            self.num_particles_per_basin, 
            *outputs_mean.shape[2:])
        # [bs, n_basins, d_out] or [bs, n_basins]
        opt_std = outputs_mean.std(2)  

        return AlgStep(output=outputs,
                       state=(), 
                       info=MbeInfo(total_std=total_std, opt_std=opt_std))

    def reward_perturbation(self, info):
        reward_std = torch.std(info.reward.view(-1))
        return torch.randn(
            self.num_particles, *info.reward.shape) * reward_std

        # def _input_bootstrap_fn(input):
        #     total_batch_size = input.shape[0]
        #     assert total_batch_size % self.num_particles_per_basin == 0, (
        #         "first dim of input must be multiples of num_particles_per_basin") 
        #     batch_size = int(total_batch_size / self.num_particles_per_basin)
        #     input = input.reshape(
        #         batch_size, self.num_particles_per_basin, *input.shape[1:])
        #     return input.repeat(1, self.num_basins, *(1,)*(input.ndim - 2))
        #     
        # return alf.nest.map_structure(_input_bootstrap_fn, inputs) 
