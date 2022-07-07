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
"""Subspace classes for MultiSwag algorithm. Adapted from
   github.com/izmailovpavel/understandingbdl/blob/master/swag/posteriors/subspaces.py
"""

import abc
import numpy as np
import torch


class Subspace(torch.nn.Module, metaclass=abc.ABCMeta):
    subclasses = {}

    @classmethod
    def register_subclass(cls, subspace_type):
        def decorator(subclass):
            cls.subclasses[subspace_type] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, subspace_type, **kwargs):
        if subspace_type not in cls.subclasses:
            raise ValueError('Bad subspaces type {}'.format(subspace_type))
        return cls.subclasses[subspace_type](**kwargs)

    def __init__(self):
        super(Subspace, self).__init__()

    @abc.abstractmethod
    def update(self, param_vec):
        pass

    @property
    @abc.abstractmethod
    def cov_factor(self):
        pass


@Subspace.register_subclass('random')
class RandomSpace(Subspace):
    def __init__(self, num_parameters, rank=20, method='dense'):
        assert method in ['dense', 'fastfood']

        super(RandomSpace, self).__init__()

        self.num_parameters = num_parameters
        self.rank = rank
        self.method = method

        if method == 'dense':
            self.subspace = torch.randn(rank, num_parameters)

        if method == 'fastfood':
            raise NotImplementedError(
                "FastFood transform hasn't been implemented yet")

    # random subspace is independent of data
    def update(self, param_vec):
        pass

    @property
    def cov_factor(self):
        return self.subspace


@Subspace.register_subclass('covariance')
class CovarianceSpace(Subspace):
    def __init__(self, num_parameters, var_clamp=1e-6, max_rank=20):
        super(CovarianceSpace, self).__init__()

        self._num_parameters = num_parameters
        self._var_clamp = var_clamp
        self._max_rank = max_rank

        # self.register_buffer('mean', torch.zeros(num_parameters))
        # self.register_buffer('sq_mean', torch.zeros(num_parameters))
        # self.register_buffer('n_samples', torch.zeros(1, dtype=torch.long))
        self.register_buffer('rank', torch.zeros(1, dtype=torch.long))
        self.register_buffer(
            'samples', torch.empty(0, num_parameters, dtype=torch.float32))
        # self.register_buffer(
        #     'cov_mat_sqrt', torch.empty(
        #         0, num_parameters, dtype=torch.float32))
        self.update(torch.zeros(num_parameters))

    @property
    def mean(self):
        return self.samples.mean(dim=0)

    @property
    def variance(self):
        return torch.clamp(self.samples.var(dim=0), self._var_clamp)
        # return torch.clamp(self.sq_mean - self.mean**2, self._var_clamp)

    @property
    def cov_factor(self):
        return (self.samples - self.mean) / (self.samples.size(0) - 1)**0.5
        # return self.cov_mat_sqrt.clone() / (self.cov_mat_sqrt.size(0) - 1)**0.5

    def get_space(self):
        return self.mean, self.variance, self.cov_factor

    def sample(self, n_sample, scale=0.5, diag_noise=True):
        if n_sample == 1:
            return self.mean.view(1, -1)  # [1, n_params]
        elif n_sample > 1:
            eps_low_rank = torch.randn(
                n_sample, self.cov_factor.shape[0])  # [n_sample, n_rank]
            z = eps_low_rank @ self.cov_factor  # [n_sample, n_params]
            if diag_noise:
                noise = torch.randn(n_sample, *self.variance.shape) * \
                    self.variance.sqrt().view(1, -1)  # [n_sample, n_params]
                z += noise
            z *= scale**0.5
            return self.mean.view(1, -1) + z  # [n_sample, n_params]
        else:
            raise ValueError("n_sample must be a positive integer")

    def update(self, param_vec):
        if self.rank > 0:
            dev_vec = param_vec - self.mean
        else:
            dev_vec = torch.zeros(self._num_parameters)
        if self.rank.item() + 1 > self._max_rank:
            self.samples = self.samples[1:, :]
            # self.cov_mat_sqrt = self.cov_mat_sqrt[1:, :]

        self.samples = torch.cat((self.samples, param_vec.view(1, -1)), dim=0)
        # self.cov_mat_sqrt = torch.cat(
        #     (self.cov_mat_sqrt, dev_vec.view(1, -1)), dim=0)
        self.rank = torch.min(self.rank + 1,
                              torch.as_tensor(self._max_rank)).view(-1)

    # def update(self, param_vec):
    #     # first moments
    #     self.mean.mul_(self.n_samples.item() / (self.n_samples.item() + 1.))
    #     self.mean.add_(param_vec / (self.n_samples.item() + 1.))

    #     # second moments
    #     self.sq_mean.mul_(self.n_samples.item() / (self.n_samples.item() + 1.))
    #     self.sq_mean.add_(param_vec**2 / (self.n_samples.item() + 1.))

    #     # cov matrix
    #     dev_vec = param_vec - self.mean
    #     if self.rank.item() + 1 > self._max_rank:
    #         self.cov_mat_sqrt = self.cov_mat_sqrt[1:, :]
    #     self.cov_mat_sqrt = torch.cat((self.cov_mat_sqrt, dev_vec.view(1, -1)),
    #                                   dim=0)
    #     self.rank = torch.min(self.rank + 1,
    #                           torch.as_tensor(self._max_rank)).view(-1)

    #     self.n_samples.add_(1)
