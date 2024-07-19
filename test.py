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

from alf.utils.dist_utils import params_to_distributions, distributions_to_params, DiagMultivariateNormal, StableTanh, AffineTransform, _get_transformed_builder, DistributionSpec, extract_distribution_parameters, PowerTransform, get_base_dist
from alf.utils.dist_utils import *
from torch.distributions import TransformedDistribution, Normal
import torch
import alf

tf = [
    # StableTanh(),
    # AffineTransform(loc=torch.arange(1, 5, dtype=torch.float),
    #                 scale=torch.arange(1, 5, dtype=torch.float)),
    # Softupper(high=torch.arange(1, 5, dtype=torch.float)),
    SoftclipTF(low=torch.arange(1, 5, dtype=torch.float), high=10),
    Softclip(low=5, high=10)
    # AffineTransform(loc=0., scale=2.)
    # AffineTransform(loc=torch.tensor(0.), scale=torch.tensor(2.))
]

# print(torch.arange(1, 4))
# exit(0)
dist = DiagMultivariateNormal(
    loc=torch.arange(1, 5, dtype=torch.float),
    scale=torch.arange(1, 5, dtype=torch.float))
# dist = DiagMultivariateNormal(loc=torch.ones((4, )), scale=torch.ones((4, )))

# print(issubclass(type(dist), Normal))
# exit(0)

tf_dist = TransformedDistribution(dist, tf)

params = distributions_to_params(dist)
params2 = distributions_to_params(tf_dist)
builder, tf_params = _get_transformed_builder(tf_dist)
# d = builder(**tf_params)

print(params)
print(params2)
print(tf_params)

split_at = 2


def split_left(x):
    if x.ndim == 0:
        return x
    return x[..., :split_at]


def split_right(x):
    if x.ndim == 0:
        return x
    return x[..., split_at:]


left_params = alf.nest.map_structure(split_left, tf_params)
right_params = alf.nest.map_structure(split_right, tf_params)
left = builder(**left_params)
right = builder(**right_params)

# print(isinstance(dist, torch.distributions.Independent))

x = DistributionSpec.from_distribution(tf_dist)
# print(x)

# print(tf[1].__dict__)
# print(dist.params

# print(dist)
# print(dist2)
# # print(dist3)
# print(tf_dist)
