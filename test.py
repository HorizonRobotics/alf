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
from torch.distributions import TransformedDistribution, Normal
import torch

tf = [
    StableTanh(),
    AffineTransform(loc=torch.zeros((4, )), scale=4 * torch.ones((4, )))
]

dist = DiagMultivariateNormal(loc=torch.zeros((4, )), scale=torch.ones((4, )))

print(get_base_dist(dist))

# print(issubclass(type(dist), Normal))
# exit(0)

tf_dist = TransformedDistribution(dist, tf)
print(get_base_dist(tf_dist))

params = distributions_to_params(dist)
params2 = distributions_to_params(tf_dist)
builder, tf_params = _get_transformed_builder(tf_dist)
d = builder(**tf_params)
# dist2 = type(dist)(**params)
# dist3 = params_to_distributions(params, DiagMultivariateNormal)
print(params)
print(params2)

# print(tf[1].__dict__)
# print(dist.params

# print(dist)
# print(dist2)
# # print(dist3)
# print(tf_dist)
