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
"""Supervised learning utilities."""

import alf
import absl
import torch
import numpy as np
import torch.nn.functional as F
from alf.data_structures import LossInfo
try:
    from sklearn.metrics import roc_auc_score
except:
    pass


def classification_loss(output, target):
    """Computes the cross entropy loss with respect to a batch of predictions and
    targets.
    
    Args:
        output (Tensor): predictions of shape ``[B, D]`` or ``[B, N, D]``.
        target (Tensor): targets of shape ``[B]``, ``[B, 1]``, ``[B, N]``,
            or ``[B, N, 1]``.

    Returns:
        LossInfo containing the computed cross entropy loss and the average
            accuracy.
    """

    if output.ndim == 2:
        output = output.reshape(output.shape[0], target.shape[1], -1)
    pred = output.max(-1)[1]
    target = target.squeeze(-1)
    acc = pred.eq(target).float().mean(0)
    avg_acc = acc.mean()
    if output.ndim == 3:
        output = output.transpose(1, 2)
    else:
        output = output.reshape(output.shape[0] * target.shape[1], -1)
        target = target.reshape(-1)
    loss = F.cross_entropy(output, target, reduction='sum')
    return LossInfo(loss=loss, extra=avg_acc)


def regression_loss(output, target):
    """Computes the MSE loss with respect to a batch of predictions and
    targets.
    
    Args:
        output (Tensor): predictions of shape ``[B, 1]`` or ``[B, N, 1]`` 
        target (Tensor): targets of shape ``[B, 1]`` or ``[B, N, 1]``

    Returns:
        LossInfo containing the computed MSE loss
    """

    out_shape = output.shape[-1]
    assert (target.shape[-1] == out_shape), (
        "feature dimension of output and target does not match.")
    loss = 0.5 * F.mse_loss(
        output.reshape(-1, out_shape),
        target.reshape(-1, out_shape),
        reduction='sum')
    return LossInfo(loss=loss, extra=())


def auc_score(inliers, outliers):
    """Computes the AUROC score w.r.t network outputs on two distinct datasets.
    Typically, one dataset is the main training/testing set, while the
    second dataset represents a set of unseen outliers.
    
    Args: 
        inliers (torch.tensor): set of predictions on inlier data
        outliers (torch.tensor): set of predictions on outlier data
    
    Returns:
        AUROC score (float)
    """
    inliers = inliers.detach().cpu().numpy()
    outliers = outliers.detach().cpu().numpy()
    y_true = np.array([0] * len(inliers) + [1] * len(outliers))
    y_score = np.concatenate([inliers, outliers])
    try:
        auc_score = roc_auc_score(y_true, y_score)
    except NameError:
        absl.logging.info('roc_auc_score function not defined')
        auc_score = 0.5
    return auc_score


def predict_dataset(model, testset):
    """Computes predictions for an input dataset. 
    
    Args: 
        model (Callable): model with which to compute predictions.
        testset (torch.utils.data.DataLoader): dataset for which to compute
            predictions.

    Returns:
        model_outputs (torch.tensor): a tensor of shape [N, S, D] where
            N refers to the number of predictors, S is the number of data
            points, and D is the output dimensionality. 
    """
    if hasattr(testset.dataset, 'dataset'):
        cls = len(testset.dataset.dataset.classes)
    else:
        cls = len(testset.dataset.classes)
    outputs = []
    for batch, (data, target) in enumerate(testset):
        data = data.to(alf.get_default_device())
        output, _ = model(data)
        if output.dim() == 2:
            output = output.unsqueeze(1)
        output = output.transpose(0, 1)
        outputs.append(output)
    model_outputs = torch.cat(outputs, dim=1)  # [N, B, D]
    return model_outputs
