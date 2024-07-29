# Copyright (c) 2023 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from absl import app
from absl import flags
from absl import logging
from functools import partial
import math
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Tuple

import alf
from alf.utils import common
from alf.networks import s5
from alf.networks.s5.s5_test_data import create_mnist_classification_dataset
from alf.utils.schedulers import Scheduler, LinearScheduler, update_progress


def prep_batch(batch: tuple, seq_len: int,
               in_dim: int) -> Tuple[np.ndarray, np.ndarray, np.array]:
    """
    Take a batch and convert it to a standard x/y format.
    :param batch:       (x, y, aux_data) as returned from dataloader.
    :param seq_len:     (int) length of sequence.
    :param in_dim:      (int) dimension of input.
    :return:
    """
    batch = alf.nest.utils.convert_device(batch)
    if len(batch) == 2:
        inputs, targets = batch
        aux_data = {}
    elif len(batch) == 3:
        inputs, targets, aux_data = batch
    else:
        raise RuntimeError(
            "Err... not sure what I should do... Unhandled data type. ")

    # Grab lengths from aux if it is there.
    lengths = aux_data.get('lengths', None)

    # Make all batches have same sequence length
    num_pad = seq_len - inputs.shape[1]
    if num_pad > 0:
        # Assuming vocab padding value is zero
        inputs = F.pad(inputs, ((0, 0), (0, num_pad)))

    # Inputs is either [n_batch, seq_len] or [n_batch, seq_len, in_dim].
    # If there are not three dimensions and trailing dimension is not equal to in_dim then
    # transform into one-hot.  This should be a fairly reliable fix.
    if (inputs.ndim < 3) and (inputs.shape[-1] != in_dim):
        inputs = F.one_hot(inputs, in_dim)

    if lengths is not None:
        full_inputs = (inputs, lengths)
    else:
        full_inputs = inputs

    # If there is an aux channel containing the integration times, then add that.
    if 'timesteps' in aux_data.keys():
        integration_timesteps = torch.diff(aux_data['timesteps'])
    else:
        integration_timesteps = torch.ones((len(inputs), seq_len))

    return full_inputs, targets, integration_timesteps


def eval_step(
        batch_inputs,
        batch_labels,
        model,
):
    state = common.zero_tensor_from_nested_spec(model.state_spec,
                                                batch_inputs.shape[0])
    logits = model(batch_inputs, state)[0]
    losses = torch.nn.functional.cross_entropy(logits, batch_labels)
    accs = logits.argmax(dim=-1) == batch_labels

    return losses, accs, logits


def train_step(batch_inputs, batch_labels, model, optimizer):
    """Performs a single training step given a batch of data"""
    state = common.zero_tensor_from_nested_spec(model.state_spec,
                                                batch_inputs.shape[0])
    logits = model(batch_inputs, state)[0]
    loss = torch.nn.functional.cross_entropy(logits, batch_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_epoch(model, trainloader, seq_len, in_dim, optimizer, iteration):
    """
    Training function for an epoch that loops over batches.
    """
    model.train()
    batch_losses = []

    for batch_idx, batch in enumerate(tqdm(trainloader)):
        iteration += 1
        update_progress("iterations", iteration)
        inputs, labels, integration_times = prep_batch(batch, seq_len, in_dim)
        loss = train_step(inputs, labels, model, optimizer)
        batch_losses.append(loss)

    # Return average loss over batches
    return np.mean(np.array(batch_losses)), iteration


def validate(model, testloader, seq_len, in_dim):
    """Validation function that loops over batches"""
    model.eval()
    losses, accuracies = [], []
    for batch_idx, batch in enumerate(tqdm(testloader)):
        inputs, labels, integration_timesteps = prep_batch(
            batch, seq_len, in_dim)
        loss, acc, pred = eval_step(inputs, labels, model)
        losses.append(loss.mean().item())
        accuracies.append(acc.to(torch.float32).mean().item())

    aveloss, aveaccu = np.mean(losses), np.mean(accuracies)
    return aveloss, aveaccu


def create_optimizer(model, args, steps_per_epoch):
    other_params = []
    ssm_params = []
    for name, param in model.named_parameters():
        if any(s in name for s in
               ["_ssm._B", "_ssm._Lambda", "._norm.", "._ssm._log_step"]):
            logging.info(
                f"ssm param {name}: {param.shape}, magnitude: {param.abs().mean()}"
            )
            ssm_params.append(param)
        else:
            logging.info(
                f"regular param {name}: {param.shape}, magnitude: {param.abs().mean()}"
            )
            other_params.append(param)

    if args.cosine_anneal:
        lr_scheduler = LinearWarmupCosineScheduler(
            progress_type='iterations',
            warmup_end_step=steps_per_epoch * args.warmup_end,
            end_step=steps_per_epoch * args.epochs,
            base_lr=args.lr,
            final_lr=args.lr_min)
        ssm_lr_scheduler = LinearWarmupCosineScheduler(
            progress_type='iterations',
            warmup_end_step=steps_per_epoch * args.warmup_end,
            end_step=steps_per_epoch * args.epochs,
            base_lr=args.ssm_lr,
            final_lr=args.lr_min)
    else:
        lr_scheduler = LinearScheduler(
            progress_type='iterations',
            schedule=[(0, args.lr), (steps_per_epoch * args.warmup_end,
                                     args.lr)],
        )
        ssm_lr_scheduler = LinearScheduler(
            progress_type='iterations',
            schedule=[(0, 0.0), (steps_per_epoch * args.warmup_end,
                                 args.ssm_lr)],
        )

    optimizer = alf.optimizers.AdamW(
        lr=lr_scheduler, weight_decay=args.weight_decay)
    optimizer.add_param_group({'params': other_params})
    optimizer.add_param_group({
        'params': ssm_params,
        'lr': ssm_lr_scheduler,
        'weight_decay': 0.0
    })

    return optimizer


class LinearWarmupCosineScheduler(Scheduler):
    """

    Linearly increase learning rate from 0 to base_lr until `warmup_end_step`.
    Then decrease learning rate from base_lr to final_lr using cosine annealing
    until `end_step`.
    """

    def __init__(self, progress_type, warmup_end_step, end_step, base_lr,
                 final_lr):
        super().__init__(progress_type)
        self._warmup_end_step = warmup_end_step
        self._end_step = end_step
        self._base_lr = base_lr
        self._final_lr = final_lr

    def __call__(self):
        progress = self.progress()
        if progress < self._warmup_end_step:
            return self._base_lr * progress / self._warmup_end_step
        else:
            progress = min(progress, self._end_step)
            return self._final_lr + 0.5 * (self._base_lr - self._final_lr) * (
                1 + math.cos(math.pi * (progress - self._warmup_end_step) /
                             (self._end_step - self._warmup_end_step)))


def train(args):
    trainloader, valloader, testloader, aux_dataloaders, n_classes, seq_len, in_dim, train_size = \
    create_mnist_classification_dataset(args.dir_name, seed=args.seed, bsz=args.bsz)

    ssm_ctor = partial(
        s5.S5SSM,
        data_dim=args.d_model,
        state_dim=args.state_dim,
        num_blocks=args.num_blocks,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        step_rescale=args.step_rescale)

    model = alf.networks.Sequential(
        lambda x: x.transpose(0, 1),
        s5.create_stacked_s5_encoder(
            in_dim,
            ssm_ctor,
            num_layers=args.num_layers,
            activation=args.activation_fn,
            dropout=args.dropout,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum),
        lambda x: x.mean(dim=0),
        alf.layers.FC(args.d_model, n_classes),
        input_tensor_spec=alf.TensorSpec((
            seq_len,
            in_dim,
        )),
    )

    steps_per_epoch = int(train_size / args.bsz)
    optimizer = create_optimizer(model, args, steps_per_epoch)
    iteration = 0

    for epoch in range(args.epochs):
        logging.info(f"[*] Starting Training Epoch {epoch + 1}...")

        train_loss, iteration = train_epoch(model, trainloader, seq_len,
                                            in_dim, optimizer, iteration)

        logging.info(f"[*] Running Epoch {epoch + 1} Validation...")
        val_loss, val_acc = validate(model, valloader, seq_len, in_dim)

        logging.info(f"[*] Running Epoch {epoch + 1} Test...")
        test_loss, test_acc = validate(model, testloader, seq_len, in_dim)

        logging.info(f"\n=>> Epoch {epoch + 1} Metrics ===")
        logging.info(
            f"\tTrain Loss: {train_loss:.5f} -- Val Loss: {val_loss:.5f} --Test Loss: {test_loss:.5f} --"
            f" Val Accuracy: {val_acc:.4f}"
            f" Test Accuracy: {test_acc:.4f}")


class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def main(_):
    args = Args(
        dir_name=Path('~/data/mnist').expanduser(),
        seed=12345,
        bsz=50,
        d_model=96,
        state_dim=128,
        num_blocks=1,
        dt_min=0.001,
        dt_max=0.1,
        step_rescale=1.0,
        num_layers=4,
        activation_fn="half_glu2",
        dropout=0.1,
        prenorm=True,
        batchnorm=True,
        bn_momentum=0.05,
        epochs=150,
        cosine_anneal=True,
        warmup_end=0,
        lr=4e-3,
        ssm_lr=1e-3,
        lr_min=0.0,
        weight_decay=0.01,
    )
    train(args)


if __name__ == '__main__':
    if torch.cuda.is_available():
        alf.set_default_device("cuda")
    logging.set_verbosity(logging.INFO)
    app.run(main)
