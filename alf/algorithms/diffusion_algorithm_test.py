# Copyright (c) 2025 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from alf.algorithms.diffusion_algorithm import *
from alf.algorithms.diffusion_model import SimpleMLPNet
from functools import partial

import math
import torch
from functools import partial
import matplotlib.pyplot as plt

torch.set_default_device('cuda')


def jointplot(x, y, bins=30, figure_size=(8, 8)):
    """Draw scatter plot with marginal histograms aligned to the axes
    """
    # scatter + marginal histograms
    fig = plt.figure(figsize=figure_size)
    gs = fig.add_gridspec(4, 4, wspace=0.05, hspace=0.05)

    # Main scatter plot
    ax_scatter = fig.add_subplot(gs[1:4, 0:3])
    ax_scatter.plot(x, y, '.', alpha=0.5)

    # X histogram (above scatter, share x-axis)
    ax_histx = fig.add_subplot(gs[0, 0:3], sharex=ax_scatter)
    ax_histx.hist(x, bins=bins, color="gray")
    plt.setp(ax_histx.get_xticklabels(), visible=False)  # hide x labels here

    # Y histogram (to the right of scatter, share y-axis)
    ax_histy = fig.add_subplot(gs[1:4, 3], sharey=ax_scatter)
    ax_histy.hist(y, bins=bins, orientation='horizontal', color="gray")
    plt.setp(ax_histy.get_yticklabels(), visible=False)  # hide y labels here

    ax_histx.set_ylabel("count")
    ax_histy.set_xlabel("count")

    plt.show()


def pdist(A, B):
    """Distance between each pair of the two collections of inputs.

    Args:
        A: (b, n, d)
        B: (b, m, d)
    Returns:
        pairwise distances (b,n,m)
    """
    A2 = (A * A).sum(dim=2, keepdim=True)  # (b,n,1)
    B2 = (B * B).sum(dim=2, keepdim=True)  # (b,m,1)
    # bmm: (B,n,d) x (B,d,m) -> (B,n,m)
    M = torch.bmm(A, B.transpose(1, 2))
    D2 = A2 + B2.transpose(1, 2) - 2.0 * M
    return D2.clamp_min_(0.0).sqrt_()


def energy_stat(sample_x, sample_y, size):
    # https://en.wikipedia.org/wiki/Energy_distance#Testing_for_equal_distributions
    # pairwise Euclidean norms
    def _pdist(A, B):
        # return (((A[:,None,:]-B[None,:,:])**2).sum(-1)).sqrt()
        return pdist(A.unsqueeze(0), B.unsqueeze(0)).squeeze(0)

    X = sample_x(size)
    Y = sample_y(size)
    d_xy = _pdist(X, Y).mean()
    d_xx = _pdist(X, X).mean()
    d_yy = _pdist(Y, Y).mean()
    return (2 * d_xy - d_xx - d_yy) / d_yy


class GMM:
    name = 'gmm'
    mu1 = torch.tensor([-0.5, -0.5])
    std1 = 0.25
    mu2 = torch.tensor([0.5, 0.5])
    std2 = 0.125
    prob1 = 0.3

    def neg_energy(self, x, _):
        logp1 = math.log(self.prob1) - 2 * math.log(self.std1) - 0.5 * ((
            (x - self.mu1) / self.std1)**2).sum(-1)
        logp2 = math.log(1 - self.prob1) - 2 * math.log(self.std2) - 0.5 * ((
            (x - self.mu2) / self.std2)**2).sum(-1)
        logp = torch.stack([logp1, logp2], dim=-1)
        return logp.logsumexp(dim=-1)

    def sample(self, n):
        r = torch.rand(n)
        e = torch.randn(n, 2)
        mu = torch.where(r.unsqueeze(-1) < self.prob1, self.mu1, self.mu2)
        std = torch.where(r < self.prob1, self.std1, self.std2)
        return mu + e * std.unsqueeze(-1)


def sample_f(n, generator):
    inputs = torch.full((n, 1), 1.0)
    return generator.sample(inputs)


def train(generator, dist, sample_based, batch_size=1024, ema=0.99):
    if ema > 0:
        averager = torch.optim.swa_utils.AveragedModel(
            generator,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema))
        ema_generator = averager.module
    else:
        ema_generator = generator

    optimizer = torch.optim.Adam(generator.parameters(), lr=6e-4)
    warmup_iters = 16
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.001,
        end_factor=1.0,
        total_iters=warmup_iters,
    )
    main_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer,
                                                         total_iters=1000,
                                                         factor=1.0)
    lr_schedule = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_iters])
    for i in range(1000):
        samples = dist.sample(batch_size) if sample_based else None
        f_neg_energy = dist.neg_energy if not sample_based else None
        loss = generator.calc_loss(torch.full((batch_size, 1), 1.0), samples,
                                   f_neg_energy).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_schedule.step()
        if ema > 0:
            averager.update_parameters(generator)

    e_stat = energy_stat(partial(sample_f, generator=ema_generator),
                         dist.sample, 10000)
    return ema_generator, e_stat


def run_setting(setting):
    generator = setting['generator'](input_spec=alf.TensorSpec((1, )),
                                     output_spec=alf.TensorSpec((2, )),
                                     model_ctor=SimpleMLPNet,
                                     sde=setting['sde'],
                                     steps=setting.get('steps', 20))
    name = setting['name']
    sample_based = setting['sample_based']
    ema = setting.get('ema', 0.0)
    print('Running', name)
    e_stats = []
    repeat = 5

    while len(e_stats) < repeat:
        dist = GMM()
        model, e_stat = train(generator,
                              dist,
                              sample_based,
                              batch_size=1024,
                              ema=ema)
        print('train', len(e_stats), dist.name, 'energy_stat', e_stat.item())
        if e_stat.isfinite():
            e_stats.append(e_stat.item())

    e_stats = torch.tensor(e_stats)
    print('energy stat mean:',
          e_stats.mean().item(), "std:",
          e_stats.std().item())

    x = sample_f(2000, generator).cpu().numpy()
    jointplot(x[:, 0], x[:, 1], bins=50)
    plt.savefig(f'{setting["name"]}.png')
    return model, e_stats.mean().item(), e_stats.std().item()


settings = [
    dict(name='ot_sample_fm_ema',
         sde=OTSDE(),
         generator=FlowMatching,
         sample_based=True,
         ema=0.99),
    dict(name='rt_sample_fm_ema',
         sde=RTSDE(),
         generator=FlowMatching,
         sample_based=True,
         ema=0.99),
    dict(name='ot_fm_ema',
         sde=OTSDE(),
         generator=FlowMatching,
         sample_based=False,
         ema=0.99),
    dict(name='rt_fm_ema',
         sde=RTSDE(),
         generator=FlowMatching,
         sample_based=False,
         ema=0.99),
]

if __name__ == '__main__':
    results = []
    for setting in settings:
        results.append(run_setting(setting))
    for setting, result in zip(settings, results):
        model, e_stat_mean, e_stat_std = result
        print(setting['name'], e_stat_mean, e_stat_std)
