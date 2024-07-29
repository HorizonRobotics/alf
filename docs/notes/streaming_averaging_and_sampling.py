# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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

#%% [markdown]
"""
# Averaging and sampling from an infinite stream of samples

## Wei Xu

Often we need to calculate the average or randomly select samples from an infinite
stream of samples. Suppose the $t$-th sample is $x_t$, a standard way of
calculating the average is:
$$
\begin{equation*}
\bar{x}_t = \left(1 - \frac{1}{t}\right)\bar{x}_{t-1} + \frac{1}{t} x_t
\end{equation*}
$$

If we want to randomly select $k$ samples, the standard way of doing this is the
so called [reservoir sampling](https://en.wikipedia.org/wiki/Reservoir_sampling).
What it does is to keep the $t$-th sample with probability $k/t$ at step $t$ and
randomly replacing one of the existing $k$ items.

Sometime, we want the average to give higher weight to more recent samples and
select recent samples with higher probability. A simple modification to the
moving average and reservoir sampling can achieve this.

## Streaming average with higher weight for recent samples

We only need to change the update rate of the moving average as the following:
$$
\begin{equation}
\bar{x}_t = \left(1 - \frac{s}{t}\right)\bar{x}_{t-1} + \frac{s}{t} x_t
\end{equation}
$$

Now we analyze how the above scheme assigns weight to different samples. Let
$\bar{x}_t = \sum_{i=1}^t w_{t,i} x_i$. Now the problem is to find out the values
of $w_{t,i}$.  We can see that $w_{t,t} = \frac{s}{t}$ and
$w_{t,i} = (1 - \frac{s}{t})w_{t-1,i}$ for $t>i$. So we have
$$
\begin{array}{ll}
w_{t,i} &=& \frac{s}{i} \prod_{j=i+1}^t (1 - \frac{s}{j}) \\
    &=& \frac{s}{i} \frac{i+1-s}{i+1} \cdots \frac{t-1-s}{t-1}\frac{t-s}{t} \\
    &=& s \frac{(i-s+1)(i-s+2) \cdots (i-1)}{(t-s+1)(t-s+2)\cdots t} \\
    &\approx& s \left(\frac{i}{t}\right)^{s-1}
\end{array}
$$
This means that the weight for the $i$-th sample is approximately proportional
to $i^{s-1}$.

## Streaming sampling with higher probability for recent samples

The modification to the reservoir sampling is to keep the $t$-th sample with
probability $\frac{sk}{t}$ instead of $\frac{k}{t}$. Let $p_{t,i}$ be the probability of
the $i$-th sample still being in the reservoir at step $t$. It is clear that for
$t\ge sk$, $p_{t,t} = \frac{sk}{t}$. The probability of the $i$-th item being in
the reservoir at step $t$ is the proability that it is in reservoir
at step $t-1$ multiplied with the probability of it not being replaced at step $t$:
$p_{t,i} = p_{t-1,i} \left(1 - \frac{sk}{t} \frac{1}{k}\right) = p_{t-1,i} \left(1-\frac{s}{t}\right)$.
With this relationship, we get:
$$
\begin{array}{ll}
p_{t,i} &=& p_{i,i} \prod_{j=i+1}^t \left(1-\frac{s}{j}\right) \\
    &=& \frac{sk}{i} \frac{i+1-s}{i+1} \cdots \frac{t-1-s}{t-1}\frac{t-s}{t} \\
    &=& sk \frac{(i-s+1)(i-s+2) \cdots (i-1)}{(t-s+1)(t-s+2)\cdots t} \\
    &\approx& sk \left(\frac{i}{t}\right)^{s-1}
\end{array}
$$

So we can see that $p_{t,i}$ is approximately proportional to $i^{s-1}$.
"""

# %%
