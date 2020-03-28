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
# Stable gradient descent for Gaussian distribution

## Wei Xu

Gaussian distribution has two parameters $\mu$ and $\sigma$. Given a sample $x$
from the distribution, the negative log probability is
$$
\begin{equation*}
l = \frac{1}{2}\log 2\pi + \log \sigma + \frac{1}{2}\frac{(\mu-x)^2}{\sigma^2}
\end{equation*}
$$
Its derivatives are:
$$
\begin{array}{ll}
&& \frac{\partial l}{\partial \mu} = \frac{\mu-x}{\sigma^2} \\
&& \frac{\partial l}{\partial \sigma}
    = \frac{1}{\sigma}-\frac{(\mu-x)^2}{\sigma^3} \\
&& \frac{\partial^2l}{\partial \mu^2} = \frac{1}{\sigma^2} \\
&& \frac{\partial^2l}{\partial \sigma^2}
    = -\frac{1}{\sigma^2} + 3\frac{(\mu-x)^2}{\sigma^4}
\end{array}
$$
From these equations we can see that $\frac{\partial l}{\partial \mu}$ can
become very big if $\sigma$ become very small. This cannot be fully compensated
by optimizers with adaptive learning rate such as ADAM [1](#ADAM). For ADAM,
the learning rate is the reciprocal of $v_t$. For $\mu$ in our problem, the
learning rate is:
$$
\begin{equation*}
\frac{1}{v_t(\mu)} = \frac{1}{\sqrt{O\left(\frac{(\mu-x)^2}{\sigma^4}\right)}}
 = \frac{1}{\sqrt{O\left(\frac{1}{\sigma^2}\right)}}=O(\sigma)
\end{equation*}
$$
where we use the fact the $x$ is sampled from $N(\mu, \sigma)$ and hence
$\mu-x=O(\sigma)$. Optimization theory tells us that gradient descent will
diverge if the learning rate is larger than
$2/\frac{\partial^2l}{\partial\mu^2}=O(\sigma^2)$
(See for example, section 5.1 in [2](#EfficientBackprop). From this, we can
see that the learning rate of ADAM is too high when $\sigma$ is small and
leads to diverging behavior, which is exactly what we observed in our experiments.

Our strategy is to parameterize Gaussian in a different way so that its second
derivatives are bounded when $\sigma$ is small. Let $\mu=\alpha \sigma$,
$\sigma=\frac{1}{\log(1+\exp(\beta))}\equiv\frac{1}{\gamma}$. Now we have
$$
\begin{equation*}
l = \frac{1}{2}\log 2\pi - \log \gamma + \frac{1}{2}(\alpha-x \gamma)^2
\end{equation*}
$$
The derivatives are:
$$
\begin{array}{ll}
&& \frac{\partial l}{\partial \alpha} = \alpha-x \gamma \\
&& \frac{\partial l}{\partial \beta}
   = \left(-\frac{1}{\gamma} + x(x\gamma-\alpha)\right)\frac{d\gamma}{d\beta} \\
&& \frac{\partial^2l}{\partial \alpha^2} = 1  \\
&& \frac{\partial ^2l}{\partial \beta^2}
   = \left(\frac{1}{\gamma^2} + x^2\right)\left(\frac{d\gamma}{d\beta}\right)^2
   +\left(-\frac{1}{\gamma} + x(x\gamma-\alpha)\right) \frac{d^2\gamma}{d\beta^2} \\
&& \frac{d\gamma}{d\beta} = \frac{\exp(\beta)}{1+\exp(\beta)} \\
&& \frac{d^2\gamma}{d\beta^2} = \frac{\exp(\beta)}{(1+\exp(\beta))^2} \\
\end{array}
$$

First, we see that $\frac{\partial l}{\partial \alpha}$ is bounded  because
$x\gamma-\alpha=O(1)$ (since x is sampled from the same distribution). Second,
because $\frac{1}{\gamma}\frac{\partial\gamma}{\partial\beta}$ is bounded as
shown in the following:
$$
\begin{array}{ll}
\frac{1}{\gamma}\frac{\partial\gamma}{\partial\beta}
 &=& \frac{1}{\gamma}\frac{\exp(\beta)}{1+\exp(\beta)}
 = \frac{\exp(\gamma)-1}{\gamma \exp(\gamma)}
 = \frac{1-\exp(-\gamma)}{\gamma} \le 1
\end{array}
$$
$\frac{\partial l}{\partial \beta}$ is also bounded.
Third, $\frac{\partial^2l}{\partial \alpha^2}$ is a constant, which is nice. And
finally, for $\frac{\partial^2l}{\partial \beta^2}$, we have:
$$
\begin{equation*}
\frac{\partial^2l}{\partial\beta^2}
\le \left(\left(\frac{1}{\gamma}\frac{\partial\gamma}{\partial\beta}\right)^2
     +x^2\right) + |x(x\gamma-\alpha)|
=O(x^2+|x|)
\end{equation*}
$$
hence it is also bounded.

# References
<a name="ADAM"></a> [1] D.P. Kingma and J. L. Ba _ADAM: A Method for Stochastic
 Optimization_ ICLR 2015, arXiv:1412.6980

<a name="EfficientBackprop"></a> [2] Y. LeCun, L. Bottou, G. Orr and K. Muller
 _Efficient Backprop_ in Orr, G. and Muller K. (Eds), Neural Networks: Tricks of
  the trade, Springer, 1998
"""
#%%
