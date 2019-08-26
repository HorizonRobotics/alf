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
# Subtleties of estimating entropy

## Wei Xu

For some algorithms, we need to calculate the entropy and its derivative. If
there is no analytic formula for the entropy, we can resort to sampling. Given
the definition of entropy:

$$
\begin{equation*}
H(p) = E_{x\sim p_\theta}(-\log p_\theta(x))
\end{equation*}
$$

We can see that $-\log p_{\theta}(x)$ is an unbiased estimator of $H$ if $x$ is
sampled from $p_{\theta}$. It is tempting to use
$-\frac{\partial\log p_\theta(x)}{\partial\theta}$ as an estimator of
$\frac{\partial H}{\partial\theta}$. However, it is wrong, as shown in the
following:

$$
\begin{equation*}
E_{x\sim p_\theta}\left(\frac{\partial\log p_\theta(x)}{\partial\theta}\right)
 = \int \frac{\partial\log p_\theta(x)}{\partial\theta} p_\theta(x) dx
 = \int \frac{\partial p_\theta(x)}{\partial\theta} dx
 = \frac{\partial}{\partial\theta} \int p_\theta(x) dx
 = \frac{\partial 1}{\partial\theta} = 0
\end{equation*}
$$

We need to actually go through the process of calculating the derivative to get
the unbiased estimator of $\frac{\partial H}{\partial\theta}$:

$$
\begin{array}{ll}
\frac{\partial H}{\partial\theta}
&=&-\frac{\partial}{\partial\theta}\int \log p_\theta(x) p_\theta(x) dx \\
&=& - \int \left(\frac{\partial\log p_\theta(x)}{\partial\theta}p_\theta(x)
    + \log p_\theta(x) \frac{\partial p_\theta(x)}{\partial\theta}\right) dx \\
&=& - \int \left(\frac{\partial\log p_\theta(x)}{\partial\theta}p_\theta(x)
     + \log p_\theta(x) \frac{\partial\log p_\theta(x)}{\partial\theta} p_\theta(x)\right) dx \\
&=& - \int (1+\log p_\theta(x))\frac{\partial\log p_\theta(x)}{\partial\theta} p_\theta(x) dx \\
&=& -E_{x\sim p_\theta}\left(\log p_\theta(x)\frac{\partial\log p_\theta(x)}{\partial\theta}\right)
    -E_{x\sim p_\theta}\left(\frac{\partial\log p_\theta(x)}{\partial\theta}\right) \\
&=& -\frac{1}{2}E_{x\sim p_\theta}\left(\frac{\partial}{\partial\theta}(\log p_\theta(x))^2\right) \\
\end{array}
$$

This means that $-\frac{1}{2}\frac{\partial}{\partial\theta}(\log p_\theta(x))^2$
is an unbiased estimator of $\frac{\partial H}{\partial\theta}$. Actually,
$-\frac{1}{2}\frac{\partial}{\partial\theta}(c+\log p_\theta(x))^2$ is an
unbiased estimator for any constant $c$.

For some distributions, the sample of $p_\theta$ is generated by transforming
$\epsilon \sim q$ by $f_\theta(\epsilon)$, where $q$ is a fixed distribution and
$f_\theta$ is a smooth bijective mapping. $p_\theta(x)$ is implicitly defined by
$q$ and $f_\theta$ as:

$$
\begin{equation*}
p_\theta(x) = q(f_\theta^{-1}(x)) / \left|\det \left.
  \frac{\partial f_\theta(\epsilon)}{\partial\epsilon}\right|
  _{\epsilon=f_\theta^{-1}(x)}\right|
\end{equation*}
$$

Interestingly, when calculating $-\frac{\partial\log p_\theta(x)}{\partial\theta}$,
if we treat $x$ as $x=f_\theta(\epsilon)$, we get an unbiased estimator of
$\frac{\partial H}{\partial\theta}$:

$$
\begin{array}{ll}
&& E_{x\sim p_\theta}\left(-\frac{\partial\log p_\theta(x)}{\partial\theta}\right)
   = E_{\epsilon \sim q}\left(-\frac{\partial\log p_\theta(f_\theta(\epsilon))}{\partial\theta}\right) \\
&=& -\frac{\partial}{\partial\theta}E_{\epsilon \sim q}\left(\log p_\theta(f_\theta(\epsilon))\right)
    = -\frac{\partial}{\partial\theta}E_{x \sim p_\theta}\left(\log p_\theta(x)\right)
    = \frac{\partial}{\partial\theta}H(p)
\end{array}
$$

So we can use $-\frac{\partial\log p_\theta(x)}{\partial\theta}$ as an unbiased
estimator of $\frac{\partial H(p)}{\partial\theta}$ if $x=f_\theta(\epsilon)$
and we allow gradient to propagate through $x$ to $\theta$.
"""
#%%
