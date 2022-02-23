Computing probabilities of a transformed distribution
=====================================================

Background
----------

Change of variable
^^^^^^^^^^^^^^^^^^

Given :math:`x\sim p_X` and :math:`y=g(x)` where :math:`g` is a bijective mapping,
the probability of the changed variable :math:`y` is

.. math::

    \begin{array}{rcl}
        p_Y(y) &=& p_X(g^{-1}(y))\Big|\text{det}\big[\frac{d g(x)}{d x}\Big|_{x=g^{-1}(y)}\big]\Big|^{-1}\\
               &=& p_X(g^{-1}(y)) h(g^{-1}(y))\\
    \end{array}

Note that the inverted absolute determinant of the Jacobian can just be treated as a
function :math:`h` of :math:`x`, and we would like to evaluate its value for
each particular value :math:`\hat{y}`, i.e., :math:`p_Y(\hat{y})=p_X(g^{-1}(\hat{y}))h(g^{-1}(\hat{y}))`.

The following two common cases involve computing the (log) probability of a changed
variable, but should be treated differently.

Policy gradient (PG)
^^^^^^^^^^^^^^^^^^^^

The policy gradient for training a policy :math:`\pi_{\theta}(a)` is

.. math::

    -\int_{a}\pi_{\theta}(a)\nabla_{\theta}\log\pi_{\theta}(a) A(a) da

where we omit the dependency on the observation :math:`s` for simplicity. It should
be noted that the above formula is an expectation of **gradient vectors**. If we
denote :math:`\mathbf{g}(\cdot)=\nabla_{\theta}\log\pi_{\theta}(\cdot)`, then policy
gradient tells us that for each sampled :math:`a`, we evaluate :math:`\mathbf{g}(a)`,
and compute the average. Or alternatively, we could first detach :math:`a` (it might
be re-parameterized) and then compute :math:`\nabla_{\theta}\log\pi_{\theta}(a)`.

For other objectives like MLE, a similar issue should be paid attention to.

Entropy gradient
^^^^^^^^^^^^^^^^

The gradient of entropy can be estimated by the re-parameterization trick:

.. math::

    \begin{array}{rl}
     &-\nabla_{\theta}\int_{a}\pi_{\theta}(a)\log\pi_{\theta}(a) da\\
    =&-\nabla_{\theta}\int_{\epsilon}p(\epsilon)\log\pi_{\theta}(f(\epsilon;\theta)) d\epsilon\\
    =&-\int_{\epsilon}p(\epsilon)\nabla_{\theta}\log\pi_{\theta}(f(\epsilon;\theta)) d\epsilon\\
    \end{array}

That is, the derivative takes place **after** the substitution. As a result, we should not
detach the gradient of :math:`f(\epsilon;\theta)` when computing the log probability
in this case. Also see the analysis in :doc:`./estimating_derivative_of_expectation`.

In the PG scenario, it is much easier for us to make mistakes of forgetting to detach
sampled :math:`a`, thus we will discuss it in depth below.

What is the essential difference between detach and no detach of :math:`y` for PG?
----------------------------------------------------------------------------------

Back to the general case :math:`y=g(x)`, if we just want to compute the probability without
taking the derivative w.r.t. some parameters of :math:`g`, then whether detach :math:`y` or
not does not matter. Namely, the following two values are equal:

.. math::

    \begin{array}{rclr}
        p_Y(y) &=& p_X(g^{-1}(y)) h(g^{-1}(y)) & \text{(detach)} \\
               &=& p_X(g^{-1}(g(x)))h(g^{-1}(g(x)))=p_X(x)h(x) & \text{(no detach)}\\
    \end{array}

However, when :math:`g` contains some **trainable parameters** we'd like to optimize,
whether to detach :math:`y` or not should strictly depend on the gradient formula.
Otherwise, the gradient might be incorrect. For example, the second form :math:`p_X(x)h(x)`
might wipe out some parameters because :math:`g^{-1}` is not included.

An example
^^^^^^^^^^

For a simple example, if :math:`x\sim\mathcal{N}(0,1)`, :math:`y=g(x;\sigma,\mu)=x\sigma + \mu`,
then with detached :math:`y`

.. math::

    \begin{array}{rcl}
        p_Y(y) &=& p_X(g^{-1}(y)) \Big|\frac{\partial g(x)}{\partial x}\big|_{x=g^{-1}(y)}\Big|^{-1} \\
             &=& p_X(\frac{y-\mu}{\sigma}) \sigma^{-1}\\
             &=& \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}\\
    \end{array}

which is the correct p.d.f. for training :math:`\mu` with PG. However, with
:math:`y`` undetached:

.. math::

    \begin{array}{rcl}
        p_Y(y) &=& p_X(x) \Big|\frac{\partial g(x)}{\partial x}\big|_{x=x}\Big|^{-1} \\
               &=& \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{x^2}{2}}\\
    \end{array}

with :math:`x=g^{-1}(g(x))`, not including :math:`\mu` in the resulting p.d.f.
Thus PG won't be able to adjust :math:`\mu`.

Transform caches
^^^^^^^^^^^^^^^^

A cache is usually helpful for a bijective transform where the inverse is computationally
expensive or numerically unstable. The way PyTorch uses a cache is that for an
inquery :math:`y`, it checks whether it has the same object id with :math:`y_{old}`
from the cached pair :math:`(x_{old}, y_{old})`. If yes, :math:`x_{old}` is returned.
Any out-of-place operation (e.g., ``detach()``) that makes :math:`y` a different
object will invalidate the cache, and an inverse has to be computed.

If a transform :math:`g` has a cache turned on and :math:`y` is undetached, it
will circumvent the computation of :math:`g^{-1}` and retrieves
:math:`x(\equiv g^{-1}(g(x)))` directly for numerical stability. Again, PyTorch
transform cache only works when :math:`y` is not detached; the role of a cache is
no more than solving the numerical issue for an undetached :math:`y`.

.. list-table::

    * -
      - Cache on
      - Cache off
    * - :math:`y` detached
      - :math:`x \leftarrow g^{-1}(y)` (*)
      - :math:`x \leftarrow g^{-1}(y)` (*)
    * - :math:`y` undetached
      - :math:`x \leftarrow x`
      - :math:`x \leftarrow g^{-1}(g(x))` (*)

'*' means potential numerical issues when inverting.

A composition of transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose :math:`g` is now a composition of transforms
:math:`g=g_N\circ \ldots g_k \ldots \circ g_1`, where :math:`g_k` is the first
(from left to right) transform that contains trainable parameters. Then for PG,
it is completely fine to do either of the following things:


1. Simply detach :math:`y` and invalidate all caches.

2. Don't detach :math:`y`, turn on caches for transforms :math:`g_N, \ldots, g_{k+1}`,
   and detach the input :math:`g^{-1}_{k+1}\circ \ldots \circ g^{-1}_N(y)`
   to :math:`g_1^{-1}\circ \ldots \circ g^{-1}_k`.

PyTorch's ``sample()`` and ``rsample()``
----------------------------------------

So far we've assumed that :math:`p_X(\cdot)` does not contain any trainable parameter,
and all model parameters are in the transforms. This is in fact a very general and
valid assumption, and it will work well in practice. However, PyTorch's implementation
is different by assuming :math:`p_{\theta}(x)`, which can be treated as merging
the original :math:`p_X` with the first parameterized transform :math:`g_1(\circ;\theta)`.
This is convenient if there is a closed form p.d.f. of :math:`p_X(\cdot;\theta)` (e.g.
:math:`\mathcal{N}(\mu,\sigma)`). For this parameterized distribution, PyTorch introduces
two functions of ``sample()`` and ``rsample()``. We can simply assume that

.. list-table::

    * - ``rsample()`` of :math:`p_{\theta}(x)`
      - :math:`x\sim p_X, y_1=g_1(x;\theta)`
    * - ``sample()`` of :math:`p_{\theta}(x)`
      - additional detach on the rsampled :math:`y_1`

When we call ``sample()`` (``rsample()``) on the transformed distribution :math:`p_Y(y)`,
PyTorch will first call ``sample()`` (``rsample()``) of :math:`p_{\theta}(x)`, and
apply the remaining transforms. This means that even with ``sample()``, the final
:math:`y=g_N\circ\ldots\circ g_2` itself is not detached by default! This might cause
errors when it is directly used for computing PG w.r.t. parameters in :math:`g_2,\ldots,g_N`
(e.g., normalizing flow transforms). Thus for PG, no matter whether an action is
sampled by ``rsample()`` or ``sample()``, the safest way is to always detach it
before computing the probability.

However, if you are certain that there is no trainable parameter in :math:`g_2,\ldots,g_N`,
then detach is not necessary for ``sample()``. But detach is still necessary for
``rsample()`` because of :math:`g_1` (:math:`p_{\theta}(x)`) parameters.

SAC and DDPG are ``rsample`` safe by nature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For non policy gradient methods like SAC and DDPG, it's safe to use undetached action
from ``rsample()`` anywhere in the code and have all transform caches turned on.
The only place they need to compute its log probability is when estimating the
entropy gradient. But as explained above, in this case, the action must not be
detached.

Action transformations as environment wrappers
----------------------------------------------

ALF's AC and PPO algorithms always detach the action from ``sample()`` for PG loss,
without checking if the transforms have trainable parameters or not. This simplicity
invalidates caches and sometimes causes numerical issues even when all the transforms
do not have trainable parameters.

If we know the transforms are not trainable, then a better way is that we don't
detach the action but exploit the cache for PG to avoid inverting. When the transform
:math:`g=g_N\circ\ldots\circ g_2` has no trainable parameters
(e.g., ``StableTanh``), we have parameters only exist in :math:`p_{\theta}(x)`. It
follows

.. math::

    \begin{array}{rcl}
        \log p_Y(y) &=& \log p_{\theta}(g^{-1}(y)) - \log \Big|\frac{\partial g(x)}{\partial x}\big|_{x=g^{-1}(y)}\Big|\\
        \int_y P_Y(y)\nabla_{\theta}\log p_Y(y) dy &=& \int_y P_Y(y)\nabla_{\theta}\log p_{\theta}(g^{-1}(y)) dy\\
        &=&\int_x p_{\theta}(x) \nabla_{\theta}\log p_{\theta}(x) dx\\
    \end{array}

because we can discard the Jacobian determinant for :math:`\nabla_{\theta}`.
Thus in this case, regarding PG, it's equivalent to directly training
:math:`p_{\theta}(x)` in the untransformed action space :math:`X` and apply the
transformation on the **environment side**. If we do so, there is no longer an
instability issue associated with PPO and AC.

One caveat of applying nonparameterized transformations on the environment side is,
the actual entropy of environment actions is difficult to be estimated on the
algorithm side. One solution is to still have :math:`g` applied to :math:`p_{\theta}(x)`
for entropy calculation, while the PG loss directly uses :math:`p_{\theta}(x)`.
