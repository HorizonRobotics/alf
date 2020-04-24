``PyTorch`` Notes
************************************

This note is used for doucmenting some useful ``PyTorch``-related features,
which can serve as a useful reference for future relevent development, either
implementing algorithms in ``PyTorch`` or converting some eixsting code from
``TensorFlow`` to ``PyTorch``.

Many of these aspects are encountered during the conversion of ``ALF`` from
``TensorFlow`` to ``PyTorch``. Some of them are useful features to use, while
others can lead to deteriorated algorithm performance and may take some time to
figure out the root causes.


1. ``Transform`` / ``Bijector``
===============================
Description
------------------
`Transform <https://pytorch.org/docs/stable/distributions.html#module-torch.distributions.transforms>`_
(`Bijector <https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/Bijector>`_
for ``TensorFlow``) can be used to represent any differentiable and injective
(one to one) function defined on an open subset of
:math:`R^n`. It is typically used to transform a simple base Distribution
(*e.g.* `Normal <https://pytorch.org/docs/stable/distributions.html#normal>`_)
to a more complex distribution. One use case in  Reinforcement Learning
(``RL``) is the construction of the actor distribution (*e.g.* the
`NormalProjectionNetwork <https://github.com/HorizonRobotics/alf/blob/4391f3fbdc5d69da5cf1d721565fad9f8b2d4104/alf/networks/projection_networks.py#L80>`_
used in `SAC <https://github.com/HorizonRobotics/alf/blob/pytorch/alf/examples/sac_pendulum.gin>`_).

However, in practice, due to numerical limitation, some ``Transforms`` may not be
fully invertable with a naive implementation of the ``inverse`` function, which
could potentially lead to deteriorated algorithm performance.
``TensorFlow`` always use cache-based lookup for better inversion.
``PyTorch`` however does not use cache for inversion  by default, leading to a
default behavior that is different from ``TensorFlow``.  This is a potential
source of error when converting TensorFlow algorithms into PyTorch ones.
We need to explicitly turning it on in ``PyTorch`` when inversion is a problem.

Example
---------
We encountered issues related to this feature in our ``SAC`` algorithm converted
to  ``PyTorch`` from ``TensorFlow``.
Our initial version of ``SAC`` has an instability issue, expecially on complex
tasks that requires longer training time. This is caused by the inversion of the
``Transform``  used in the actor distribution of ``SAC`` as mentioned above.
For more details, please refer to this `PR on stablizing SAC <https://github.com/HorizonRobotics/alf/pull/486>`_.

2. ``Distribution`` / ``Probability``
=========================================================
Description
------------------
`PyTorch.Distribution <https://pytorch.org/docs/stable/distributions.html>`_
generally follows the design of the
`TensorFlow.Distribution paper <https://arxiv.org/pdf/1711.10604.pdf>`_
which is termed as `TensorFlow.Probability <https://www.tensorflow.org/probability>`_
in the ``TensorFlow`` package.
One functional difference which is very relevent to ``RL`` is that there is no
``mode`` function implemented in ``PyTorch.Distribution`` currently.
This is true both for the base distribution as well as for the transformed
distributions.
As ``mode``-based sampling is commently used to generate the greedy action from
the policy in ``RL``, one need to have a way to conduct this type of sampling.


Example
---------
We implement ``mode`` sampling for some common base distributions.
For the transformed distributions, we compute its mode by transforming the mode
of its base distribution for now, which may not be the actual mode.
For more details, please refer to this `PR on mode sampling
<https://github.com/HorizonRobotics/alf/commit/e3d53f567dedd3ade25f9de78432320e386d3af5>`_.



3. ``Optimizer``
===================
Description
------------------
`Adam optimizer <https://arxiv.org/pdf/1412.6980.pdf>`_ is a stochastic gradient
descent method that is based on adaptive estimation of first-order and
second-order moments and is widely used in ALF.

``PyTorch`` and ``TensorFlow``  implement the ``Adam`` optimizer differently.
``PyTorch`` mainly follows the procedure described in Algorithm 1 of the `Adam
paper <https://arxiv.org/pdf/1412.6980.pdf>`_.
``TensorFlow`` used a reformulation the update rules that are described at the
end of Section 2 of the paper.
The difference between the two can not be eliminated by matching
and setting hyper-parameters.

We implement an ``Adam`` optimizer in ``PyTorch`` following ``TensorFlow``'s way
of implementation. This is useful to eliminate the factor of optimizer when
matching the performance between ``PyTorch`` and ``TensorFlow`` implementations
of the same algorithm.

Another interface difference is that ``PyTorch`` optimizers require ``params``
as input at construction, which is less convenient in some cases (*e.g.* when
used together with ``gin.config``) than ``TensorFlow`` optimizers.

Example
---------
For more details on the ``Adam`` optimizer implemented following ``TensorFlow``'s
convention, please refer to this `PR on AdamTF <https://github.com/HorizonRobotics/alf/pull/466>`_.

For handling the required ``params`` at construction, we use a wrapper
to provide `default parameters <https://github.com/HorizonRobotics/alf/pull/508/files>`_.
We can also incorporate other features in a similar fashion (*e.g.*
`gradient clippling <https://github.com/HorizonRobotics/alf/commit/a9b07091dd208d9c5b14020146ea29245ffd2633>`_).



4. Inplace Operator
=====================
Description
-----------------
Inplace operator can be used in some cases to reduce memory usage.
However, inpalce operator should be used with care as suggested in the `PyTorch
Doc <https://pytorch.org/docs/master/notes/autograd.html#in-place-operations-with-autograd>`_.
A common scenario that is relatively safe for using inplace operator is the
activation functions.

Example
---------
``PyTorch`` supports inplace computation for some commonly used
activations such as ``ReLU``, but not all of them.
For exmaple on implementing an inplace activation function, please refer to this
`PR on inplace Softsign <https://github.com/HorizonRobotics/alf/pull/544>`_.






