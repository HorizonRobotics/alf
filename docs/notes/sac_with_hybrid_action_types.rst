Soft Actor-Critic with hybrid action types
==========================================

Actor training loss
-------------------

Let :math:`a` be the discrete action and :math:`b` the continuous action, then the actor loss is:

.. math::

    \begin{equation}
    \begin{array}{ll}
         &  \displaystyle\int_{a,b}\pi(a,b|s)[\alpha \log \pi(a,b|s) - Q(s,a,b)]\\
         =& \displaystyle\int_{a,b}\pi_{\phi}(b|s)\pi(a|b,s)[\alpha\log \pi_{\phi}(b|s) + \alpha \log \pi(a|b,s) - Q(s,a,b)]\\
         =& \displaystyle\int_b \pi_{\phi}(b|s) \left(\alpha\log \pi_{\phi}(b|s) - \int_a \pi(a|b,s) q_b(s,a)\right)\\
         =& \displaystyle\int_b \pi_{\phi}(b|s) \left(\alpha\log \pi_{\phi}(b|s) - \mathbb{E}_{\pi(a|b,s)}[q_b(s,a)]\right)\\
    \end{array}
    \end{equation}

where :math:`q_b(s,a):= Q(s,a,b)-\alpha\log\pi(a|b,s)`. Given any :math:`\pi_{\phi}(b|s)` for any :math:`s,b`, maximizing
the inner expectation we have

.. math::

    \pi^*(a|b,s)=\text{argmax}_{\pi}\mathbb{E}_{\pi(a|b,s)}[q_b(s,a)]=\frac{\exp(\frac{Q(s,a,b)}{\alpha})}{Z(s,b)}

as the optimal conditional policy for action :math:`a`. For a reasonable discrete action space, the optimal inner
expectation :math:`q^*(s,b):=\mathbb{E}_{\pi^*(a|b,s)}[\cdot]` can be easily computed. It's a function of :math:`s`
and :math:`b`, and differentiable w.r.t. :math:`b`. Thus we can still use re-parameterization trick
:math:`b=g_{\phi}(\epsilon,s)` to optimize :math:`\phi`:

.. math::

    \mathbb{E}_{\epsilon\sim p(\epsilon)}\left[\alpha \log\pi_{\phi}(g_{\phi}(\epsilon,s)|s) - q^*(s,g_{\phi}(\epsilon,s))\right]


Different entropy coefficients
------------------------------

To have different entropy coefficients, the actor loss becomes

.. math::

    \displaystyle\int_{a,b}\pi_{\phi}(b|s)\pi(a|b,s)[\alpha_b\log \pi_{\phi}(b|s) + \alpha_a \log \pi(a|b,s) - Q(s,a,b)]

And accordingly, the value definition is changed to

.. math::

    \displaystyle V(s)=\mathbb{E}_{a,b\sim \pi}[Q(s,a,b)-\alpha_a \log \pi(a|s,b) - \alpha_b \log\pi_{\phi}(b|s)]