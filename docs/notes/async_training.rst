Asynchronous training guide
===========================

We compare synchronous off-policy training *vs.* asynchronous off-policy
training on PyBullet `Humanoid
<https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_data/humanoid>`_
using `PPO2 <https://arxiv.org/abs/1707.06347>`_. Through this example, we want
to show how *training* *throughput* can affect the convergence speed in a
positive way, and how to effectively increase the throughput (up to 5 times) by
adopting the asynchronous setting. We will draw conclusions on possible
scenarios that asynchronous training is most suitable for.

By using asynchronous training and optimizing the throughput, I was able to
decrease the convergence time to 1h 40mins (from the original 6h) on humanoid.

.. image:: ../../alf/examples/async_ppo_bullet_humanoid.gif
    :width: 200

Disclaimer: You can consider this doc as a general guide of setting up async
training, but be aware that ideas in this doc might not apply to any RL problem.

Background
----------

Let :math:`K` be the batch size of the batched environment. Suppose there are
:math:`N` actors, then there are :math:`KN` environments in total, with each
actor acting on a batched environment of size :math:`K`. For sync training,
:math:`N=1`. For async training, we will use :math:`N=2` in the experiments. We
always have a single learner in both cases, and the learner can only learn from
trajectories of one actor at a time. All other hyperparameters are the same.

The difference between sync and async training is whether the parameter update
and environment rollout can happen simultaneously. The definition of *sync
training* is that prediction and learning are intervened like below:

::

    pred -> learn -> pred -> learn → ...

On the contrary, *async training* decouples them in the following way:

::

    pred → pred → pred → ...   (several actors running in parallel)
             |
    (synchronize periodically)
             |
    learn → learn → learn → ... (one learner)

The learner waits for a batch of trajectories to be collected by an actor,
updates parameters, and waits for the next batch (probably from a different
actor), and so on.

To recap, PPO uses a batch of trajectories only once (but may update parameters
many times on it). So the total number of environment steps is always equal to
the total number of training samples. This data pipeline is similar to an
on-policy setting, where no experience is replayed.

Difference with IMPALA
----------------------

Our pipeline is almost the same with `IMPALA
<https://arxiv.org/abs/1802.01561>`_. However, in IMPALA, all the environments
are independent: they all run in parallel without being tied together (yet it’s
possible that actor numbers < env numbers). In our case, we have a hierarchy:
:math:`N` batched environments, each of size :math:`K`. This could potentially
decrease the communication overhead.

Throughput EPS
--------------

Suppose the average time interval between every two training iterations is
:math:`T`. Given all the other parameters fixed (e.g., rollout length), we
define the EPS (environments per seconds) as the training throughput:
:math:`\frac{K}{T}`.  Note that EPS is independent of :math:`N`, i.e.,
irrelevant to how many actors we have in this particular problem. Intuitively,
EPS tells us how many steps (with a constant multiplier) per second the learner
sees. The figure below illustrates three different EPS values in sync and async
examples:

.. image:: ../images/throughput.jpg
    :width: 600
    :height: 200

In the above, :math:`K=32`. The EPSs for the three settings are
:math:`\frac{32}{10}=3.2, \frac{32}{9}=3.6, \frac{32}{6}=5.3`, respectively.
Even though EPS= :math:`\frac{K}{T}` doesn’t explicitly depend on :math:`N`,
:math:`T` might be affected by :math:`KN` because the rollout time of each actor
depends on the computational resource constraints (e.g., CPU/GPU, data
bandwidth, etc). As :math:`N` increases, :math:`T` might also increase due to
resource competition.

In our PPO experiments, EPS can be a highly effective indicator of the
convergence speed: we would expect the algorithm to converge faster if EPS is
higher.

Optimize EPS
------------

To verify if a higher EPS leads to a faster convergence, we tried different
values of :math:`K`, and report their corresponding :math:`T` (in seconds) below:

==== ============== ================ ===================================================
K    sync (T, EPS)  async (T, EPS)   EPS ratio: (async/sync)
==== ============== ================ ===================================================
32	 (10, 3.2)	    (6, 5.3)	     1.6
48	 (13, 3.7)	    (7, 6.9)	     1.9
96   (20, 4.8)      (6.5, 14.8)      3.1
==== ============== ================ ===================================================

We can see that with :math:`K=96`, the EPS ratio reaches a maximum value of 3.1.
And for either async or sync training, a greater :math:`K` increases the EPS.
The EPS of async :math:`K=96` is about 5 times that of sync :math:`K=32`.

Experiments
-----------

In the below, we report the actual training curves for 4 data points in the
above table, and see if convergence speed is proportional to EPS. For all runs,
we turned off the training summary code.

sync K=32
^^^^^^^^^

EPS=3.2;  5h 40mins (33M steps) to reach 2k training score

.. image:: ../images/ppo_bullet_humanoid_sync_K32.png
    :width: 300
    :height: 200

async K=32
^^^^^^^^^^

EPS=5.3;  4h 20mins (40M steps) to reach 2k training score

.. image:: ../images/ppo_bullet_humanoid_async_K32.png
    :width: 300
    :height: 200

sync K=96 (:code:`alf/examples/ppo_bullet_humanoid.gin`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

EPS=4.8;  3h 20mins (29M steps) to reach 2k training score

.. image:: ../images/ppo_bullet_humanoid_sync_K96.png
    :width: 300
    :height: 200

async K=96 (:code:`alf/examples/async_ppo_bullet_humanoid.gin`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

EPS=14.8; 1h 40mins (46M steps) to reach 2k training score

.. image:: ../images/ppo_bullet_humanoid_async_K96.png
    :width: 300
    :height: 200

Results analysis
----------------

We can see that the convergence speed is roughly proportional to EPS. However,
generally async training is less sample efficient than sync training. For
example, sync :math:`K=96` is roughly 1.5x (46M/29M) sample efficient than async
:math:`K=96`, which results in 1.5x longer training time for async :math:`K=96`
(otherwise it would be 100mins/1.5=66 mins assuming the same sample efficiency).
There might be two reasons for the lower sample efficiency in async training:

1. Due to the lag between the rollout policy and the learning policy, async
training always predicts with out-dated policies. So there are at most 100%
redundant steps from the perspective of performance logging.

2. PPO assumes that at the beginning of each training iteration, the behavior
policy and the training policy are the same. But this is not the case in async
training, which may make the algorithm itself less effective.

Different batch size
--------------------

What if we set :math:`K=192` for sync training? In this case, the training batch
size would be twice as sync :math:`K=96`, but the number of environments will be
the same with async :math:`K=96`. Note that the mini-batch size would still keep
unchanged (4096 in this case), and just the number of mini batches doubles.

sync K=192
^^^^^^^^^^

EPS=7.0; 3h 30mins (45M steps) to reach 2k training score

.. image:: ../images/ppo_bullet_humanoid_sync_K192.png
    :width: 300
    :height: 200

It might be surprising to see that the convergence speed is no better than sync
:math:`K=96` even though the EPS is about 1.5 (7.0/4.8) higher. In other words,
sync :math:`K=192` is 1.5x less sample-efficient than sync :math:`K=96`.  This
indicates that the PPO algorithm optimizes the target policy referring to the
old policy on an unnecessarily big dataset. So generally, EPSs with different
training batch sizes are not comparable indicators of convergence. If we'd like
to keep the batch size at a small value but want to increase EPS, then async
training is recommended.

Conclusion
----------

If possible, we want to minimize the time interval :math:`T` between every two
training iterations to maximize the throughput EPS (assuming :math:`K` fixed).
Generally this can be achieved by using more than one actors (:math:`N > 1`)
running asynchronously with the training. However, the :math:`T` must be
verified empirically as more actors might have resource competitions.

Assuming abundant CPU resources, we can imagine that async training is most
suitable for problems with simple neural models but complex environment
simulations (3D rendering, physics, etc) if the bottleneck is simulation speed.
In this case the rollout time is much greater than the training time, and having
multiple actors (ideally without comprising each actor's speed) in the data
pipeline can decrease the waiting time between two training iterations.

Because async training generally is less sample efficient than sync training,
it’s recommended to use it for cases where sample efficiency is not the main
metric, e.g., to have faster turn-around times for tweaking model
hyperparameters.

Another great benefit of async training is when sometimes we want to train on a
large number of parallel environments with a large unroll length per training
iteration. For both on-policy and sync off-policy training, we have to maintain
a huge computational graph during each training update. This can cause the
out-of-memory issue on a GPU. With async off-policy training, we could
effectively have the same environment batch size and unroll length by splitting
the environment batch size into several smaller ones (i.e., with :math:`K`
actors, each actor having :math:`\frac{B}{K}` environments, but only need to
main a computational graph of size :math:`\frac{1}{K}` during training updates).
