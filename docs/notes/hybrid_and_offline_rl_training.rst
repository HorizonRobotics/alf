Hybrid and Offline RL Training
================================

This is a document describing the usage of the Offline and Hybrid RL training
capability of ALF.


1. RL Training Modes in ALF: Online, Offline and Hybrid
--------------------------------------------------------
Here we only focus on the RL training functionality of ALF. For this, we
have several different training modes supported in ALF:
- Online training: the training involves online interaction with the environment
  and all the data required for training is obtained through interaction with
  with the environment. Typical RL algorthms including both on-policy and
  off-policy RL methods fall into this category.
- Offline training: there is no online interaction with the environment, and all
  the data for training is pre-collected and provided as a fixed dataset.
  Representative methods in this category incldues Behavior Cloning methods and
  Offline RL methods.
- Hybrid training: a training modes that mixes the two modes above. It involves
  both offline data and online interaction for training.


2. Hybrid RL Training
-----------------------
The usage of the hybrid RL training is very straightforward, simply providing a
valid file path to ``offline_buffer_dir`` in the config on top of existing
methods (e.g. SAC), as long as the buffer contains all the information required
by the algorithm. How to generate the offline buffer is explained in Section 4.
How to implement a customized algorithm is detailed in Section 5.

For example:

.. code-block:: python

    alf.config( "TrainerConfig", offline_buffer_dir="/data_collection/train/algorithm/ckpt-replay_buffer" )

A runnable example can be found at ``/examples/hybrid_rl/hybrid_sac_pendulum.gin``.



3. Offline Buffer Structure
-------------------------------------------------
For generality, we assumes a minimal structure for the offlinereplay buffer,
which contains the essential information that is common for all RL algorithms.
Concretely, we assume the offline replay buffer has the structure that is
described by the following data spec:

.. code-block:: python
    offline_buffer_data_spec = Experience(
                                time_step=time_step_spec,
                                action=self._action_spec,
                                rollout_info=Info(
                                    rl=RLInfo(action=self._action_spec),
                                    rewards={},
                                    repr={},
                                ))

As can be observed, the offline replay buffer contains information including:
- time step: which is the ``TimeStep`` structure holding one-step transition
  information including previous action, observation, reward, discout
  and step type etc.
- action: the action taken at the current step
- rollout_info: assumes following the same structure as used in ``Agent``,
  and includes action information in the ``RLInfo``, which is required by
  ALF functions (e.g. ``train_step()``) in training.



4. Offline Buffer Generation
------------------------------
As long as the offline buffer file satisfies the requirements presented in
Section 3, it can be readily used in ALF.

There are many different ways to generate an offline replay buffer.

- One straightforward approach is to use the replay buffer checkpoint saved by
  ALF during RL training (typically named as ``ckpt-xxxx-replay_buffer``
  under ``/log_dir/train/algorithm/``).
  Note that the ``enable_checkpoint`` option for ``ReplayBuffer`` should be set
  as ``True`` to enable replay buffer checkpointing:

.. code-block:: python

    alf.config('ReplayBuffer', enable_checkpoint=True)


- The offline data buffer can also be generated with other types of agents,
  e.g. rule-based expert. For example, we provide a way to generate offline
  buffers containing demos from a scripted expert in CARLA:
  `CARLA Agents and offline data collection <https://github.com/HorizonRobotics/alf/pull/1160>`_



5. How to Implement New Offline and Hybrid RL Algorithms
------------------------------------------------------------
The implementation of new algorithms is also simple.
The minimal requirement is to implement two additional functions:
``train_step_offline()`` and ``calc_loss_offline()`` for customized offline
training and loss calculation based on the offline data provided as input to
the functions, and inherit all other functions from an existing method such as
SAC. Of course, these functions such as ``train_step()`` and ``calc_loss()``
can also be customized when necessary.


