A minimal working example
=========================

We start with a minimal working example of ALF. The example, as a pure ALF
configuration file, is located at ``<ALF_ROOT>/alf/examples/tutorial/minimal_example_conf.py``,
and consists of only 8 lines.

Train and play
--------------

Let's ignore its content for a moment (see the next section
:doc:`./understanding_ALF_via_the_minimal_working_example` for an explanation of
the configuration content), and just focus on how to launch the training,
interpret the output training messages, and play a trained model.

Train from scratch
^^^^^^^^^^^^^^^^^^

We can train from scratch by

.. code-block:: bash

    cd <ALF_ROOT>/alf/examples/tutorial
    python -m alf.bin.train --root_dir /tmp/alf_tutorial1 --conf minimal_example_conf.py

assuming ``/tmp/alf_tutorial1`` doesn't exist or is empty. The training will finish
in several seconds, but with some informative messages shown in the terminal. First
of all, you should see a message from ``checkpoint_utils.py`` like

::

    There is no checkpoint in directory /tmp/alf_tutorial1/train/algorithm. Train
    from scratch

which basically confirms that the training is from scratch and all algorithm parameters
and states are randomly initialized. Also ``policy_trainer.py`` will output
message lines like

::

    minimal_example_conf.py -> alf_tutorial1: 0 time=0.046 throughput=5169.30

which has the formatting template as

::

    [conf_file_name] -> [training_root_dir]: [iteration_numer] time=[current_time_per_training_iter] throughput=[current_training_throughput]

Finally, you will see

::

    Checkpoint 'ckpt-1' is saved successfully.

as the training finishes. Here we have the checkpoint numbered by the training
iteration, which is '1' because only one iteration is performed by this example.

Train from a checkpoint
^^^^^^^^^^^^^^^^^^^^^^^

By launching the same command again, this time the checkpoint messages are different.
First it should say

::

    Checkpoint 'ckpt-1' is loaded successfully.

which means the training is no longer from scratch, but instead reads the saved
checkpoint from the last run. By default ALF reads the most recent checkpoint in
a training root dir if multiple checkpoints exist. Also at the end of training,
``checkpoint_utils.py`` outputs:

::

    Checkpoint 'ckpt-2' is saved successfully.

It's important to understand that when training from a checkpoint, the numbering
will continue, i.e., the previous checkpoints won't be overwritten.

Play from a checkpoint
^^^^^^^^^^^^^^^^^^^^^^

Here we only introduce three basic usages of the ALF ``play`` module. For advanced
play (e.g., rendering customized model inference results, play from an ALF snapshot,
headless rendering, etc), we refer the reader to :doc:`./advanced_play_and_alf_snapshot`.

To play the trained model while rendering the environment on the screen,

.. code-block:: bash

    python -m alf.bin.play --root_dir /tmp/alf_tutorial1

By default, play will choose the most recent checkpoint for evaluation. If you
don't want to render, but just play to evaluate:

.. code-block:: bash

    python -m alf.bin.play --root_dir /tmp/alf_tutorial1 --norender

Or you can save the rendered result to a ``mp4`` video file:

.. code-block:: bash

    python -m alf.bin.play --root_dir /tmp/alf_tutorial1 --record_file /tmp/alf_tutorial1.mp4

We recommend the reader to read the various commandline flags in ``<ALF_ROOT>/alf/bin/play.py``,
for specifying different options such as checkpoint number and number of episodes to
evaluate.

Use Tensorboard for monitoring the training progress
----------------------------------------------------

While the training is ongoing, we can monitor the real-time progress by

.. code-block::

    tensorboard --logdir /tmp/alf_tutorial1

We leave the interpretation of various Tensorboard statistics to later sections.

Conclusion
----------

So far, we've talked about how to train a conf file and play the trained model,
with very basic options of ``train.py`` and ``play.py``. We really haven't explained
the content of the example conf file and ALF RL pipeline yet. In the next section,
we will try to view a rough picture of ALF through the lens of this minimal working
example.